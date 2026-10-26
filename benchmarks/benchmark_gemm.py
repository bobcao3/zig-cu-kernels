"""Measures the BF16 GEMM ops on products `MxNxK` with one epilogue each, and prints the percentages of the limits of
README, section Benchmarks. A gated product has N/2 output columns from a weight of N rows. `--baselines` adds the
PyTorch ops on the same tensors and the ratio of the kernel rate to the PyTorch rate. With `--baselines` and without
`--shapes`, the script times only `BASELINE_CASES`."""

import argparse
import itertools

import torch
import torch.nn.functional as F

import dse
import timing
from zig_cu_kernels import gemm as launcher
from zig_cu_kernels import roofline, runtime
from zig_cu_kernels.torch import gated_gemm, gemm

EPILOGUES = ("store", "bias_silu", "residual", "f32", "gated_silu")
DEFAULT_CASES = (
    ("4096x4096x4096", "store"),
    ("4096x4096x4096", "gated_silu"),
    ("128x4096x4096", "residual"),
    ("1x4096x4096", "store"),
)
BASELINE_CASES = (DEFAULT_CASES[0], DEFAULT_CASES[3])
PYTORCH_LAUNCHES = {"store": 1, "bias_silu": 2, "residual": 1, "f32": 1, "gated_silu": 3}


def shape_of(text):
    m, n, k = (int(size) for size in text.split("x"))
    return m, n, k


def runs(epilogue, m, n, k, plan):
    """Returns the kernel and PyTorch calls of one epilogue. Any value in `plan` selects the launchers."""
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if epilogue == "bias_silu" else None
    out = torch.empty(m, n, device="cuda", dtype=torch.float32 if epilogue == "f32" else torch.bfloat16)
    residual = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) if epilogue == "residual" else None
    direct = any(value is not None for value in plan.values())
    options = {"bias_silu": {"activation": "silu"}, "residual": {"residual": residual}}.get(epilogue, {})
    f = n // 2

    def kernel():
        stream = torch.cuda.current_stream().cuda_stream
        if epilogue == "gated_silu" and direct:
            launcher.gated_gemm(a, b[:f], b[f:], out[:, :f], stream=stream, **plan)
        elif epilogue == "gated_silu":
            gated_gemm(a, b[:f], b[f:], out=out[:, :f])
        elif direct:
            launcher.gemm(a, b, out, bias=bias, stream=stream, **options, **plan)
        else:
            gemm(a, b, bias=bias, out=out, out_dtype=out.dtype, **options)

    def pytorch():
        if epilogue == "store":
            torch.mm(a, b.T, out=out)
        elif epilogue == "bias_silu":
            F.silu(F.linear(a, b, bias))
        elif epilogue == "residual":
            torch.addmm(residual, a, b.T, out=out)
        elif epilogue == "f32":
            torch.mm(a, b.T, out_dtype=torch.float32)
        else:
            h = F.linear(a, b)
            F.silu(h[:, :f]) * h[:, f:]

    return kernel, pytorch


def rotated_runs(epilogue, m, n, k, plan):
    """Returns the calls of `runs` as `timing.Rotation`s over copies of all operands, so that each call reads DRAM."""
    _, read_bytes, write_bytes = work(epilogue, m, n, k)
    pairs = [runs(epilogue, m, n, k, plan) for _ in range(timing.rotation_copies(read_bytes + write_bytes))]
    return timing.Rotation(kernel for kernel, _ in pairs), timing.Rotation(pytorch for _, pytorch in pairs)


def work(epilogue, m, n, k):
    """Returns the FLOPs and the bytes read and written of one call that moves every operand through DRAM once."""
    columns = n // 2 if epilogue == "gated_silu" else n
    extra = {"bias_silu": 2 * columns, "residual": 2 * m * columns}.get(epilogue, 0)
    return 2 * m * n * k, 2 * (m + n) * k + extra, (4 if epilogue == "f32" else 2) * m * columns


def rate_note(epilogue, m, n, k, seconds, main_loop=False):
    flops, read_bytes, write_bytes = work(epilogue, m, n, k)
    return (f"{flops / seconds * 1e-12:6.1f} TF/s {(read_bytes + write_bytes) / seconds * 1e-9:5.0f} GB/s  "
            f"{timing.roofline_note(seconds, flops, read_bytes, write_bytes, main_loop=main_loop)}")


def product_plan(epilogue, m, n, k, plan):
    kind = {"f32": "store_f32", "gated_silu": "gated"}.get(epilogue, "store_bf16")
    columns = n // 2 if kind == "gated" else n
    raster = None if plan["raster"] is None else tuple(plan["raster"])
    return launcher.plan(runtime.current_device().device_id, kind, m, columns, k, plan["tiling"], raster,
                         plan["splits"])


def measured(function, sol_seconds, launches, graph):
    if graph:
        seconds = timing.graph_seconds(function, sol_seconds=sol_seconds, launches=launches)
        return seconds, timing.last_telemetry[2] if timing.last_telemetry else None
    measurement = timing.measure(function, sol_seconds, launches=launches)
    return measurement.seconds, measurement.temperature_c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", help="MxNxK; each shape runs every epilogue of --epilogues")
    parser.add_argument("--epilogues", nargs="+", choices=EPILOGUES)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--tiling", help="a tiling name such as 128x128x32 instead of the modeled choice")
    parser.add_argument("--raster", nargs=2, type=int, metavar=("PANEL", "BY_COLUMNS"))
    parser.add_argument("--splits", type=int, help="the number of parts of k instead of the modeled choice")
    parser.add_argument("--graph", action="store_true",
                        help="time the calls in CUDA graphs, which leave out the host time of each call")
    arguments = parser.parse_args()
    if arguments.shapes:
        cases = list(itertools.product(arguments.shapes, arguments.epilogues or EPILOGUES))
    else:
        defaults = BASELINE_CASES if arguments.baselines else DEFAULT_CASES
        cases = [case for case in defaults if arguments.epilogues is None or case[1] in arguments.epilogues]
    plan = {"tiling": arguments.tiling, "raster": arguments.raster, "splits": arguments.splits}
    timing.print_limits("bf16", main_loop=True)
    for text, epilogue in cases:
        m, n, k = shape_of(text)
        flops, read_bytes, write_bytes = work(epilogue, m, n, k)
        sol_seconds = roofline.sol_seconds(flops, read_bytes + write_bytes)
        chosen = product_plan(epilogue, m, n, k, plan)
        tiled = not launcher.is_few_rows(chosen.info)
        kernel, pytorch = rotated_runs(epilogue, m, n, k, plan)
        seconds, temperature_c = measured(kernel, sol_seconds, 1, arguments.graph)
        line = f"{text:>15} {epilogue:10} {rate_note(epilogue, m, n, k, seconds, tiled)}  {timing.clock_note()}"
        dse.log(f"gemm {epilogue} {text}", f"{flops / seconds * 1e-12:.2f}", "TFLOP/s", temperature_c=temperature_c)
        if arguments.baselines:
            baseline, _ = measured(pytorch, sol_seconds, PYTORCH_LAUNCHES[epilogue], arguments.graph)
            line += f"  torch {rate_note(epilogue, m, n, k, baseline)}  ratio {baseline / seconds:4.2f}"
        print(f"{line}  [{launcher.plan_name(chosen)}]", flush=True)
        del kernel, pytorch


if __name__ == "__main__":
    timing.run(main)
