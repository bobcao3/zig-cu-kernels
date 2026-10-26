"""Measures the device time per call of the FP8 GEMMs and quantizers against the limits of README, section Benchmarks.

Products with at most `GRAPH_ROWS` rows replay as CUDA graphs. A shape option replaces every default list. Without a
shape option, `--baselines` and `--e8m0` time only the first default GEMM shape, because a process has time for the
added cases of one shape only. `--baselines-only` moves the PyTorch calls of a large shape to a process of their own."""

import argparse

import torch

import dse
import timing
from timing import clock_note, graph_seconds, roofline_note, sustained_seconds
from zig_cu_kernels import roofline, runtime
from zig_cu_kernels.torch import (
    fp8_gated_gemm,
    fp8_gemm,
    quantize_fp8_rows,
    quantize_fp8_tensorwise,
    quantize_fp8_tiles,
)

E8M0 = torch.float8_e8m0fnu
SCALE_GROUP = 128
SHAPE_LISTS = ("shapes", "gated_shapes", "quantize_shapes")
DEFAULT_SHAPES = {"shapes": ["8192x8192x8192", "32x4096x4096"], "gated_shapes": [], "quantize_shapes": ["8192x8192"]}
EXTENDED_SHAPES = {"shapes": ["8192x8192x8192"], "gated_shapes": [], "quantize_shapes": []}
GRAPH_ROWS = 512
SPLIT_LAUNCHES = 2


def shape_of(text):
    return tuple(int(size) for size in text.split("x"))


def measured(label, call, flops, read_bytes, write_bytes, launches=1, graph=False, main_loop=False):
    sol_seconds = roofline.sol_seconds(flops, read_bytes + write_bytes, "e4m3")
    timed = graph_seconds if graph else sustained_seconds
    seconds = timed(call, sol_seconds=sol_seconds, launches=launches)
    throughput = f"{(read_bytes + write_bytes) / seconds * 1e-9:6.0f} GB/s"
    if flops:
        throughput = f"{flops / seconds * 1e-12:6.1f} TF/s {throughput}"
    note = roofline_note(seconds, flops, read_bytes, write_bytes, "e4m3", main_loop)
    print(f"{label:48} {seconds * 1e6:9.1f} us {throughput}  {note} {clock_note()}", flush=True)
    temperature_c = timing.last_telemetry[2] if timing.last_telemetry else None
    dse.log(f"fp8 {label}", seconds * 1e6, "us", temperature_c=temperature_c)
    return seconds


def rotated(make, tensors, distinct_bytes):
    """Returns a `timing.Rotation` of the calls that `make` returns for `tensors` and for clones of `tensors`."""
    copies = [tensors] + [tuple(tensor.clone() for tensor in tensors)
                          for _ in range(timing.rotation_copies(distinct_bytes) - 1)]
    return timing.Rotation(make(*copy) for copy in copies)


def scale_bytes(m, columns, k, mode, weights=1):
    if mode == "rowwise":
        return 4 * (m + weights * columns)
    element = 1 if mode.endswith("E8M0") else 4
    return element * (m + weights * -(-columns // SCALE_GROUP)) * (k // SCALE_GROUP)


def operands(m, n, k):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k**-0.5
    a, b = quantize_fp8_tensorwise(x)[0], quantize_fp8_tensorwise(w)[0]
    scales = {"rowwise": (torch.rand(m, 1, device="cuda") + 0.5, torch.rand(n, 1, device="cuda") + 0.5)}
    block_a, block_b = quantize_fp8_rows(x)[0], quantize_fp8_tiles(w)[0]
    scales["block128"] = (quantize_fp8_rows(x)[1], quantize_fp8_tiles(w)[1])
    scales["block128 E8M0"] = (quantize_fp8_rows(x, scale_dtype=E8M0)[1], quantize_fp8_tiles(w, scale_dtype=E8M0)[1])
    codes = {"rowwise": (a, b), "block128": (block_a, block_b), "block128 E8M0": (block_a, block_b)}
    return codes, scales


def benchmark_gemm(m, n, k, baselines, e8m0, kernels):
    codes, scales = operands(m, n, k)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    graph = m <= GRAPH_ROWS
    launches = SPLIT_LAUNCHES if graph else 1
    cases = [("rowwise", False), ("block128", False)]
    if not graph:
        cases += [("rowwise", True)]
    if e8m0:
        cases += [("block128 E8M0", False)]
    for mode, fast in cases if kernels else []:
        read_bytes, write_bytes = m * k + n * k + scale_bytes(m, n, k, mode), 2 * m * n

        def make(a, b, scale_a, scale_b, out, fast=fast):
            return lambda: fp8_gemm(a, b, scale_a, scale_b, out=out, use_fast_accum=fast)

        call = rotated(make, (*codes[mode], *scales[mode], out), read_bytes + write_bytes)
        label = f"gemm {m}x{n}x{k} {mode}{' fast' if fast else ''}"
        measured(label, call, 2 * m * n * k, read_bytes, write_bytes, launches, graph, main_loop=True)
    if baselines:
        read_bytes, write_bytes = m * k + n * k + scale_bytes(m, n, k, "rowwise"), 2 * m * n
        for fast in (False, True):
            def make(a, b, scale_a, scale_b, out, fast=fast):
                return lambda: torch._scaled_mm(a, b.t(), scale_a, scale_b.t(), out_dtype=torch.bfloat16, out=out,
                                                use_fast_accum=fast)

            call = rotated(make, (*codes["rowwise"], *scales["rowwise"], out), read_bytes + write_bytes)
            label = f"gemm {m}x{n}x{k} torch._scaled_mm rowwise{' fast' if fast else ''}"
            measured(label, call, 2 * m * n * k, read_bytes, write_bytes, 1, graph)


def benchmark_gated(m, f, k, baselines, e8m0, kernels):
    """Times the gated product with `f` output columns from `2 f` weight rows."""
    codes, scales = operands(m, 2 * f, k)
    out = torch.empty(m, f, device="cuda", dtype=torch.bfloat16)
    graph = m <= GRAPH_ROWS
    for mode in (["rowwise", "block128"] + (["block128 E8M0"] if e8m0 else [])) if kernels else []:
        split = f if mode == "rowwise" else f // SCALE_GROUP
        read_bytes, write_bytes = m * k + 2 * f * k + scale_bytes(m, f, k, mode, weights=2), 2 * m * f

        def make(a, b, scale_a, scale_b, out, split=split):
            return lambda: fp8_gated_gemm(a, b[:f], b[f:], scale_a, scale_b[:split], scale_b[split:], out=out)

        call = rotated(make, (*codes[mode], *scales[mode], out), read_bytes + write_bytes)
        measured(f"gated {m}x{f}x{k} {mode}", call, 4 * m * f * k, read_bytes, write_bytes,
                 SPLIT_LAUNCHES if graph else 1, graph, main_loop=True)
    if baselines:
        fused = torch.empty(m, 2 * f, device="cuda", dtype=torch.bfloat16)
        read_bytes, write_bytes = m * k + 2 * f * k, 2 * m * f

        def make(a, b, scale_a, scale_b, out):
            def scaled_mm_swiglu():
                torch._scaled_mm(a, b.t(), scale_a, scale_b.t(), out_dtype=torch.bfloat16, out=fused)
                torch.mul(torch.nn.functional.silu(fused[:, :f]), fused[:, f:], out=out)

            return scaled_mm_swiglu

        call = rotated(make, (*codes["rowwise"], *scales["rowwise"], out), read_bytes + write_bytes)
        measured(f"gated {m}x{f}x{k} torch._scaled_mm+silu*mul", call, 4 * m * f * k, read_bytes, write_bytes, 3,
                 graph)


def benchmark_quantizers(rows, columns, baselines, e8m0, kernels):
    """Times the quantizers on a BF16 input. The tensorwise quantizer reads the input again after it computes the
    amax, and the L2 cache can hold at most `runtime.l2_bytes` of the input between the two reads."""
    x = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    values = rows * columns
    second_read = max(0, 2 * values - runtime.l2_bytes(runtime.current_device().device_id))
    scale_groups = values // SCALE_GROUP
    tiles = -(-rows // SCALE_GROUP) * (columns // SCALE_GROUP)
    cases = {
        "rows": (quantize_fp8_rows, 2 * values, values + 4 * scale_groups, 1),
        "tiles": (quantize_fp8_tiles, 2 * values, values + 4 * tiles, 1),
        "tensorwise": (quantize_fp8_tensorwise, 2 * values + second_read, values, 3),
    } if kernels else {}
    if e8m0 and kernels:
        cases["rows E8M0"] = (lambda x: quantize_fp8_rows(x, scale_dtype=E8M0), 2 * values, values + scale_groups, 1)
        cases["tiles E8M0"] = (lambda x: quantize_fp8_tiles(x, scale_dtype=E8M0), 2 * values, values + tiles, 1)
    if baselines:
        def torch_rows(x):
            scale_groups_of_x = x.float().reshape(rows, -1, SCALE_GROUP)
            amax = scale_groups_of_x.abs().amax(-1, keepdim=True)
            scales = torch.where(amax == 0, 1.0, amax / 448)
            return (scale_groups_of_x / scales).clamp(-448, 448).to(torch.float8_e4m3fn), scales

        cases["torch rows"] = (torch_rows, 2 * values, values + 4 * scale_groups, 6)
    for label, (quantize, read_bytes, write_bytes, launches) in cases.items():
        call = rotated(lambda x, quantize=quantize: lambda: quantize(x), (x,), read_bytes + write_bytes)
        measured(f"quantize {rows}x{columns} {label}", call, 0, read_bytes, write_bytes, launches)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--shapes", nargs="*", help="GEMM shapes MxNxK")
    parser.add_argument("--gated-shapes", nargs="*", help="gated shapes MxFxK")
    parser.add_argument("--quantize-shapes", nargs="*", help="quantizer shapes ROWSxCOLUMNS")
    parser.add_argument("--baselines", action="store_true", help="also time the PyTorch calls (one shape per process)")
    parser.add_argument("--e8m0", action="store_true", help="also time E8M0 block scales (one shape per process)")
    parser.add_argument("--baselines-only", action="store_true", help="time only the PyTorch calls")
    arguments = parser.parse_args()
    baselines = arguments.baselines or arguments.baselines_only
    flags = (baselines, arguments.e8m0, not arguments.baselines_only)
    given = {name: getattr(arguments, name) for name in SHAPE_LISTS if getattr(arguments, name) is not None}
    if given:
        shapes = {name: given.get(name, []) for name in SHAPE_LISTS}
    else:
        shapes = EXTENDED_SHAPES if baselines or arguments.e8m0 else DEFAULT_SHAPES
    timing.print_limits("e4m3", main_loop=True)
    for text in shapes["shapes"]:
        benchmark_gemm(*shape_of(text), *flags)
    for text in shapes["gated_shapes"]:
        benchmark_gated(*shape_of(text), *flags)
    for text in shapes["quantize_shapes"]:
        benchmark_quantizers(*shape_of(text), *flags)


if __name__ == "__main__":
    timing.run(main)
