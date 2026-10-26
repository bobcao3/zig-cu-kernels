"""Measures the device time per call of axpby in CUDA graphs against the DRAM limits of README, section Benchmarks.
The script labels a case L2-resident when its tensors fit in the L2 cache, and gives such a case no DRAM limit."""

import argparse

import torch

import dse
import timing
from timing import clock_note, graph_seconds, host_seconds, roofline_note
from zig_cu_kernels import roofline, runtime
from zig_cu_kernels.torch import axpby, axpby_

DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
HOST_COUNT = 4096
ALPHA, BETA = 0.5, -1.0
CASES = {
    "to_out": (2, 3, lambda x, y, out: lambda: axpby(ALPHA, x, BETA, y, out=out),
               lambda x, y, out: lambda: torch.add(y, x, alpha=ALPHA, out=out)),
    "in_place": (2, 2, lambda x, y, out: lambda: axpby_(ALPHA, x, BETA, y),
                 lambda x, y, out: lambda: y.add_(x, alpha=ALPHA)),
    "beta_zero": (1, 2, lambda x, y, out: lambda: axpby(ALPHA, x, 0.0, y, out=out),
                  lambda x, y, out: lambda: torch.mul(x, ALPHA, out=out)),
}


def measured(name, count, label, call, read_bytes, write_bytes, resident):
    seconds = graph_seconds(call, sol_seconds=roofline.sol_seconds(0, read_bytes + write_bytes), launches=1)
    rate = (read_bytes + write_bytes) / seconds
    if resident:
        bound = "L2-resident, so the DRAM limits do not apply"
    else:
        bound = roofline_note(seconds, read_bytes=read_bytes, write_bytes=write_bytes)
    print(f"{name:4} n={count:>10} {label:22} {seconds * 1e6:9.1f} us {rate * 1e-9:7.0f} GB/s  {bound} "
          f"{clock_note()}", flush=True)
    temperature_c = timing.last_telemetry[2] if timing.last_telemetry else None
    dse.log(f"axpby_{name} {label} n={count}", seconds * 1e6, "us", temperature_c=temperature_c)
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1 << 26], help="element counts")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["to_out", "beta_zero"])
    parser.add_argument("--baselines", action="store_true", help="also time the PyTorch call of each case")
    parser.add_argument("--host", action="store_true", help=f"also print the host time of one call on {HOST_COUNT}")
    arguments = parser.parse_args()
    timing.print_limits()
    l2_bytes = runtime.l2_bytes(runtime.current_device().device_id)
    for name in arguments.dtypes:
        for count in arguments.counts:
            tensor_bytes = count * DTYPES[name].itemsize
            operand_sets = []
            for case in arguments.cases:
                operands, tensors, ours, theirs = CASES[case]
                resident = tensors * tensor_bytes <= l2_bytes
                copies = 1 if resident else timing.rotation_copies(tensors * tensor_bytes)
                while len(operand_sets) < copies:
                    operand_sets.append([torch.randn(count, device="cuda", dtype=DTYPES[name]) for _ in range(3)])
                traffic = (operands * tensor_bytes, tensor_bytes, resident)
                call = timing.Rotation(ours(*operand_set) for operand_set in operand_sets[:copies])
                seconds = measured(name, count, case, call, *traffic)
                if arguments.baselines:
                    call = timing.Rotation(theirs(*operand_set) for operand_set in operand_sets[:copies])
                    torch_seconds = measured(name, count, f"{case} torch", call, *traffic)
                    print(f"{name:4} n={count:>10} {case:22} {100 * torch_seconds / seconds:9.0f}% of the PyTorch "
                          "speed", flush=True)
            del operand_sets, call
            torch.cuda.empty_cache()
    if arguments.host:
        for name in arguments.dtypes:
            x, y, out = (torch.randn(HOST_COUNT, device="cuda", dtype=DTYPES[name]) for _ in range(3))
            print(f"host time per call on {HOST_COUNT} {name:4} axpby to out "
                  f"{host_seconds(lambda: axpby(ALPHA, x, BETA, y, out=out)) * 1e6:5.1f} us, torch add to out "
                  f"{host_seconds(lambda: torch.add(y, x, alpha=ALPHA, out=out)) * 1e6:5.1f} us", flush=True)


if __name__ == "__main__":
    timing.run(main)
