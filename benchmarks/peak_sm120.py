"""Measures the FP8 `mma.sync` rates of compute capability 12.x with the `mma_rate_sm120` kernels."""

import torch

import timing
from zig_cu_kernels import roofline, runtime

KINDS = ("f8f6f4", "mxf8f6f4")
FLOPS_PER_INSTRUCTION = 2 * 16 * 8 * 32
CHAINS = 8
WORDS_PER_LANE = CHAINS * 6 + 2
WORK_GROUPS_PER_SM = 2
ITERATIONS = 4096
SCALE_SPREAD = 3
SEED = 1


def kernel_name(kind):
    return f"mma_rate_sm120_{kind}"


def operands(distribution):
    """Returns the E4M3 operand words of each lane for each chain, then the E8M0 scale words of A and B."""
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    value_bytes = 32 * CHAINS * 6 * 4
    if distribution == "random":
        values = torch.randn(value_bytes, generator=generator, device="cuda").to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.randint(127 - SCALE_SPREAD, 128 + SCALE_SPREAD, (32 * 8,), generator=generator,
                               device="cuda", dtype=torch.uint8)
    else:
        values = torch.zeros(value_bytes, dtype=torch.uint8, device="cuda")
        scales = torch.full((32 * 8,), 127, dtype=torch.uint8, device="cuda")
    return torch.cat([values.view(32, CHAINS * 6 * 4), scales.view(32, 8)], dim=1).contiguous().view(torch.int32)


def rate(kind, distribution, duration=timing.MEASURE_SECONDS):
    kernel, info = runtime.load(kernel_name(kind))
    work_groups = runtime.num_sm(runtime.current_device().device_id) * WORK_GROUPS_PER_SM
    out = torch.empty(info.threads, device="cuda")
    words = operands(distribution)
    values = (out.data_ptr(), words.data_ptr(), ITERATIONS)
    flops = work_groups * info.threads // 32 * CHAINS * ITERATIONS * FLOPS_PER_INSTRUCTION
    try:
        sol_seconds = roofline.compute_seconds(flops, "e4m3")
    except ValueError:
        sol_seconds = None

    def run():
        runtime.run(kernel, (work_groups, 1, 1), info.threads, 0, "ppu", values,
                    torch.cuda.current_stream().cuda_stream)

    measurement = timing.measure(run, sol_seconds, duration=duration)
    return flops / measurement.seconds, measurement


def report(distributions=("random", "zero")):
    for kind in KINDS:
        reason = runtime.unavailable_reason(kernel_name(kind))
        if reason is not None:
            print(f"sm120 {kind:8} skipped: {reason}", flush=True)
            continue
        for distribution in distributions:
            flops, measurement = rate(kind, distribution)
            print(f"sm120 {kind:8} {distribution:6} {flops * 1e-12:7.1f} TF/s  {timing.clock_note()}", flush=True)
