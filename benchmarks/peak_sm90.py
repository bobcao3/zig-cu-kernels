"""Measures the sustained FP8 `wgmma` rate of the `mma_rate_sm90` kernels, one work group per SM, on operands in
shared memory: random normal values quantized to E4M3, and all zeros for comparison."""

import torch

import timing
from zig_cu_kernels import runtime

SHAPES = ((256, 2), (256, 3), (192, 2), (128, 2))
K_BLOCK = 128
MMAS_PER_ITERATION = 4
ITERATIONS = 1024


def kernel_name(n, warpgroups):
    return f"mma_rate_sm90_n{n}_w{warpgroups}"


def unavailable_reason():
    return runtime.unavailable_reason(kernel_name(*SHAPES[0]))


def operands(n, warpgroups, zero):
    count = (64 * warpgroups + n) * K_BLOCK
    if zero:
        return torch.zeros(count, dtype=torch.uint8, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(timing.MMA_RATE_SEED)
    values = torch.randn(count, device="cuda", generator=generator)
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def rate(n, warpgroups, zero=False, iterations=ITERATIONS):
    """Returns the FLOP rate and the `Measurement` of chains of `wgmma.mma_async.m64n<n>k32.f32.e4m3.e4m3`."""
    kernel, info = runtime.load(kernel_name(n, warpgroups))
    device_id = runtime.current_device().device_id
    work_groups = runtime.num_sm(device_id)
    shared_bytes = max(info.shared_bytes, runtime.shared_bytes_for(kernel, device_id, 1))
    data = operands(n, warpgroups, zero)
    out = torch.empty(work_groups * info.threads, device="cuda")
    values = (data.data_ptr(), out.data_ptr(), iterations)
    flops = work_groups * warpgroups * iterations * MMAS_PER_ITERATION * 2 * 64 * n * 32

    def run():
        runtime.run(kernel, (work_groups, 1, 1), info.threads, shared_bytes, "ppu", values,
                    torch.cuda.current_stream().cuda_stream)

    measurement = timing.measure(run)
    return flops / measurement.seconds, measurement


def report():
    reason = unavailable_reason()
    if reason is not None:
        print(f"wgmma e4m3 skipped: {reason}", flush=True)
        return
    cases = [(n, warpgroups, False) for n, warpgroups in SHAPES] + [(*SHAPES[0], True)]
    for n, warpgroups, zero in cases:
        flops, measurement = rate(n, warpgroups, zero)
        telemetry = "no clock or power samples" if measurement.clock_mhz is None else \
            f"SM clock {measurement.clock_mhz:4.0f} MHz  power {measurement.power_w:4.0f} W"
        print(f"wgmma m64n{n}k32 e4m3 {warpgroups} warpgroups {'zero' if zero else 'random normal':13} "
              f"{flops * 1e-12:7.1f} TF/s  {telemetry}  drift {measurement.drift:+.3f}", flush=True)


if __name__ == "__main__":
    report()
