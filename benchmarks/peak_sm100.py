"""Measures the sustained rate of `tcgen05.mma.cta_group::2` M256 N256 K32 FP8 MMAs on operands in shared memory,
with one CTA pair per two SMs."""

import torch
from cuda.bindings import driver

import timing
from zig_cu_kernels import runtime

STAGES = 4
STAGE_BYTES = 2 * 128 * 128
SCALE_STAGE_BYTES = 3 * 512
FLOPS_PER_BATCH = STAGES * 4 * 2 * 256 * 256 * 32
BATCHES = 4096
SEED = 1
KINDS = {"f8f6f4": "mma_rate_sm100_f8f6f4", "mxf8f6f4": "mma_rate_sm100_mxf8f6f4"}


def operands(kind, random):
    generator = torch.Generator().manual_seed(SEED)
    count = 2 * STAGES * STAGE_BYTES
    if random:
        values = torch.randn(count, generator=generator).to(torch.float8_e4m3fn).view(torch.uint8)
        scales = torch.randint(124, 131, (2 * STAGES * SCALE_STAGE_BYTES,), generator=generator).to(torch.uint8)
    else:
        values = torch.zeros(count, dtype=torch.uint8)
        scales = torch.full((2 * STAGES * SCALE_STAGE_BYTES,), 127, dtype=torch.uint8)
    pointers = (values.cuda(), scales.cuda()) if kind == "mxf8f6f4" else (values.cuda(),)
    return pointers


def active_clusters(kernel, info, stream):
    """Returns the number of CTA pairs that the device runs at once, or None when the driver does not say."""
    config = runtime.launch_config((2, 1, 1), (info.threads, 1, 1), info.shared_bytes, stream, (2, 1, 1), False)
    try:
        status, count = driver.cuOccupancyMaxActiveClusters(driver.CUfunction(int(kernel.handle)), config)
    except (TypeError, RuntimeError):
        return None
    return count if status == driver.CUresult.CUDA_SUCCESS else None


def rate(kind, random):
    kernel, info = runtime.load(KINDS[kind])
    stream = torch.cuda.current_stream().cuda_stream
    sms = runtime.num_sm(runtime.current_device().device_id)
    clusters = active_clusters(kernel, info, stream) or sms // 2
    tensors = operands(kind, random)
    signature = "p" * len(tensors) + "u"
    values = (*(tensor.data_ptr() for tensor in tensors), BATCHES)

    def run():
        runtime.run(kernel, (2 * clusters, 1, 1), info.threads, info.shared_bytes, signature, values, stream,
                    cluster=(2, 1, 1))

    measurement = timing.measure(run)
    return clusters * BATCHES * FLOPS_PER_BATCH / measurement.seconds, clusters, measurement


def report():
    device = runtime.current_device().device_id
    print(f"tcgen05 peak: {torch.cuda.get_device_name()}, {runtime.num_sm(device)} SMs", flush=True)
    for kind, name in KINDS.items():
        reason = runtime.unavailable_reason(name)
        if reason is not None:
            print(f"tcgen05 {kind} skipped: {reason}", flush=True)
            continue
        for random in (True, False):
            flops, clusters, measurement = rate(kind, random)
            operand = "random" if random else "zeros"
            print(f"tcgen05 {kind:8} {operand:6} {flops * 1e-12:7.1f} TF/s  {clusters} CTA pairs  "
                  f"{timing.clock_note()}", flush=True)


if __name__ == "__main__":
    report()
