"""Measures the device time per call of the RMS norm functions in CUDA graphs. Each result line shows the bytes that
one call must read and write per second, and the percentages of the limits of README, section Benchmarks."""

import argparse

import torch
import torch.nn.functional as F

import dse
import timing
from zig_cu_kernels import roofline
from zig_cu_kernels.torch import (
    dual_rms_norm,
    fused_add_adaptive_rms_norm,
    fused_add_adaptive_rms_norm_fp8_quant,
    fused_add_rms_norm,
    fused_add_rms_norm_fp8_quant,
    rms_norm,
    rms_norm_fp8_quant,
)

SHAPES = ["16384x4096"]
VARIANTS = ["plain", "f32", "residual", "fp8", "fp8_rowwise", "fp8_blockwise", "fp8_blockwise_e8m0",
            "residual_fp8_blockwise", "adaptive", "adaptive_fp8_blockwise", "dual"]
DEFAULT_VARIANTS = ["plain", "residual", "fp8_rowwise", "fp8_blockwise_e8m0", "adaptive", "dual"]
EPS = 1e-6
SAMPLES = 4
FP8_MAX = 448.0
SCALE_GROUP = 128
HOST_SHAPE = "4x1024"


def shape_of(text):
    rows, width = (int(size) for size in text.split("x"))
    return rows, width


def size(*tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def quantized_rows(y, scale_dtype=torch.float32):
    """Returns the E4M3 codes and the scales of the rows of `y`. An E8M0 scale is the least power of two at or above
    the FP32 scale."""
    amax = y.float().abs().amax(-1, keepdim=True)
    scale = torch.where(amax == 0, 1.0, amax / FP8_MAX)
    if scale_dtype == torch.float8_e8m0fnu:
        mantissa, exponent = torch.frexp(scale)
        scale = torch.ldexp(torch.ones_like(scale), exponent - (mantissa == 0.5).to(exponent.dtype)).to(scale_dtype)
    return (y.float() / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale


def quantized_blocks(y, scale_dtype=torch.float32):
    rows, width = y.shape
    q, scale = quantized_rows(y.view(rows, width // SCALE_GROUP, SCALE_GROUP), scale_dtype)
    return q.view(rows, width), scale


def runs(variant, rows, width):
    """Returns the call, the PyTorch sequence of the same steps, and the bytes that the call must read and write. The
    call rounds the normalized rows to BF16 before a BF16 weight multiplies them, and `F.rms_norm` skips that rounding,
    so the results of the two can differ."""
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    x, other = torch.randn(rows, width, **bf16), torch.randn(rows, width, **bf16)
    weight = torch.randn(width, **bf16)
    residual_out, y = torch.empty(rows, width, **bf16), torch.empty(rows, width, **bf16)
    q = torch.empty(rows, width, device="cuda", dtype=torch.float8_e4m3fn)
    row_scales = torch.empty(rows, 1, device="cuda")
    blockwise_scales = torch.empty(width // SCALE_GROUP, rows, device="cuda").t()
    blockwise_e8m0_scales = torch.empty(width // SCALE_GROUP, rows, device="cuda", dtype=torch.float8_e8m0fnu).t()
    if variant == "plain":
        return (lambda: rms_norm(x, weight, EPS, out=y), lambda: F.rms_norm(x, (width,), weight, EPS),
                size(x, weight), size(y))
    if variant == "f32":
        x32, weight32, y32 = x.float(), weight.float(), y.float()
        return (lambda: rms_norm(x32, weight32, EPS, out=y32), lambda: F.rms_norm(x32, (width,), weight32, EPS),
                size(x32, weight32), size(y32))
    if variant == "residual":
        def pytorch():
            torch.add(x, other, out=residual_out)
            return F.rms_norm(residual_out, (width,), weight, EPS)

        return (lambda: fused_add_rms_norm(x, other, weight, EPS, out=y, residual_out=residual_out), pytorch,
                size(x, other, weight), size(residual_out, y))
    if variant == "fp8":
        scale = torch.tensor(0.01, device="cuda")
        return (lambda: rms_norm_fp8_quant(x, weight, EPS, fp8_scaling="tensorwise", fp8_scale=scale, out=q),
                lambda: (F.rms_norm(x, (width,), weight, EPS).float() / scale).clamp(-FP8_MAX, FP8_MAX).to(q.dtype),
                size(x, weight), size(q))
    if variant in ("fp8_rowwise", "fp8_blockwise", "fp8_blockwise_e8m0"):
        scaling = "rowwise" if variant == "fp8_rowwise" else "block128"
        scales = {"fp8_rowwise": row_scales, "fp8_blockwise": blockwise_scales,
                  "fp8_blockwise_e8m0": blockwise_e8m0_scales}[variant]
        quantizer = quantized_rows if scaling == "rowwise" else quantized_blocks
        return (lambda: rms_norm_fp8_quant(x, weight, EPS, fp8_scaling=scaling, fp8_scale_dtype=scales.dtype, out=q,
                                           scales_out=scales),
                lambda: quantizer(F.rms_norm(x, (width,), weight, EPS), scales.dtype), size(x, weight),
                size(q, scales))
    if variant == "residual_fp8_blockwise":
        def pytorch():
            torch.add(x, other, out=residual_out)
            return quantized_blocks(F.rms_norm(residual_out, (width,), weight, EPS))

        return (lambda: fused_add_rms_norm_fp8_quant(x, other, weight, EPS, fp8_scaling="block128", out=q,
                                                     scales_out=blockwise_scales, residual_out=residual_out),
                pytorch, size(x, other, weight), size(residual_out, q, blockwise_scales))
    modulation = torch.randn(SAMPLES, 3 * width, **bf16)
    scales, shifts, gates = modulation.chunk(3, dim=-1)

    def adaptive_pytorch():
        h = x.view(SAMPLES, -1, width) + gates[:, None] * other.view(SAMPLES, -1, width)
        n = F.rms_norm(h, (width,), None, EPS)
        return (1 + scales[:, None]) * n + shifts[:, None], h

    if variant == "adaptive":
        return (lambda: fused_add_adaptive_rms_norm(x, other, gates, scales, shifts, eps=EPS, out=y,
                                                    residual_out=residual_out),
                adaptive_pytorch, size(x, other, modulation), size(residual_out, y))
    if variant == "adaptive_fp8_blockwise":
        return (lambda: fused_add_adaptive_rms_norm_fp8_quant(x, other, gates, scales, shifts, eps=EPS,
                                                              fp8_scaling="block128", out=q,
                                                              scales_out=blockwise_scales, residual_out=residual_out),
                lambda: quantized_blocks(adaptive_pytorch()[0].view(rows, width)), size(x, other, modulation),
                size(residual_out, q, blockwise_scales))
    branch_weight = torch.randn(width, **bf16)

    def dual_pytorch():
        h = x + F.rms_norm(other, (width,), branch_weight, EPS)
        return F.rms_norm(h, (width,), weight, EPS), h

    return (lambda: dual_rms_norm(x, other, branch_weight, weight, EPS, out=y, residual_out=residual_out), dual_pytorch,
            size(x, other, branch_weight, weight), size(residual_out, y))


def rotated_runs(variant, rows, width):
    """Returns the results of `runs` with the two calls as `timing.Rotation`s over copies of the operands."""
    call, pytorch, read_bytes, written_bytes = runs(variant, rows, width)
    copies = [(call, pytorch)] + [runs(variant, rows, width)[:2]
                                  for _ in range(timing.rotation_copies(read_bytes + written_bytes) - 1)]
    return (timing.Rotation(call for call, _ in copies), timing.Rotation(pytorch for _, pytorch in copies), read_bytes,
            written_bytes)


def measured(call, read_bytes, written_bytes):
    sol = roofline.sol_seconds(0, read_bytes + written_bytes)
    seconds = timing.graph_seconds(call, sol_seconds=sol, launches=1)
    rate = (read_bytes + written_bytes) / seconds
    return seconds, (f"{seconds * 1e6:8.1f} us {rate * 1e-9:6.0f} GB/s "
                     f"({timing.roofline_note(seconds, read_bytes=read_bytes, write_bytes=written_bytes)})")


launched = []


def record_launches():
    """Records the kernel and grid of every launch, so that each result line names the kernel that ran."""
    from zig_cu_kernels import rms_norm as module

    launch = module.Launch.launch

    def recording(plan, *arguments):
        launched.append((plan.kernel, plan.grid[0], plan.threads, plan.shared_bytes, plan.extra))
        return launch(plan, *arguments)

    module.Launch.launch = recording


def kernel_note():
    if not launched:
        return ""
    kernel, grid, threads, shared_bytes, extra = launched[-1]
    detail = f" {extra[0]} stages" if extra else ""
    return f"[{kernel} grid {grid} block {threads} smem {shared_bytes}{detail}]"


def override_geometry(in_flight_kib, chunks_per_thread):
    """Replaces the launch choices of `zig_cu_kernels.rms_norm` for design-space exploration."""
    from zig_cu_kernels import rms_norm as module

    if in_flight_kib is not None:
        limit = None if in_flight_kib == 0 else in_flight_kib * module.KIB
        module.in_flight_limit = lambda device_id, options, work_group_bytes: limit
    if chunks_per_thread is not None:
        def size_for(options, width, engine="registers", chunks=None):
            for entry in module.sizes(options.name, engine):
                if entry.chunks_per_thread == chunks_per_thread and width <= entry.max_width:
                    return entry
            raise ValueError(f"{options.name} has no size of {chunks_per_thread} chunks per thread for {width}")
        module.size_for = size_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=SHAPES, help="ROWSxWIDTH")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument("--baselines", action="store_true", help="also measure the PyTorch sequences")
    parser.add_argument("--host", action="store_true", help=f"also print the host time of one call on {HOST_SHAPE}")
    parser.add_argument("--in-flight-kib", type=int,
                        help="override the KiB of loads in flight per SM that the launch geometry targets; 0 removes "
                        "the limit")
    parser.add_argument("--chunks-per-thread", type=int, help="override the chunks per thread of the kernel size")
    parser.add_argument("--engine", choices=("auto", "registers", "bulk"), help="the engine of the RMS norm kernels")
    parser.add_argument("--bulk-threads", type=int, help="threads per work group of the bulk engine")
    parser.add_argument("--bulk-stages", type=int, help="stages of the bulk engine; 0 fills the shared memory")
    parser.add_argument("--bulk-work-groups-per-sm", type=int, help="resident work groups per SM of the bulk engine")
    arguments = parser.parse_args()
    override_geometry(arguments.in_flight_kib, arguments.chunks_per_thread)
    record_launches()
    from zig_cu_kernels import rms_norm as module
    for name in ("engine", "bulk_threads", "bulk_stages", "bulk_work_groups_per_sm"):
        if getattr(arguments, name) is not None:
            setattr(module.tuning, name, getattr(arguments, name))
    timing.print_limits()
    for text in arguments.shapes:
        rows, width = shape_of(text)
        if rows % SAMPLES and any(variant.startswith("adaptive") for variant in arguments.variants):
            raise SystemExit(f"the rows of {text} do not split into the {SAMPLES} samples of the adaptive variants")
        for variant in arguments.variants:
            call, pytorch, read_bytes, written_bytes = rotated_runs(variant, rows, width)
            seconds, line = measured(call, read_bytes, written_bytes)
            line = f"{text:>11} {variant:18} {line} {timing.clock_note()} {kernel_note()}"
            dse.log(f"rms_norm_{variant}", f"{(read_bytes + written_bytes) / seconds * 1e-9:.1f}", "GB/s",
                    axis="shape", value=text, temperature_c=(timing.last_telemetry or (None,) * 3)[2])
            if arguments.baselines:
                torch_seconds, torch_line = measured(pytorch, read_bytes, written_bytes)
                line += (f"\n{'':>11} {'torch':18} {torch_line} {timing.clock_note()}; the kernel runs at "
                         f"{100 * torch_seconds / seconds:.0f}% of the torch speed")
            print(line, flush=True)
            del call, pytorch
            torch.cuda.empty_cache()
    if arguments.host:
        for variant in arguments.variants:
            call, pytorch, *_ = runs(variant, *shape_of(HOST_SHAPE))
            print(f"host time per call on {HOST_SHAPE} {variant:18} {timing.host_seconds(call) * 1e6:5.1f} us  "
                  f"torch {timing.host_seconds(pytorch) * 1e6:5.1f} us", flush=True)


if __name__ == "__main__":
    timing.run(main)
