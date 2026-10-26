import importlib
import itertools
import math
from collections import namedtuple

import numpy as np
import pytest
import torch

from checks import case, queue_equal_check, queue_requirement, same_bits
from conftest import cuda_seed, declare_gpu_work
from opcheck import (
    assert_symbolic,
    call_symbolic,
    check_compiled,
    check_graph_replay,
    check_op,
    fresh_compiler,  # noqa: F401
)
from zig_cu_kernels import rms_norm as launcher
from zig_cu_kernels import runtime
from zig_cu_kernels.rms_norm import Options
from zig_cu_kernels.torch import (
    adaptive_rms_norm,
    dual_rms_norm,
    dual_rms_norm_fp8_quant,
    fused_add_adaptive_rms_norm,
    fused_add_adaptive_rms_norm_fp8_quant,
    fused_add_rms_norm,
    fused_add_rms_norm_fp8_quant,
    rms_norm,
    rms_norm_fp8_quant,
)

torch_layer = importlib.import_module("zig_cu_kernels.torch.rms_norm")

BF16 = torch.bfloat16
F32 = torch.float32
FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
ELEMENTS = {BF16: "bf16", F32: "f32"}
EPS = 1e-6
# Rounding this integer to a double first and then to FP32 would give 2^60, while its FP32 rounding is 2^60 + 2^37.
INTEGER_EPS = 2**60 + 2**36 + 1
# The widths give rows of 1 to 1024 threads with 1, 2 and 4 chunks per thread, rows that leave threads idle, work
# groups of several rows and rows of several warps.
WIDTHS = [8, 64, 136, 256, 264, 520, 1000, 1032, 4088, 4104, 8200, 16384, 16392, 32768]
FP8_WIDTHS = [8, 64, 264, 1032, 4104, 16392, 32768]
BLOCK_WIDTHS = [128, 384, 1152, 4224, 16384, 24576]
MAX_ADD_WIDTHS = {BF16: 32768, F32: 16384}
ROWS = [1, 13]
SAMPLE_ROWS = [(1, 1), (13, 5), (24, 8)]
BF16_BOUND = 3 * 2**-7
FP32_EXACT_BOUND = 2**-19
FP32_PYTORCH_BOUND = 8 * 2**-23
TINY = 2**-126
MISMATCH_FRACTION = 0.01
FP8_MAX = 448.0
SCALINGS = [("tensorwise", F32), ("rowwise", F32), ("block128", F32), ("block128", E8M0)]
# A case of an accuracy test runs one kernel and fewer than this many PyTorch launches for its references, each on at
# most four FP64 tensors of the size of `x`.
CASE_LAUNCHES = 40
CASE_BYTES_PER_VALUE = 32
SENTINEL = -7.0
SENTINEL_BYTE = 0x55
POINTER_NAMES = ("x", "addend", "residual_out", "weight", "branch_weight", "gate", "scale", "shift", "y", "scales")
Launch = namedtuple("Launch", "kernel shared_bytes pointers")
picked = []


def declare_cases(count, rows, width):
    declare_gpu_work(count * CASE_LAUNCHES, dram_bytes=CASE_BYTES_PER_VALUE * rows * width)


@pytest.fixture(autouse=True)
def record_launches(monkeypatch):
    launch = launcher.Launch.launch

    def recording(plan, pointers, *arguments):
        pointer_map = dict(zip(POINTER_NAMES, pointers, strict=True))
        picked.append(Launch(launcher.SIZE_SUFFIX.sub("", plan.kernel), plan.shared_bytes,
                             pointer_map))
        return launch(plan, pointers, *arguments)

    picked.clear()
    monkeypatch.setattr(launcher.Launch, "launch", recording)


def ran(options):
    return any(launch.kernel == options.name for launch in picked)


def wrote_in_place(**outputs):
    """Returns whether the last launch wrote each output, named as in `POINTER_NAMES`, at the given tensor."""
    return all(picked[-1].pointers[name] == tensor.data_ptr() for name, tensor in outputs.items())


def require_kernels(*options):
    for option in options:
        reason = launcher.unavailable_reason(option)
        if reason is not None:
            pytest.skip(reason)


def fp32(value):
    return float(np.float32(value))


def weight_mode(weight_dtype, zero_centered):
    if weight_dtype is None:
        return "none"
    return ("zero_centered_" if zero_centered else "") + ELEMENTS[weight_dtype]


def fp8_output(scaling, scale_dtype):
    if scaling in ("tensorwise", "rowwise"):
        return f"fp8_{scaling}"
    return "fp8_blockwise_e8m0" if scale_dtype == E8M0 else "fp8_blockwise"


def random_rows(rows, width, dtype, zero_row=False):
    x = (torch.randn(rows, width, device="cuda") * 2).to(dtype)
    if zero_row and rows > 1:
        x[rows // 2] = 0
    return x


def random_weight(width, dtype):
    return (torch.randn(width, device="cuda") * 0.5 + 1).to(dtype)


def modulation_rows(samples, width, count=3):
    return torch.randn(samples, count * width, device="cuda", dtype=BF16).chunk(count, -1)


def expand(modulation, rows_per_sample, rows):
    return modulation.repeat_interleave(rows_per_sample, dim=0)[:rows]


def norm_pytorch(h, weight, zero_centered, eps=EPS):
    """Returns `n0` without a weight, the `LlamaRMSNorm` result with a BF16 weight, `w * n0` with an FP32 weight and
    the `GemmaRMSNorm` result with a zero-centered weight."""
    hf = h.float()
    n0 = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is None:
        return n0
    if zero_centered:
        return n0 * (1 + weight.float())
    return weight * n0.to(h.dtype) if weight.dtype == BF16 else weight * n0


def norm_exact(h, weight, zero_centered, eps=EPS):
    h = h.double()
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + fp32(eps))
    if weight is None:
        return n
    return n * (1 + weight.double()) if zero_centered else n * weight.double()


def modulate(n, scale, shift, rows_per_sample):
    """The PyTorch BF16 expression `(1 + scale) * n + shift` on the BF16 norm `n`."""
    rows = n.shape[0]
    y = (1 + expand(scale, rows_per_sample, rows)) * n
    return y if shift is None else y + expand(shift, rows_per_sample, rows)


def e8m0_bytes(quotient):
    """The least byte `e` with `2^(e - 127) >= quotient`, clamped to 0..254; 127 for 0 and 255 for NaN or inf."""
    mantissa, exponent = torch.frexp(quotient)
    byte = (exponent - (mantissa == 0.5).to(exponent.dtype) + 127).clamp(0, 254)
    byte = torch.where(quotient == 0, 127, byte)
    byte = torch.where(quotient.isnan() | quotient.isinf(), 255, byte)
    return byte.to(torch.uint8).view(E8M0)


def quantize(v, scaling, scale_dtype, fp8_scale=None):
    """The quantizer rules applied to the BF16 rows `v`; returns the codes and the scales."""
    rows, width = v.shape
    v = v.float()
    if scaling == "tensorwise":
        return (v / fp8_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), fp8_scale.reshape(())
    scale_groups = v.view(rows, 1, width) if scaling == "rowwise" else v.view(rows, width // 128, 128)
    quotient = scale_groups.abs().amax(-1) / torch.tensor(FP8_MAX, device=v.device)
    if scale_dtype == E8M0:
        scales = e8m0_bytes(quotient)
        divisor = scales.float()
    else:
        scales = divisor = torch.where(quotient == 0, 1.0, quotient)
    codes = (scale_groups / divisor[..., None]).clamp(-FP8_MAX, FP8_MAX).to(FP8).view(rows, width)
    return codes, scales


def queue_bound(tensor, reference, relative, name, magnitude=None):
    error = (tensor.double() - reference.double()).abs()
    allowed = relative * (reference if magnitude is None else magnitude).double().abs() + TINY
    queue_requirement(f"{name} within {relative:.3g}", (error <= allowed).all())


class Mismatches:
    """Counts the BF16 results of a test that differ in bits from the PyTorch sequence, over all of its cases."""

    def __init__(self, name):
        self.name = name
        self.counts = []
        self.total = 0

    def add(self, tensor, pytorch):
        self.counts.append((~same_bits(tensor, pytorch)).sum())
        self.total += tensor.numel()

    def queue_check(self):
        differing = torch.stack(self.counts).sum()
        allowed = MISMATCH_FRACTION * self.total
        queue_requirement(f"{self.name} differs from PyTorch in at most 1%", differing <= allowed)


def widths_for(dtype, add=False):
    return [width for width in WIDTHS if not add or width <= MAX_ADD_WIDTHS[dtype]]


NORMS = [
    # x dtype, weight dtype, zero_centered, output dtype, residual add
    (BF16, None, False, BF16, False),
    (BF16, BF16, False, BF16, False),
    (BF16, F32, False, BF16, False),
    (BF16, BF16, True, BF16, False),
    (BF16, F32, True, BF16, False),
    (F32, None, False, F32, False),
    (F32, F32, False, F32, False),
    (F32, BF16, False, BF16, False),
    (F32, BF16, False, F32, False),
    (F32, F32, True, F32, False),
    (BF16, None, False, BF16, True),
    (BF16, BF16, False, BF16, True),
    (BF16, BF16, True, BF16, True),
    (F32, BF16, False, BF16, True),
    (F32, BF16, False, F32, True),
]


def norm_options(dtype, weight_dtype, zero_centered, out_dtype, residual):
    return Options(x=ELEMENTS[dtype], add="residual" if residual else "none",
                   weight=weight_mode(weight_dtype, zero_centered), output=ELEMENTS[out_dtype])


@pytest.mark.parametrize("dtype,weight_dtype,zero_centered,out_dtype,residual", NORMS)
def test_norms_meet_the_bounds_of_their_regime(dtype, weight_dtype, zero_centered, out_dtype, residual):
    cuda_seed(0)
    widths = widths_for(dtype, residual)
    declare_cases(3 * len(widths) * len(ROWS), max(ROWS), max(widths))
    mismatches = Mismatches("y")
    for width in widths:
        weight = None if weight_dtype is None else random_weight(width, weight_dtype)
        for rows in ROWS:
            x = random_rows(rows, width, dtype, zero_row=True)
            with case(width=width, rows=rows):
                if residual:
                    other = random_rows(rows, width, dtype)
                    y, h = fused_add_rms_norm(x, other, weight, EPS, zero_centered=zero_centered, out_dtype=out_dtype)
                    queue_equal_check(h, x + other, "residual_out")
                    queue_equal_check(y, rms_norm(h, weight, EPS, zero_centered=zero_centered, out_dtype=out_dtype),
                                      "y against the norm of residual_out")
                else:
                    h = x
                    y = rms_norm(x, weight, EPS, zero_centered=zero_centered, out_dtype=out_dtype)
                assert (y.shape, y.dtype) == (x.shape, out_dtype)
                pytorch = norm_pytorch(h, weight, zero_centered).to(out_dtype)
                if out_dtype == BF16:
                    queue_bound(y, pytorch, BF16_BOUND, "y against PyTorch")
                    mismatches.add(y, pytorch)
                else:
                    queue_bound(y, norm_exact(h, weight, zero_centered), FP32_EXACT_BOUND, "y against FP64")
                    queue_bound(y, pytorch, FP32_PYTORCH_BOUND, "y against PyTorch")
    assert ran(norm_options(dtype, weight_dtype, zero_centered, out_dtype, residual))
    if out_dtype == BF16:
        mismatches.queue_check()


ADAPTIVE = [
    # weight dtype, zero_centered, shift, gated add
    (weight_dtype, weight_dtype is not None, shifted, gated)
    for weight_dtype, shifted, gated in itertools.product((None, F32), (False, True), (False, True))
]


def adaptive_options(weight_dtype, zero_centered, shifted, gated, output="bf16"):
    return Options(add="gated" if gated else "none", weight=weight_mode(weight_dtype, zero_centered),
                   modulation="scale_shift" if shifted else "scale", output=output)


def check_default_rows_per_sample(weight_dtype, zero_centered, shifted, gated):
    """Checks that the modulation tensors of different row counts take `rows_per_sample` from the first of `gate`,
    `scale` and `shift` that the call passes."""
    rows, width, rows_per_sample = 24, 256, 4
    first = rows // rows_per_sample
    gate = torch.randn(first, width, device="cuda", dtype=BF16)
    scale = torch.randn(2 * first if gated else first, width, device="cuda", dtype=BF16)
    shift = torch.randn(first + 2, width, device="cuda", dtype=BF16) if shifted else None
    x, branch = random_rows(rows, width, BF16), random_rows(rows, width, BF16)
    weight = None if weight_dtype is None else random_weight(width, weight_dtype)
    arguments = dict(eps=EPS, weight=weight, zero_centered=zero_centered)

    def call(**options):
        if gated:
            return fused_add_adaptive_rms_norm(x, branch, gate, scale, shift, **arguments, **options)
        return (adaptive_rms_norm(x, scale, shift, **arguments, **options),)

    for result, expected in zip(call(), call(rows_per_sample=rows_per_sample), strict=True):
        queue_equal_check(result, expected, "rows_per_sample from the first modulation tensor")


@pytest.mark.parametrize("weight_dtype,zero_centered,shifted,gated", ADAPTIVE)
def test_adaptive_norms_modulate_their_own_norm(weight_dtype, zero_centered, shifted, gated):
    cuda_seed(1)
    declare_cases(2 * len(WIDTHS) * len(SAMPLE_ROWS) + 2, 24, max(WIDTHS))
    for width in WIDTHS:
        weight = None if weight_dtype is None else random_weight(width, weight_dtype)
        for rows, rows_per_sample in SAMPLE_ROWS:
            scale, shift, gate = modulation_rows(-(-rows // rows_per_sample), width)
            shift = shift if shifted else None
            x = random_rows(rows, width, BF16)
            arguments = dict(eps=EPS, weight=weight, zero_centered=zero_centered, rows_per_sample=rows_per_sample)
            with case(width=width, rows=rows):
                if gated:
                    branch = random_rows(rows, width, BF16)
                    y, h = fused_add_adaptive_rms_norm(x, branch, gate, scale, shift, **arguments)
                    queue_equal_check(h, x + expand(gate, rows_per_sample, rows) * branch, "residual_out")
                else:
                    h = x
                    y = adaptive_rms_norm(x, scale, shift, **arguments)
                n = rms_norm(h, weight, EPS, zero_centered=zero_centered)
                queue_equal_check(y, modulate(n, scale, shift, rows_per_sample), "y")
    assert ran(adaptive_options(weight_dtype, zero_centered, shifted, gated))
    if gated or shifted:
        check_default_rows_per_sample(weight_dtype, zero_centered, shifted, gated)


DUAL = [
    # zero_centered, gate, scale
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, False, False),
]


def dual_options(zero_centered, gated, scaled, output="bf16"):
    return Options(add="dual_gated" if gated else "dual", weight=weight_mode(BF16, zero_centered),
                   modulation="scale" if scaled else "none", output=output)


@pytest.mark.parametrize("zero_centered,gated,scaled", DUAL)
def test_dual_norms_meet_the_bounds_of_their_regime(zero_centered, gated, scaled):
    cuda_seed(2)
    declare_cases(2 * len(WIDTHS) * len(SAMPLE_ROWS), 24, max(WIDTHS))
    mismatches = Mismatches("residual_out")
    for width in WIDTHS:
        weight, branch_weight = random_weight(width, BF16), random_weight(width, BF16)
        for rows, rows_per_sample in SAMPLE_ROWS:
            samples = -(-rows // rows_per_sample)
            gate, scale = modulation_rows(samples, width, 2)
            gate, scale = gate if gated else None, scale if scaled else None
            x, branch = random_rows(rows, width, BF16), random_rows(rows, width, BF16) * 3
            added = norm_pytorch(branch, branch_weight, zero_centered).to(BF16)
            if gated:
                added = expand(gate, rows_per_sample, rows) * added
            with case(width=width, rows=rows):
                y, h = dual_rms_norm(x, branch, branch_weight, weight, EPS, zero_centered=zero_centered, gate=gate,
                                     scale=scale, rows_per_sample=rows_per_sample)
                queue_bound(h, x + added, BF16_BOUND, "residual_out against PyTorch", x.abs() + added.abs())
                mismatches.add(h, x + added)
                n = rms_norm(h, weight, EPS, zero_centered=zero_centered)
                queue_equal_check(y, modulate(n, scale, None, rows_per_sample) if scaled else n, "y")
    assert ran(dual_options(zero_centered, gated, scaled))
    mismatches.queue_check()


def fp8_case(function, rows, width, scaling, scale_dtype, fp8_scale, **call):
    """Returns the FP8 call of `function` on new random inputs and the same call with a BF16 output."""
    x = random_rows(rows, width, BF16, zero_row=True)
    weight = random_weight(width, BF16)
    quantized = dict(fp8_scaling=scaling, fp8_scale_dtype=scale_dtype)
    if scaling == "tensorwise":
        quantized["fp8_scale"] = fp8_scale
    if function == "rms_norm":
        return (lambda: rms_norm_fp8_quant(x, weight, EPS, **quantized, **call),
                lambda: (rms_norm(x, weight, EPS, **call),))
    other = random_rows(rows, width, BF16)
    if function == "fused_add":
        return (lambda: fused_add_rms_norm_fp8_quant(x, other, weight, EPS, **quantized, **call),
                lambda: fused_add_rms_norm(x, other, weight, EPS, **call))
    if function == "dual":
        branch_weight = random_weight(width, BF16)
        return (lambda: dual_rms_norm_fp8_quant(x, other, branch_weight, weight, EPS, **quantized, **call),
                lambda: dual_rms_norm(x, other, branch_weight, weight, EPS, **call))
    scale, shift, gate = modulation_rows(rows, width)
    return (lambda: fused_add_adaptive_rms_norm_fp8_quant(x, other, gate, scale, shift, eps=EPS, **quantized, **call),
            lambda: fused_add_adaptive_rms_norm(x, other, gate, scale, shift, eps=EPS, **call))


@pytest.fixture
def eight_chunks(monkeypatch):
    """Makes the launcher prefer eight chunks per work item for the FP8 outputs, the size that no default launch
    on this device may choose."""
    yield from preferring(monkeypatch, 8)


@pytest.fixture
def six_chunks(monkeypatch):
    """Makes the launcher prefer six chunks per work item for the FP8 outputs."""
    yield from preferring(monkeypatch, 6)


@pytest.fixture
def four_chunks(monkeypatch):
    """Makes the launcher prefer four chunks per work item for the FP8 outputs, which with an addend takes the size
    of at most 512 work items."""
    yield from preferring(monkeypatch, 4)


def preferring(monkeypatch, chunks):
    monkeypatch.setattr(launcher, "preferred_chunks", lambda device_id, options, width: chunks if options.fp8 else None)
    launcher.launch_for.cache_clear()
    yield
    launcher.launch_for.cache_clear()


FP8_FUNCTIONS = {
    "rms_norm": Options(weight="bf16"),
    "fused_add": Options(add="residual", weight="bf16"),
    "dual": Options(add="dual", weight="bf16"),
    "fused_add_adaptive": Options(add="gated", modulation="scale_shift"),
}
FP8_CASES = [(function, scaling, scale_dtype) for function in FP8_FUNCTIONS for scaling, scale_dtype in SCALINGS
             if not (function == "fused_add_adaptive" and scaling == "tensorwise")]


def fp8_options(function, scaling, scale_dtype):
    base = FP8_FUNCTIONS[function]
    return Options(x=base.x, add=base.add, weight=base.weight, modulation=base.modulation,
                   output=fp8_output(scaling, scale_dtype))


def queue_fp8_checks(result, reference, scaling, scale_dtype, fp8_scale):
    q, scales, *rest = result
    v, *reference_rest = reference
    rows = math.prod(v.shape[:-1])
    codes, expected_scales = quantize(v.reshape(rows, -1), scaling, scale_dtype, fp8_scale)
    queue_equal_check(q.reshape(rows, -1), codes, "codes")
    queue_equal_check(scales, expected_scales, "scales")
    if scaling == "tensorwise":
        assert scales.data_ptr() == fp8_scale.data_ptr()
    elif scaling == "block128":
        assert scales.stride() == (1, rows) or scales.shape[1] == 1
    for residual_out, expected in zip(rest, reference_rest):
        queue_equal_check(residual_out, expected, "residual_out")


@pytest.mark.parametrize("function,scaling,scale_dtype", FP8_CASES)
def test_fp8_outputs_quantize_the_bf16_result(function, scaling, scale_dtype):
    require_kernels(fp8_options(function, scaling, scale_dtype))
    cuda_seed(3)
    widths = BLOCK_WIDTHS if scaling == "block128" else FP8_WIDTHS
    fp8_scale = torch.tensor([0.0123], device="cuda")
    declare_cases(2 * len(widths) * 3, 13, max(widths))
    for width in widths:
        for rows in (1, 6, 13):
            call, reference = fp8_case(function, rows, width, scaling, scale_dtype, fp8_scale)
            with case(width=width, rows=rows):
                queue_fp8_checks(call(), reference(), scaling, scale_dtype, fp8_scale)
    assert ran(fp8_options(function, scaling, scale_dtype))


@pytest.mark.parametrize("scaling,scale_dtype", SCALINGS)
def test_fp8_outputs_of_eight_chunks_quantize_the_bf16_result(scaling, scale_dtype, eight_chunks):
    test_fp8_outputs_quantize_the_bf16_result("rms_norm", scaling, scale_dtype)


@pytest.mark.parametrize("scaling,scale_dtype", SCALINGS)
def test_fp8_outputs_of_six_chunks_quantize_the_bf16_result(scaling, scale_dtype, six_chunks):
    test_fp8_outputs_quantize_the_bf16_result("rms_norm", scaling, scale_dtype)


@pytest.mark.parametrize("function,scaling,scale_dtype", [case for case in FP8_CASES if case[0] != "rms_norm"])
def test_fp8_outputs_with_an_addend_of_four_chunks_quantize_the_bf16_result(function, scaling, scale_dtype,
                                                                          four_chunks):
    test_fp8_outputs_quantize_the_bf16_result(function, scaling, scale_dtype)


SUBNORMAL_COLUMN = 1


def special_rows(width, dtype):
    """Returns rows with a NaN, with an infinity among finite values, of zeros, of values whose squares overflow FP32,
    and of Gaussian values with a subnormal value in `SUBNORMAL_COLUMN`."""
    rows = random_rows(5, width, dtype)
    rows[0, width // 3] = math.nan
    rows[1, width // 2] = -math.inf
    rows[2] = 0
    rows[3] = torch.where(rows[3] > 0, 1e30, -1e30)
    rows[4, SUBNORMAL_COLUMN] = 2.0**-130
    return rows


@pytest.mark.parametrize("dtype", [BF16, F32])
def test_special_values_follow_the_pytorch_sequence(dtype):
    cuda_seed(4)
    widths = [8, 1032, 32768]
    declare_cases(4 * len(widths), 5, max(widths))
    for width in widths:
        weight = random_weight(width, BF16)
        x = special_rows(width, dtype)
        with case(width=width):
            for eps in (EPS, 0.0):
                y = rms_norm(x, weight, eps)
                pytorch = norm_pytorch(x, weight, False, eps).to(dtype)
                queue_equal_check(y[:4], pytorch[:4], f"special rows with eps={eps}")
            y = rms_norm(x, weight, EPS)
            queue_requirement("NaN row", y[0].isnan().all())
            infinite = x[1].isinf()
            queue_requirement("infinite row", (y[1][infinite].isnan().all() & (y[1][~infinite] == 0).all()))
            queue_requirement("subnormal input kept", y[4, SUBNORMAL_COLUMN] != 0)
            queue_bound(y[4], pytorch[4], BF16_BOUND if dtype == BF16 else FP32_PYTORCH_BOUND, "Gaussian row")


@pytest.mark.parametrize("scaling,scale_dtype", SCALINGS)
def test_fp8_special_values_follow_the_quantizer_rules(scaling, scale_dtype):
    require_kernels(Options(weight="bf16", output=fp8_output(scaling, scale_dtype)))
    width = 1024
    declare_cases(6, 5, width)
    x = special_rows(width, BF16)
    weight = random_weight(width, BF16)
    weight[:128] = 0
    weight[200] = math.nan
    weight[300] = math.inf
    weight[384:512] = 1e-38
    fp8_scale = torch.tensor(1e-4, device="cuda")
    given = dict(fp8_scale=fp8_scale) if scaling == "tensorwise" else {}
    result = rms_norm_fp8_quant(x, weight, EPS, fp8_scaling=scaling, fp8_scale_dtype=scale_dtype, **given)
    v = rms_norm(x, weight, EPS)
    queue_fp8_checks(result, (v,), scaling, scale_dtype, fp8_scale)
    q, scales = result
    queue_requirement("NaN row gives NaN codes", q[0].float().isnan().all())
    if scaling == "tensorwise":
        queue_requirement("saturated codes", q[4, 512:].float().abs().amax() == FP8_MAX)
    elif scaling == "rowwise":
        queue_requirement("NaN weight makes every rowwise scale NaN", scales.isnan().all())
    else:
        nan_scale = scales.view(torch.uint8) == 255 if scale_dtype == E8M0 else scales.isnan()
        one = scales.view(torch.uint8) == 127 if scale_dtype == E8M0 else scales == 1
        queue_requirement("a NaN row or a NaN weight gives NaN scales", nan_scale[0].all() & nan_scale[:, 1].all())
        infinite = nan_scale[:, 2].all() if scale_dtype == E8M0 else scales[4, 2] == math.inf
        queue_requirement("an infinite weight gives byte 255 or an infinite scale", infinite)
        queue_requirement("scale groups of zeros get the scale 1", one[1:, 0].all() & (q[1:, :128].float() == 0).all())


def large_case(kind):
    """Returns the row length, the dtype of `x`, the call and its options for one kind of instance."""
    if kind == "plain":
        weight = random_weight(1024, BF16)
        return 1024, BF16, lambda x: (rms_norm(x, weight, EPS),), Options(weight="bf16")
    if kind == "residual":
        weight = random_weight(4096, BF16)
        return (4096, BF16, lambda x: fused_add_rms_norm(x, x * 0.5, weight, EPS),
                Options(add="residual", weight="bf16"))
    weight = random_weight(1024, BF16)
    return (1024, BF16, lambda x: rms_norm_fp8_quant(x, weight, EPS, fp8_scaling="rowwise"),
            Options(weight="bf16", output="fp8_rowwise"))


@pytest.mark.parametrize("kind", ["plain", "residual", "fp8_rowwise"])
def test_launches_beyond_the_l2_cache_match_small_launches(kind):
    cuda_seed(5)
    width, dtype, call, options = large_case(kind)
    require_kernels(options)
    rows = runtime.l2_bytes(runtime.current_device().device_id) // (width * dtype.itemsize) + 13
    declare_gpu_work(8, dram_bytes=4 * rows * width * dtype.itemsize)
    x = random_rows(rows, width, dtype)
    picked.clear()
    results = call(x)
    device_id = runtime.current_device().device_id
    if launcher.tuning.engine == "registers" and launcher.in_flight_limit(device_id, options, 1) is not None:
        assert picked[-1].shared_bytes > 0, "the launch keeps every work group that fits"
    for part in (slice(0, 13), slice(rows - 13, rows)):
        for large, expected in zip(results, call(x[part])):
            queue_equal_check(large[part], expected, f"{kind} rows {part.start}")


def layout_inputs(width=1032):
    storage = torch.randn(40, width + 24, device="cuda", dtype=BF16)
    return storage, storage[:, 8 : 8 + width], storage[:, 1 : 1 + width], random_weight(width, BF16)


def test_strided_and_misaligned_tensors_give_the_contiguous_results():
    cuda_seed(6)
    declare_cases(30, 40, 1056)
    storage, x, misaligned, weight = layout_inputs()
    width = x.shape[1]
    expected = rms_norm(x.contiguous(), weight, EPS)
    queue_equal_check(rms_norm(x, weight, EPS), expected, "strided x")
    queue_equal_check(rms_norm(misaligned, weight, EPS), rms_norm(misaligned.contiguous(), weight, EPS), "misaligned x")
    queue_equal_check(rms_norm(x[5:9], weight, EPS), expected[5:9], "rows taken alone")
    queue_equal_check(rms_norm(x.view(4, 10, width), weight, EPS), expected.view(4, 10, width), "batch shape")
    batches = x[:4].view(2, 2, width).transpose(0, 1)
    queue_equal_check(rms_norm(batches, weight, EPS), rms_norm(batches.contiguous(), weight, EPS),
                      "leading dimensions that do not flatten")
    modulation = storage[:4, 16 : 16 + width].view(2, 2, width).transpose(0, 1)
    queue_equal_check(adaptive_rms_norm(x[:4], modulation, eps=EPS),
                      adaptive_rms_norm(x[:4], modulation.contiguous(), eps=EPS),
                      "modulation whose leading dimensions do not flatten")
    sliced = storage.view(8, 5, width + 24)[:, 2:3, 8 : 8 + width]
    sliced_out = torch.empty(8, 3, width, device="cuda", dtype=BF16)[:, 1:2]
    rms_norm(sliced, weight, EPS, out=sliced_out)
    assert wrote_in_place(x=sliced, y=sliced_out), "size-1 dimensions of any stride go in place"
    queue_equal_check(sliced_out, rms_norm(sliced.contiguous(), weight, EPS), "size-1 dimensions of any stride")
    for offset in (0, 3):
        out = torch.empty(40, width + 16, device="cuda", dtype=BF16)[:, offset : offset + width]
        assert rms_norm(x, weight, EPS, out=out) is out
        queue_equal_check(out, expected, f"out at offset {offset}")
    residual = storage[:, 16 : 16 + width]
    stream_out = torch.empty(40, width + 16, device="cuda", dtype=BF16)[:, 1 : 1 + width]
    y, h = fused_add_rms_norm(x, residual, weight, EPS, residual_out=stream_out)
    assert h is stream_out
    queue_equal_check(stream_out, x + residual, "misaligned residual_out")
    queue_equal_check(y, rms_norm(x + residual, weight, EPS), "y")
    misaligned_weight = torch.empty(width + 1, device="cuda", dtype=BF16)[1:]
    misaligned_weight.copy_(weight)
    queue_equal_check(rms_norm(x, misaligned_weight, EPS), expected, "misaligned weight")


def test_exact_aliases_write_in_place():
    cuda_seed(7)
    width, rows = 1024, 13
    declare_cases(20, rows, width)
    x, residual, branch = (random_rows(rows, width, BF16) for _ in range(3))
    weight, branch_weight = random_weight(width, BF16), random_weight(width, BF16)
    scale, shift, gate = modulation_rows(rows, width)
    expected_y, expected_h = fused_add_rms_norm(x, residual, weight, EPS)
    xs, rs = x.clone(), residual.clone()
    y, h = fused_add_rms_norm(xs, rs, weight, EPS, out=xs, residual_out=rs)
    assert y is xs and h is rs
    assert wrote_in_place(x=xs, addend=rs, y=xs, residual_out=rs)
    queue_equal_check(xs, expected_y, "out = x")
    queue_equal_check(rs, expected_h, "residual_out = residual")
    xs, rs = x.clone(), residual.clone()
    fused_add_rms_norm(xs, rs, weight, EPS, out=rs, residual_out=xs)
    assert wrote_in_place(x=xs, addend=rs, y=rs, residual_out=xs)
    queue_equal_check(rs, expected_y, "out = residual")
    queue_equal_check(xs, expected_h, "residual_out = x")
    expected = fused_add_adaptive_rms_norm(x, branch, gate, scale, shift, eps=EPS)
    xs, bs = x.clone(), branch.clone()
    fused_add_adaptive_rms_norm(xs, bs, gate, scale, shift, eps=EPS, out=xs, residual_out=bs)
    assert wrote_in_place(x=xs, addend=bs, y=xs, residual_out=bs)
    queue_equal_check(xs, expected[0], "adaptive out = x")
    queue_equal_check(bs, expected[1], "adaptive residual_out = branch")
    expected = dual_rms_norm(x, branch, branch_weight, weight, EPS)
    xs, bs = x.clone(), branch.clone()
    dual_rms_norm(xs, bs, branch_weight, weight, EPS, out=bs, residual_out=xs)
    assert wrote_in_place(x=xs, addend=bs, y=bs, residual_out=xs)
    queue_equal_check(bs, expected[0], "dual out = branch")
    queue_equal_check(xs, expected[1], "dual residual_out = x")
    xs = x.clone()
    rms_norm(xs, weight, EPS, out=xs)
    assert wrote_in_place(x=xs, y=xs)
    queue_equal_check(xs, rms_norm(x, weight, EPS), "rms_norm out = x")


def test_overlapping_outputs_go_through_temporaries():
    cuda_seed(8)
    width, rows = 1024, 13
    declare_cases(16, rows + 1, 2 * width)
    weight = random_weight(width, BF16)
    buffer = random_rows(rows + 1, width, BF16)
    x, out = buffer[:rows], buffer[1:]
    expected = rms_norm(x.clone(), weight, EPS)
    rms_norm(x, weight, EPS, out=out)
    assert wrote_in_place(x=x) and not wrote_in_place(y=out), "out over x goes through a temporary"
    queue_equal_check(out, expected, "out shifted by one row over x")
    x, residual = random_rows(rows, width, BF16), random_rows(rows, width, BF16)
    halves = torch.empty(rows, 2 * width, device="cuda", dtype=BF16)
    y, h = fused_add_rms_norm(x, residual, weight, EPS, out=halves[:, :width], residual_out=halves[:, width:])
    assert wrote_in_place(y=halves[:, :width], residual_out=halves[:, width:]), "outputs that share no memory"
    expected_y, expected_h = fused_add_rms_norm(x, residual, weight, EPS)
    queue_equal_check(halves[:, :width], expected_y, "out in the left half")
    queue_equal_check(halves[:, width:], expected_h, "residual_out in the right half")
    shifted = torch.empty(rows * width + 8, device="cuda", dtype=BF16)
    stream_out = shifted[8:].view(rows, width)
    residual_in = shifted[: rows * width].view(rows, width)
    residual_in.copy_(residual)
    y, h = fused_add_rms_norm(x, residual_in, weight, EPS, residual_out=stream_out)
    assert wrote_in_place(addend=residual_in) and not wrote_in_place(residual_out=stream_out)
    queue_equal_check(stream_out, expected_h, "residual_out shifted over residual")
    queue_equal_check(y, expected_y, "y with a shifted residual_out")
    if not launcher.is_available(Options(weight="bf16", output="fp8_rowwise")):
        return
    storage = torch.empty(rows * width // 2 + rows, device="cuda")
    x = storage[: rows * width // 2].view(BF16).view(rows, width)
    x.copy_(random_rows(rows, width, BF16))
    scales_out = storage[rows * width // 2 - 1 :][:rows].view(rows, 1)
    expected = rms_norm_fp8_quant(x.clone(), weight, EPS, fp8_scaling="rowwise")
    q, scales = rms_norm_fp8_quant(x, weight, EPS, fp8_scaling="rowwise", scales_out=scales_out)
    assert wrote_in_place(x=x) and not wrote_in_place(scales=scales_out), "scales over x go through a temporary"
    queue_equal_check(q, expected[0], "codes with scales over x")
    queue_equal_check(scales_out, expected[1], "scales over x")


@pytest.mark.parametrize("scaling,scale_dtype", SCALINGS[1:])
def test_scale_outputs_write_only_their_elements(scaling, scale_dtype):
    require_kernels(Options(weight="bf16", output=fp8_output(scaling, scale_dtype)))
    cuda_seed(9)
    width, rows = 384 if scaling == "block128" else 64, 13
    declare_cases(6, rows, width)
    x, weight = random_rows(rows, width, BF16), random_weight(width, BF16)
    expected = rms_norm_fp8_quant(x, weight, EPS, fp8_scaling=scaling, fp8_scale_dtype=scale_dtype)
    if scaling == "rowwise":
        buffer = torch.full((16, 2), SENTINEL, device="cuda")
        scales_out, untouched = buffer[:rows, :1], [buffer[:, 1], buffer[rows:, 0]]
    else:
        buffer = torch.full((width // 128, rows + 3), SENTINEL, device="cuda")
        if scale_dtype == E8M0:
            buffer = torch.full(buffer.shape, SENTINEL_BYTE, device="cuda", dtype=torch.uint8).view(E8M0)
        scales_out, untouched = buffer[:, :rows].t(), [buffer[:, rows:]]
    sentinels = [part.clone() for part in untouched]
    q, scales = rms_norm_fp8_quant(x, weight, EPS, fp8_scaling=scaling, fp8_scale_dtype=scale_dtype,
                                   scales_out=scales_out)
    assert scales is scales_out
    queue_equal_check(q, expected[0], "codes with given scales")
    queue_equal_check(scales_out, expected[1], "given scales")
    for part, sentinel in zip(untouched, sentinels):
        queue_equal_check(part, sentinel, "elements outside the scales")
    contiguous = torch.empty(expected[1].shape, device="cuda", dtype=scale_dtype)
    rms_norm_fp8_quant(x, weight, EPS, fp8_scaling=scaling, fp8_scale_dtype=scale_dtype, scales_out=contiguous)
    queue_equal_check(contiguous, expected[1], "contiguous given scales")


def test_empty_inputs_launch_nothing():
    declare_gpu_work(0)
    width = 1024
    weight = random_weight(width, BF16)
    empty = torch.empty(0, width, device="cuda", dtype=BF16)
    picked.clear()
    assert rms_norm(empty, weight, EPS).shape == (0, width)
    y, h = fused_add_rms_norm(torch.empty(2, 0, width, device="cuda", dtype=BF16), torch.empty(2, 0, width,
                              device="cuda", dtype=BF16), weight, EPS)
    assert y.shape == h.shape == (2, 0, width)
    assert rms_norm(torch.empty(4, 0, device="cuda", dtype=BF16), None, EPS).shape == (4, 0)
    for samples in (0, 3):
        scale = torch.empty(samples, width, device="cuda", dtype=BF16)
        assert adaptive_rms_norm(empty, scale, eps=EPS).shape == (0, width)
    if launcher.is_available(Options(weight="bf16", output="fp8_blockwise")):
        q, scales = rms_norm_fp8_quant(empty, weight, EPS, fp8_scaling="block128")
        assert (q.shape, q.dtype, scales.shape) == ((0, width), FP8, (0, width // 128))
        q, scales = rms_norm_fp8_quant(empty, weight, EPS, fp8_scaling="rowwise")
        assert scales.shape == (0, 1)
    assert not picked


def op_arguments(x, fp8):
    """Returns (functional op, its arguments, `_out` op, its arguments) for every op pair."""
    width = x.shape[-1]
    residual, branch = torch.randn_like(x), torch.randn_like(x)
    weight, branch_weight = random_weight(width, BF16), random_weight(width, BF16)
    scale, shift, gate = (torch.randn(x.shape[0], 1, width, device="cuda", dtype=BF16) for _ in range(3))
    fp8_scale = torch.tensor([0.02], device="cuda")
    rows = math.prod(x.shape[:-1])
    y, residual_out = torch.empty_like(x), torch.empty_like(x)
    q = torch.empty_like(x, dtype=FP8)
    pairs = [
        (torch_layer.RMS_NORM, (x, weight, None, EPS, False, BF16), (y,)),
        (torch_layer.FUSED_ADD, (x, residual, weight, None, EPS, False, BF16), (y, residual_out)),
        (torch_layer.ADAPTIVE, (x, scale, shift, None, EPS, False, None), (y,)),
        (torch_layer.FUSED_ADD_ADAPTIVE, (x, branch, gate, scale, shift, None, EPS, False, 3), (y, residual_out)),
        (torch_layer.DUAL, (x, branch, branch_weight, weight, gate, scale, None, EPS, False, None), (y, residual_out)),
    ]
    if fp8:
        rowwise = torch.empty(rows, 1, device="cuda")
        blockwise_e8m0 = torch.empty(width // 128, rows, device="cuda", dtype=E8M0).t()
        blockwise_f32 = torch.empty(width // 128, rows, device="cuda").t()
        pairs += [
            (torch_layer.RMS_NORM, (x, weight, fp8_scale, EPS, False, FP8), (q,)),
            (torch_layer.FUSED_ADD, (x, residual, weight, fp8_scale, EPS, False, FP8), (q, residual_out)),
            (torch_layer.RMS_NORM_FP8, (x, weight, EPS, False, "rowwise", F32), (q, rowwise)),
            (torch_layer.FUSED_ADD_FP8, (x, residual, weight, EPS, False, "block128", E8M0),
             (q, blockwise_e8m0, residual_out)),
            (torch_layer.FUSED_ADD_ADAPTIVE_FP8, (x, branch, gate, scale, shift, None, EPS, False, None, "rowwise",
                                                  F32), (q, rowwise, residual_out)),
            (torch_layer.DUAL, (x, branch, branch_weight, weight, None, None, fp8_scale, EPS, False, None),
             (q, residual_out)),
            (torch_layer.DUAL_FP8, (x, branch, branch_weight, weight, None, None, EPS, False, None, "block128", F32),
             (q, blockwise_f32, residual_out)),
        ]
    return pairs


def out_arguments(pair, arguments, outputs):
    dtype_positions = [index for index, name in enumerate(pair.functional_names) if name in
                       torch_layer.DTYPE_PARAMETERS]
    return (*(value for index, value in enumerate(arguments) if index not in dtype_positions), *outputs)


def test_ops_pass_opcheck():
    fp8 = launcher.is_available(Options(weight="bf16", output="fp8_rowwise"))
    x = torch.randn(2, 3, 256, device="cuda", dtype=BF16)
    pairs = op_arguments(x, fp8)
    declare_gpu_work(len(pairs) * 2 * CASE_LAUNCHES, dram_bytes=CASE_BYTES_PER_VALUE * x.numel())
    for pair, arguments, outputs in pairs:
        check_op(pair.functional, *arguments)
        check_op(pair.out, *out_arguments(pair, arguments, outputs))
    names = {pair.name for pair, _, _ in pairs}
    assert len(names) == (9 if fp8 else 5)


def every_function(x, residual, branch, weight, branch_weight, gate, scale, shift, fp8_scale, fp8):
    results = [
        rms_norm(x, weight, EPS),
        fused_add_rms_norm(x, residual, weight, EPS),
        adaptive_rms_norm(x, scale, shift, eps=EPS),
        fused_add_adaptive_rms_norm(x, branch, gate, scale, shift, eps=EPS),
        dual_rms_norm(x, branch, branch_weight, weight, EPS, gate=gate, scale=scale),
    ]
    if fp8:
        results += [
            rms_norm_fp8_quant(x, weight, EPS, fp8_scaling="tensorwise", fp8_scale=fp8_scale),
            rms_norm_fp8_quant(x, weight, EPS, fp8_scaling="rowwise"),
            fused_add_rms_norm_fp8_quant(x, residual, weight, EPS, fp8_scaling="block128"),
            fused_add_adaptive_rms_norm_fp8_quant(x, branch, gate, scale, shift, eps=EPS, fp8_scaling="block128",
                                                  fp8_scale_dtype=E8M0),
            dual_rms_norm_fp8_quant(x, branch, branch_weight, weight, EPS, fp8_scaling="tensorwise",
                                    fp8_scale=fp8_scale),
            dual_rms_norm_fp8_quant(x, branch, branch_weight, weight, EPS, fp8_scaling="rowwise"),
        ]
    return results


def with_aliases(x, residual, branch, weight, branch_weight, gate, scale, shift, fp8_scale, fp8):
    """Calls the functions in order with every kind of exact alias of outputs and inputs, so each call changes the
    inputs of the later calls."""
    results = list(fused_add_rms_norm(x, residual, weight, EPS, out=x, residual_out=residual))
    results += fused_add_adaptive_rms_norm(x, branch, gate, scale, shift, eps=EPS, out=branch, residual_out=x)
    results += dual_rms_norm(x, branch, branch_weight, weight, EPS, out=x, residual_out=branch)
    results += fused_add_rms_norm(x, residual, weight, EPS, out=residual, residual_out=x)
    results.append(rms_norm(x, weight, EPS, out=x))
    if fp8:
        results += fused_add_rms_norm_fp8_quant(x, residual, weight, EPS, fp8_scaling="rowwise", residual_out=x)
    return results + [x.clone(), residual.clone(), branch.clone()]


def with_outputs(x, residual, branch, weight, branch_weight, gate, scale, shift, fp8_scale, fp8):
    """Calls the `_out` ops that `with_aliases` leaves out and each `_fp8_quant` function with E8M0 scales and given
    outputs. Two calls write a misaligned `residual_out` and an `out` over the rows of `x` through temporaries."""
    results = [adaptive_rms_norm(x, scale, shift, eps=EPS, out=torch.empty_like(x))]
    misaligned = x.new_empty(x.numel() + 1)[1:].view(x.shape)
    results += fused_add_rms_norm(x, residual, weight, EPS, residual_out=misaligned)
    shifted = x.new_empty(x.numel() + x.shape[-1])
    rows_of_x, out = shifted[: x.numel()].view(x.shape), shifted[x.shape[-1] :].view(x.shape)
    rows_of_x.copy_(x)
    results.append(rms_norm(rows_of_x, weight, EPS, out=out))
    if fp8:
        blockwise_scales = [x.new_empty(x.shape[-1] // 128, math.prod(x.shape[:-1]), dtype=dtype).t()
                        for dtype in (F32, E8M0, E8M0)]
        results += rms_norm_fp8_quant(x, weight, EPS, fp8_scaling="block128", scales_out=blockwise_scales[0])
        results += fused_add_adaptive_rms_norm_fp8_quant(x, branch, gate, scale, shift, eps=EPS, fp8_scaling="rowwise",
                                                         residual_out=torch.empty_like(x))
        results += dual_rms_norm_fp8_quant(x, branch, branch_weight, weight, EPS, fp8_scaling="block128",
                                           fp8_scale_dtype=E8M0, out=torch.empty_like(x, dtype=FP8))
        results += dual_rms_norm_fp8_quant(x, branch, branch_weight, weight, EPS, fp8_scaling="tensorwise",
                                           fp8_scale=fp8_scale, out=torch.empty_like(x, dtype=FP8))
        e8m0 = dict(fp8_scaling="block128", fp8_scale_dtype=E8M0)
        results += rms_norm_fp8_quant(x, weight, EPS, out=torch.empty_like(x, dtype=FP8), **e8m0)
        results += fused_add_rms_norm_fp8_quant(x, residual, weight, EPS, scales_out=blockwise_scales[1], **e8m0)
        results += fused_add_adaptive_rms_norm_fp8_quant(x, branch, gate, scale, shift, eps=EPS,
                                                         out=torch.empty_like(x, dtype=FP8),
                                                         scales_out=blockwise_scales[2],
                                                         residual_out=torch.empty_like(x), **e8m0)
    return results


def function_inputs(batch, sequence, width, fp8):
    x, residual, branch = (torch.randn(batch, sequence, width, device="cuda", dtype=BF16) for _ in range(3))
    gate, scale, shift = (torch.randn(batch, 1, width, device="cuda", dtype=BF16) for _ in range(3))
    return (x, residual, branch, random_weight(width, BF16), random_weight(width, BF16), gate, scale, shift,
            torch.tensor([0.05], device="cuda"), fp8)


def queue_same_results(results, expected, name="result"):
    if isinstance(results, torch.Tensor):
        queue_equal_check(results, expected, name)
        return
    assert len(results) == len(expected)
    for index, (part, expected_part) in enumerate(zip(results, expected)):
        queue_same_results(part, expected_part, f"{name}[{index}]")


def test_fakes_keep_every_size_symbolic():
    declare_gpu_work(0)
    inputs = function_inputs(6, 5, 256, True)

    def every_call(*tensors):
        return every_function(*tensors, True) + with_aliases(*tensors, True) + with_outputs(*tensors, True)

    results = call_symbolic(every_call, *inputs[:-1])
    for result in results:
        for tensor in (result,) if isinstance(result, torch.Tensor) else result:
            assert_symbolic(tensor, [dim for dim, size in enumerate(tensor.shape) if not isinstance(size, int)])
            assert all(isinstance(size, torch.SymInt) or size == 1 for size in tensor.shape)
    assert not picked


@pytest.mark.usefixtures("fresh_compiler")
def test_compiled_calls_match_eager_calls():
    fp8 = launcher.is_available(Options(weight="bf16", output="fp8_rowwise"))
    cuda_seed(10)
    sizes = ((2, 3, 256), (3, 5, 384), (4, 7, 512))
    declare_gpu_work(6 * 40 * len(sizes), dram_bytes=CASE_BYTES_PER_VALUE * 4 * 7 * 512)
    for function in (every_function, with_aliases, with_outputs):
        cases = [lambda size=size, seed=seed: (cuda_seed(seed), function_inputs(*size, fp8))[1]
                 for seed, size in enumerate(sizes)]
        check_compiled(function, cases, compare=queue_same_results)


def cloned(results):
    return results.clone() if isinstance(results, torch.Tensor) else [cloned(part) for part in results]


@pytest.mark.parametrize("calls", [every_function, with_aliases, with_outputs], ids=lambda calls: calls.__name__)
def test_calls_never_synchronize_and_replay_in_graphs(calls):
    fp8 = launcher.is_available(Options(weight="bf16", output="fp8_rowwise"))
    cuda_seed(11)
    inputs = function_inputs(4, 12, 1024, fp8)
    tensors = inputs[:-1]
    declare_gpu_work(5 * 40, dram_bytes=CASE_BYTES_PER_VALUE * 48 * 1024)
    refreshed = []

    def refresh():
        refreshed[:] = [torch.randn_like(tensor) for tensor in tensors[:-1]] + [torch.full_like(tensors[-1], 0.03)]
        for tensor, value in zip(tensors, refreshed):
            tensor.copy_(value)

    graph, outputs = check_graph_replay(lambda: calls(*inputs), refresh)
    replayed = cloned(outputs)
    for tensor, value in zip(tensors, refreshed):
        tensor.copy_(value)
    queue_same_results(replayed, calls(*inputs))


class ArrayInterface:
    """Exports a PyTorch tensor through the CUDA Array Interface only, with `typestr` naming its elements."""

    def __init__(self, tensor, typestr):
        strides = tuple(stride * tensor.element_size() for stride in tensor.stride())
        self.__cuda_array_interface__ = {"shape": tuple(tensor.shape), "strides": strides, "typestr": typestr,
                                         "data": (tensor.data_ptr(), False), "version": 3}


def test_launcher_takes_array_interface_objects():
    cuda_seed(12)
    rows, width = 13, 1032
    declare_cases(8, rows, width)
    x, residual = random_rows(rows, width, F32), random_rows(rows, width, F32)
    weight = random_weight(width, BF16)
    y, h = torch.empty_like(x), torch.empty_like(x)
    options = Options(x="f32", add="residual", weight="bf16", output="f32")
    arrays = [ArrayInterface(tensor, "<f4") for tensor in (x, residual, y, h)]
    launcher.rms_norm(options, arrays[0], arrays[2], epsilon=EPS, residual=arrays[1], residual_out=arrays[3],
                      weight=ArrayInterface(weight, "<V2"), stream=torch.cuda.current_stream().cuda_stream)
    expected_y, expected_h = fused_add_rms_norm(x, residual, weight, EPS)
    queue_equal_check(y, expected_y, "y through the array interface")
    queue_equal_check(h, expected_h, "residual_out through the array interface")
    operands = dict(residual=residual, residual_out=h, weight=weight, stream=torch.cuda.current_stream().cuda_stream)
    launcher.rms_norm(options, x, y, epsilon=INTEGER_EPS, **operands)
    queue_equal_check(y, fused_add_rms_norm(x, residual, weight, INTEGER_EPS)[0], "y with an integer epsilon")
    with pytest.raises(TypeError, match="epsilon must be a Python number"):
        launcher.rms_norm(options, x, y, epsilon=torch.tensor(EPS), **operands)


def test_availability_follows_binary_compatibility(monkeypatch):
    declare_gpu_work(0)
    plain, fp8 = Options(weight="bf16"), Options(weight="bf16", output="fp8_rowwise")
    assert launcher.is_available(plain)
    major, minor = min(map(runtime.target_capability, runtime.targets_of(launcher.sizes(plain.name)[0])))
    monkeypatch.setattr(runtime, "compute_capability", lambda device_id: 10 * major + minor)
    assert launcher.is_available(plain)
    if 10 * major + minor < launcher.sizes(fp8.name)[0].arch:
        assert not launcher.is_available(fp8)
        assert f"compute capability {major}.{minor}" in launcher.unavailable_reason(fp8)
    targets = runtime.targets_of(launcher.sizes(plain.name)[0])
    monkeypatch.setattr(runtime, "compute_capability", lambda device_id: 90)
    if all(capability[0] != 9 for capability in map(runtime.target_capability, targets)):
        assert "compute capability 9.0" in launcher.unavailable_reason(plain)


def test_cases_cover_every_built_kernel():
    declare_gpu_work(0)
    cases = [(norm_options(*norm), widths_for(norm[0], norm[4])) for norm in NORMS]
    cases += [(adaptive_options(*adaptive), WIDTHS) for adaptive in ADAPTIVE]
    cases += [(dual_options(*dual), WIDTHS) for dual in DUAL]
    cases += [(fp8_options(*fp8), BLOCK_WIDTHS if fp8[1] == "block128" else FP8_WIDTHS) for fp8 in FP8_CASES]
    covered = {launcher.size_for(options, width, engine, chunks).kernel for options, widths in cases
               for width in widths for engine in launcher.ENGINES[1:] if launcher.sizes(options.name, engine)
               for chunks in (None, 4 if options.fp8 else None,
                              *((6, 8) if options.fp8 and options.add == "none" else ()))}
    _, entries = runtime.manifest()
    missing = [name for name in entries if name.startswith("rms_norm_") and name not in covered]
    assert not missing, f"no test runs {missing}"
