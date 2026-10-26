import pytest
import torch

from zig_cu_kernels import runtime
from zig_cu_kernels.torch import adaptive_rms_norm, fp8_gemm, fused_add_rms_norm, rms_norm, rms_norm_fp8_quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
FP8_UNAVAILABLE = (runtime.unavailable_reason("fp8_gemm_rowwise_store_128x256")
                   or runtime.unavailable_reason("rms_norm_bf16_weight_bf16_to_fp8_rowwise_c2_max16384")
                   if torch.cuda.is_available() else "needs a CUDA device")
requires_fp8 = pytest.mark.skipif(FP8_UNAVAILABLE is not None, reason=FP8_UNAVAILABLE or "")

ROWS, WIDTH = 6, 256


def rows_with_stride(stride, offset=0):
    """Returns `[ROWS, WIDTH]` BF16 rows at the row stride `stride`, starting `offset` elements into the storage."""
    generator = torch.Generator(device="cuda").manual_seed(stride + offset)
    storage = torch.randn(ROWS * stride + offset, device="cuda", dtype=torch.bfloat16, generator=generator)
    return storage[offset:].as_strided((ROWS, WIDTH), (stride, 1))


@pytest.mark.parametrize("stride, offset", [(WIDTH, 0), (WIDTH + 8, 0), (WIDTH + 4, 0), (WIDTH, 1)])
def test_rms_norm_matches_for_layouts_in_place_and_copied(stride, offset):
    x = rows_with_stride(stride, offset)
    weight = torch.randn(WIDTH, device="cuda", dtype=torch.bfloat16)
    expected = rms_norm(x.contiguous(), weight, 1e-6)
    out = torch.empty(ROWS, stride, device="cuda", dtype=torch.bfloat16)[:, :WIDTH]
    assert torch.equal(rms_norm(x, weight, 1e-6, out=out), expected)
    assert torch.equal(rms_norm(x, weight, 1e-6), expected)


def test_rms_norm_output_that_overlaps_input_goes_through_a_temporary():
    storage = torch.randn((ROWS + 1) * WIDTH, device="cuda", dtype=torch.bfloat16)
    x = storage[: ROWS * WIDTH].view(ROWS, WIDTH)
    expected = rms_norm(x.clone(), None, 1e-6)
    out = storage[WIDTH:].view(ROWS, WIDTH)
    rms_norm(x, None, 1e-6, out=out)
    assert torch.equal(out, expected)


def test_fused_add_writes_exact_aliases_in_place():
    x = torch.randn(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(WIDTH, device="cuda", dtype=torch.bfloat16)
    expected_y, expected_sum = fused_add_rms_norm(x, residual, weight, 1e-6)
    y, total = fused_add_rms_norm(x, residual, weight, 1e-6, out=x, residual_out=residual)
    assert y.data_ptr() == x.data_ptr() and total.data_ptr() == residual.data_ptr()
    assert torch.equal(y, expected_y) and torch.equal(total, expected_sum)


@requires_fp8
@pytest.mark.parametrize("scaling", ["rowwise", "block128"])
def test_fp8_quant_scales_with_any_accepted_stride(scaling):
    x = torch.randn(ROWS, WIDTH, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(WIDTH, device="cuda", dtype=torch.bfloat16)
    expected_q, expected_scales = rms_norm_fp8_quant(x, weight, 1e-6, fp8_scaling=scaling)
    groups = 1 if scaling == "rowwise" else WIDTH // 128
    scales = torch.empty(groups, ROWS + 4, device="cuda").t()[:ROWS] if scaling == "block128" else \
        torch.empty(ROWS, 3, device="cuda")[:, :1]
    q, written = rms_norm_fp8_quant(x, weight, 1e-6, fp8_scaling=scaling, scales_out=scales)
    assert written.data_ptr() == scales.data_ptr()
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(written, expected_scales)


@requires_fp8
@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e8m0fnu])
def test_fp8_gemm_block_scales_in_both_layouts(dtype):
    m, n, k = 16, 128, 256
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.rand(m, k // 128, device="cuda").add(0.5).to(dtype)
    scale_b = torch.rand(n // 128, k // 128, device="cuda").add(0.5).to(dtype)
    expected = fp8_gemm(a, b, scale_a, scale_b)
    transposed = scale_a.t().contiguous().t()
    assert torch.equal(fp8_gemm(a, b, transposed, scale_b), expected)


def bf16(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


NORM_REJECTIONS = {
    "residual_dtype": lambda: fused_add_rms_norm(bf16(ROWS, WIDTH), bf16(ROWS, WIDTH).float(), None, 1e-6),
    "residual_shape": lambda: fused_add_rms_norm(bf16(ROWS, WIDTH), bf16(ROWS - 1, WIDTH), None, 1e-6),
    "weight_shape": lambda: rms_norm(bf16(ROWS, WIDTH), bf16(WIDTH - 8), 1e-6),
    "row_length": lambda: rms_norm(bf16(ROWS, WIDTH - 4), None, 1e-6),
    "row_length_of_one_row": lambda: rms_norm(bf16(1, WIDTH - 4), None, 1e-6),
    "block_row_length": lambda: rms_norm_fp8_quant(bf16(ROWS, WIDTH - 64), bf16(WIDTH - 64), 1e-6,
                                                   fp8_scaling="block128"),
    "rows_per_sample": lambda: adaptive_rms_norm(bf16(ROWS, WIDTH), bf16(1, WIDTH), eps=1e-6, rows_per_sample=0),
    "modulation_rows": lambda: adaptive_rms_norm(bf16(ROWS, WIDTH), bf16(1, WIDTH), eps=1e-6, rows_per_sample=2),
    "out_shape": lambda: rms_norm(bf16(ROWS, WIDTH), None, 1e-6, out=bf16(ROWS - 1, WIDTH)),
}


@pytest.mark.parametrize("case", NORM_REJECTIONS)
def test_rms_norm_rejects_each_checked_rule(case):
    with pytest.raises(ValueError):
        NORM_REJECTIONS[case]()


def test_fused_add_rejects_outputs_that_share_memory():
    x, residual = bf16(ROWS, WIDTH), bf16(ROWS, WIDTH)
    with pytest.raises(ValueError):
        fused_add_rms_norm(x, residual, None, 1e-6, out=x, residual_out=x)


def fp8(*shape):
    return torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)


def one():
    return torch.ones((), device="cuda")


GEMM_REJECTIONS = {
    "k_multiple": lambda: fp8_gemm(fp8(16, 136), fp8(128, 136), one(), one()),
    "weight_shape": lambda: fp8_gemm(fp8(16, 256), fp8(128, 128), one(), one()),
    "scale_dtype": lambda: fp8_gemm(fp8(16, 256), fp8(128, 256), one().half(), one().half()),
    "fast_accum_block": lambda: fp8_gemm(fp8(16, 256), fp8(128, 256), torch.ones(16, 2, device="cuda"),
                                         torch.ones(1, 2, device="cuda"), use_fast_accum=True),
}


@pytest.mark.parametrize("case", GEMM_REJECTIONS)
def test_fp8_gemm_rejects_each_checked_rule(case):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("needs compute capability 8.9")
    with pytest.raises(ValueError):
        GEMM_REJECTIONS[case]()
