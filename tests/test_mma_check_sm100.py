"""Compares the `tcgen05.mma` FP8 check kernels with PyTorch on small integer-valued operands, whose sums are
exact in FP32."""

import pytest
import torch

from zig_cu_kernels import packs, runtime, tensormap

K = 128
N = 256
BLOCKS = K // packs.MXFP8_BLOCK


def kernel_or_skip(name):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    reason = runtime.unavailable_reason(name)
    if reason is not None:
        pytest.skip(reason)
    return runtime.load(name)


def integer_e4m3(rows, generator):
    values = torch.randint(-4, 5, (rows, K), generator=generator, dtype=torch.int32)
    return values.float().to(torch.float8_e4m3fn).cuda()


def scale_bytes(rows, generator):
    return torch.randint(126, 130, (rows, BLOCKS), generator=generator, dtype=torch.int32).to(torch.uint8)


def run(name, rows, arguments, ctas=1, cluster=None):
    kernel, info = kernel_or_skip(name)
    d = torch.full((rows, N), float("nan"), device="cuda")
    signature = "m" * 2 + "p" * (len(arguments) - 1)
    stream = torch.cuda.current_stream().cuda_stream
    runtime.run(kernel, (ctas, 1, 1), info.threads, info.shared_bytes, signature, (*arguments, d.data_ptr()),
                stream, cluster=cluster)
    torch.cuda.synchronize()
    return d


def assert_equal_by_tiles(d, expected):
    """Names the mismatched elements of each 128x128 tile, so that a failure shows which operand rows it hits."""
    mismatches = (d != expected).view(d.shape[0] // 128, 128, d.shape[1] // 128, 128).sum(dim=(1, 3))
    assert not mismatches.any(), f"mismatched elements per 128x128 tile:\n{mismatches.tolist()}\nfirst rows of " \
        f"the result:\n{d[:2, :8].tolist()}\nand of the expected values:\n{expected[:2, :8].tolist()}"


def test_one_cta_f8f6f4_matches_torch():
    kernel_or_skip("mma_check_sm100_f8f6f4")
    generator = torch.Generator().manual_seed(1)
    a, b = integer_e4m3(128, generator), integer_e4m3(N, generator)
    maps = (tensormap.tensor_map(a, (128, K)), tensormap.tensor_map(b, (N, K)))
    d = run("mma_check_sm100_f8f6f4", 128, maps)
    assert_equal_by_tiles(d, a.float() @ b.float().T)


def test_one_cta_block_scaled_mxf8f6f4_matches_the_dequantized_product():
    kernel_or_skip("mma_check_sm100_mxf8f6f4")
    generator = torch.Generator().manual_seed(2)
    a, b = integer_e4m3(128, generator), integer_e4m3(N, generator)
    a_scales = packs.scale_atoms(scale_bytes(128, generator), 127).cuda()
    b_scales = packs.scale_atoms(scale_bytes(N, generator), 127).cuda()
    atoms = torch.cat([a_scales.flatten(), b_scales.flatten()]).contiguous()
    maps = (tensormap.tensor_map(a, (128, K)), tensormap.tensor_map(b, (N, K)))
    d = run("mma_check_sm100_mxf8f6f4", 128, (*maps, atoms.data_ptr()))
    expected = packs.dequantize_mxfp8(a, a_scales) @ packs.dequantize_mxfp8(b, b_scales).T
    assert_equal_by_tiles(d, expected)


def test_cta_pair_f8f6f4_matches_torch():
    kernel_or_skip("mma_check_sm100_pair_f8f6f4")
    generator = torch.Generator().manual_seed(3)
    a, b = integer_e4m3(256, generator), integer_e4m3(N, generator)
    maps = (tensormap.tensor_map(a, (128, K)), tensormap.tensor_map(b, (128, K)))
    d = run("mma_check_sm100_pair_f8f6f4", 256, maps, ctas=2, cluster=(2, 1, 1))
    assert_equal_by_tiles(d, a.float() @ b.float().T)
