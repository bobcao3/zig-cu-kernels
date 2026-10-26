"""Compares the `mma_check_sm90` kernels, which copy FP8 tiles with tensor copies and multiply them with `wgmma`, with
the torch product. Integer-valued E4M3 operands make every FP32 sum exact, so any difference is a layout error."""

import pytest
import torch

from conftest import declare_gpu_work
from zig_cu_kernels import runtime, tensormap

K_BLOCK = 128
K_BLOCKS = 4
CHECKS = ((128, 1, 1), (256, 1, 1), (256, 2, 1), (256, 2, 2), (128, 2, 2))


def kernel_name(n, consumers, ctas):
    return f"mma_check_sm90_n{n}_c{consumers}_x{ctas}"


def integers(rows, columns, generator):
    values = torch.randint(-4, 5, (rows, columns), device="cuda", generator=generator)
    return values.float().to(torch.float8_e4m3fn)


def product(n, consumers, ctas, seed=0):
    """Runs the kernel on one cluster of `ctas` work groups and returns the operands and the result."""
    kernel, info = runtime.load(kernel_name(n, consumers, ctas))
    runtime.allow_dynamic_shared(kernel, torch.cuda.current_device())
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = integers(64 * consumers * ctas, K_BLOCK * K_BLOCKS, generator)
    b = integers(n, K_BLOCK * K_BLOCKS, generator)
    out = torch.full((a.shape[0], n), float("nan"), device="cuda")
    a_map = tensormap.tensor_map(a, (64 * consumers, K_BLOCK))
    b_map = tensormap.tensor_map(b, (n // ctas, K_BLOCK))
    runtime.run(kernel, (ctas, 1, 1), info.threads, info.shared_bytes, "mmpu",
                (a_map, b_map, out.data_ptr(), K_BLOCKS), torch.cuda.current_stream().cuda_stream,
                cluster=(ctas, 1, 1) if ctas > 1 else None)
    return a, b, out


@pytest.mark.parametrize("n, consumers, ctas", CHECKS)
def test_wgmma_tiles_equal_the_torch_product(n, consumers, ctas):
    reason = runtime.unavailable_reason(kernel_name(n, consumers, ctas))
    if reason is not None:
        pytest.skip(reason)
    declare_gpu_work(1)
    a, b, out = product(n, consumers, ctas)
    expected = a.float() @ b.float().T
    wrong = (out != expected).nonzero()
    assert len(wrong) == 0, f"{len(wrong)} of {out.numel()} elements differ, first at {wrong[:4].tolist()}"
