import pytest
import torch

import fp8_mm_references as references
from zig_cu_kernels import fp8_mm, qkv_prep, runtime

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or fp8_mm.arch_of(runtime.compute_capability(0)) != 89,
                                reason="the QKV kernels run on compute capability 8.9")


def sums_in_kernel_order(x):
    """Sums the BF16 squares of each head as `zig/qkv_prep.zig` does: 8 values per lane from the lowest column up,
    then a butterfly over the lanes."""
    squares = (x * x).bfloat16().float().unflatten(-1, (-1, 8))
    partial = squares[..., 0]
    for j in range(1, 8):
        partial = partial + squares[..., j]
    lanes = torch.arange(partial.shape[-1])
    mask = 1
    while mask < partial.shape[-1]:
        partial = partial + partial[..., lanes ^ mask]
        mask *= 2
    return partial[..., :1]


def heads_in_kernel_order(plane, scales, rope, eps, shift):
    x = plane.float()
    mean = sums_in_kernel_order(x) * (1.0 / x.shape[-1])
    root = torch.rsqrt((mean + eps).double()) * (1 + shift)
    coefficient = (scales.float()[:, None] * root.float()).bfloat16().float()
    return references.rope_pairs((x * coefficient).bfloat16(), rope)


@pytest.mark.parametrize("m, heads, dim, rotated", [(37, 4, 64, 64), (64, 2, 128, 64), (5, 1, 256, 256),
                                                    (300, 2, 256, 96)])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.bfloat16])
def test_heads_match_the_kernel_order_bitwise(m, heads, dim, rotated, scale_dtype):
    generator = torch.Generator().manual_seed(m * dim + rotated)
    attn_dim = heads * dim
    z = (torch.randn((m, 3 * attn_dim), generator=generator) * 3).bfloat16()
    z[0, : dim] = 0
    scales = (torch.rand(heads, generator=generator) + 0.5).to(scale_dtype)
    rope = torch.randn((1, m, 1, rotated // 2, 2), generator=generator)
    q, kv = torch.empty((m, attn_dim), dtype=torch.bfloat16, device="cuda"), \
        torch.empty((m, 2 * attn_dim), dtype=torch.bfloat16, device="cuda")
    qkv_prep.mxfp8_qkv_prep(z.cuda(), q, kv, scales.cuda(), rope.cuda(), 1e-6, dim,
                            torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    q, kv = q.cpu(), kv.cpu()
    assert torch.equal(kv[:, attn_dim:], z[:, 2 * attn_dim:])
    planes = z.view(m, 3, heads, dim).unbind(1)
    rows = rope.reshape(m, -1)
    shifts = (-references.RSQRT_RELATIVE_ERROR, 0.0, references.RSQRT_RELATIVE_ERROR)
    for result, plane in ((q, planes[0]), (kv[:, :attn_dim], planes[1])):
        candidates = torch.stack([heads_in_kernel_order(plane, scales, rows, 1e-6, shift) for shift in shifts])
        matches = (candidates.view(torch.int16) == result.view(m, heads, dim).view(torch.int16)).all(-1).any(0)
        assert matches.all(), matches.logical_not().nonzero()[:8]
