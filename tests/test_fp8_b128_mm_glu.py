import pytest
import torch

import pack_references
from zig_cu_kernels import fp8_gemm, glu_quant

SM89 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)
if SM89:
    from zig_cu_kernels.torch import fp8_mm as ops


@pytest.mark.skipif(not SM89, reason="the fused block128 GLU kernel needs compute capability 8.9")
@pytest.mark.parametrize("e8m0", [True, False])
@pytest.mark.parametrize("activation", ["swiglu", "geglu_tanh"])
@pytest.mark.parametrize("m, hidden, k", [(1000, 1536, 1024), (2125, 640, 512)])
def test_fused_glu_is_bitwise_equal_to_the_unfused_chain(e8m0, activation, m, hidden, k):
    assert fp8_gemm.store_tiling("block128", m, 2 * hidden, k) in fp8_gemm.GLU_TILINGS
    generator = torch.Generator().manual_seed(m + hidden + k)
    x = torch.randn((m, k), generator=generator) * 3
    w = torch.randn((2 * hidden, k), generator=generator) * 0.05
    a, s_a = (t.cuda() for t in pack_references.fp8_b128_quant(x, e8m0=e8m0))
    w1, s_w1 = (t.cuda() for t in pack_references.fp8_b128_quant_weight(w, e8m0=e8m0))
    no_z, h_alone, s_alone = ops.fp8_b128_mm_glu(a, s_a, w1, s_w1, activation, False)
    z_full = ops.fp8_b128_mm_abt(a, s_a, w1, s_w1)
    z, h, s_h = torch.empty_like(z_full), torch.empty_like(h_alone), torch.full_like(s_alone, 0)
    gate = 0 if activation == "geglu_tanh" else 1
    halves = [(w1[:hidden], s_w1[:hidden // 128, :k // 128]), (w1[hidden:], s_w1[hidden // 128:, :k // 128])]
    z_halves = (z[:, :hidden], z[:, hidden:])
    view = (lambda s: s.view(torch.float8_e8m0fnu)) if e8m0 else (lambda s: s)
    if hidden // 128 % 4:
        s_h.zero_()
    fp8_gemm.fp8_b128_glu_gemm(a, halves[gate][0], halves[1 - gate][0], view(s_a), view(halves[gate][1]),
                               view(halves[1 - gate][1]), h, view(s_h), activation=activation, z_gate=z_halves[gate],
                               z_up=z_halves[1 - gate])
    h_chain = torch.empty_like(h)
    s_chain = torch.full_like(s_h.view(torch.uint8), 0x55).view(s_h.dtype)
    recipe = "fp8_b128_e8m0" if e8m0 else "fp8_b128_fp32"
    glu_quant.glu_quant(z_full, h_chain, s_chain, recipe, activation, activation == "geglu_tanh", None, hidden,
                        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert no_z is None and torch.equal(z.view(torch.int16), z_full.view(torch.int16))
    for codes, scales in ((h, s_h), (h_alone, s_alone)):
        assert torch.equal(codes.view(torch.uint8), h_chain.view(torch.uint8))
        assert torch.equal(scales.view(torch.uint8), s_chain.view(torch.uint8))
