import pytest
import torch

import pack_references
from zig_cu_kernels import fp8_gemm, glu_quant

SM89 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)
if SM89:
    from zig_cu_kernels.torch import fp8_mm as ops


def mxfp8_packs(m, hidden, k):
    padded = -(-hidden // 128) * 128
    generator = torch.Generator().manual_seed(m + hidden + k)
    x = torch.randn((m, k), generator=generator) * 3
    w = torch.randn((2 * padded, k), generator=generator) * 0.05
    return padded, *(t.cuda() for t in (*pack_references.mxfp8_quant(x), *pack_references.mxfp8_quant(w)))


@pytest.mark.skipif(not SM89, reason="the fused MXFP8 GLU kernel needs compute capability 8.9")
@pytest.mark.parametrize("activation, gate_first, clamp_limit", [("swiglu", False, None), ("swiglu", True, 7.0),
                                                                 ("geglu_tanh", True, None)])
@pytest.mark.parametrize("m, hidden, k", [(1000, 1536, 1024), (2125, 600, 512)])
def test_mxfp8_fused_glu_is_bitwise_equal_to_the_unfused_chain(activation, gate_first, clamp_limit, m, hidden, k):
    padded, a, s_a, w1, s_w1 = mxfp8_packs(m, hidden, k)
    assert fp8_gemm.store_tiling("mxfp8", m, 2 * padded, k) in fp8_gemm.FAMILY_GLU_TILINGS["mxfp8"]
    no_z, h_alone, s_alone = ops.mxfp8_mm_glu(a, s_a, w1, s_w1, gate_first, False, clamp_limit, activation, hidden)
    z_composed, h_composed, s_composed = ops.mxfp8_mm_glu(a, s_a, w1, s_w1, gate_first, True, clamp_limit,
                                                          activation, hidden)
    z_full = ops.mxfp8_mm_abt(a, s_a, w1, s_w1)
    z, h, s_h = torch.empty_like(z_full), torch.empty_like(h_alone), torch.full_like(s_alone.view(torch.uint8), 0x55)
    halves = [(w1[:padded], s_w1[:padded // 128]), (w1[padded:], s_w1[padded // 128:])]
    z_halves = (z[:, :padded], z[:, padded:])
    gate = 0 if gate_first else 1
    e8m0 = torch.float8_e8m0fnu
    fp8_gemm.mxfp8_glu_gemm(a, halves[gate][0], halves[1 - gate][0], s_a.view(e8m0), halves[gate][1].view(e8m0),
                            halves[1 - gate][1].view(e8m0), h, s_h.view(e8m0), activation=activation, hidden=hidden,
                            clamp_limit=glu_quant.bf16_limit(clamp_limit), z_gate=z_halves[gate],
                            z_up=z_halves[1 - gate])
    h_chain = torch.empty_like(h)
    s_chain = torch.full_like(s_h, 0x55)
    glu_quant.glu_quant(z_full, h_chain, s_chain, "mxfp8", activation, gate_first, clamp_limit, hidden,
                        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert no_z is None
    assert torch.equal(z.view(torch.int16), z_full.view(torch.int16))
    assert torch.equal(z_composed[:, :hidden].view(torch.int16), z_full[:, :hidden].view(torch.int16))
    for codes, scales in ((h, s_h), (h_alone, s_alone), (h_composed, s_composed)):
        assert torch.equal(codes.view(torch.uint8), h_chain.view(torch.uint8))
        assert torch.equal(scales.view(torch.uint8), s_chain)


@pytest.mark.skipif(not SM89, reason="the fused rowwise GLU kernel needs compute capability 8.9")
@pytest.mark.parametrize("activation", ["swiglu", "geglu_tanh"])
@pytest.mark.parametrize("m, hidden, k", [(1000, 1536, 1024), (2125, 640, 512)])
def test_rowwise_fused_glu_is_bitwise_equal_to_the_unfused_chain(activation, m, hidden, k):
    assert fp8_gemm.store_tiling("rowwise", m, 2 * hidden, k) in fp8_gemm.GLU_TILINGS
    generator = torch.Generator().manual_seed(m + hidden + k)
    x = torch.randn((m, k), generator=generator) * 3
    w = torch.randn((2 * hidden, k), generator=generator) * 0.05
    a, s_a, w1, s_w1 = (t.cuda() for t in (*pack_references.fp8_rowwise_quant(x),
                                           *pack_references.fp8_rowwise_quant(w)))
    no_z, h_alone, s_alone = ops.fp8_rowwise_mm_glu(a, s_a, w1, s_w1, activation, False)
    z_composed, h_composed, s_composed = ops.fp8_rowwise_mm_glu(a, s_a, w1, s_w1, activation, True)
    z_full = ops.fp8_rowwise_mm_abt(a, s_a, w1, s_w1, None)
    gate_first = activation == "geglu_tanh"
    halves = [(w1[:hidden], s_w1[:hidden]), (w1[hidden:], s_w1[hidden:])]
    gate = 0 if gate_first else 1
    z = torch.empty_like(z_full)
    h_bf16 = torch.empty((m, hidden), dtype=torch.bfloat16, device="cuda")
    fp8_gemm.fp8_rowwise_glu_gemm(a, halves[gate][0], halves[1 - gate][0], s_a, halves[gate][1], halves[1 - gate][1],
                                  h_bf16, activation=activation, z_gate=z[:, gate * hidden:(gate + 1) * hidden],
                                  z_up=z[:, (1 - gate) * hidden:(2 - gate) * hidden])
    h_chain, s_chain = torch.empty_like(h_alone), torch.empty_like(s_alone)
    glu_quant.glu_quant(z_full, h_chain, s_chain, "fp8_rowwise", activation, gate_first, None, hidden,
                        torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert no_z is None
    assert torch.equal(z.view(torch.int16), z_full.view(torch.int16))
    assert torch.equal(z_composed.view(torch.int16), z_full.view(torch.int16))
    for codes, scales in ((h_alone, s_alone), (h_composed, s_composed)):
        assert torch.equal(codes.view(torch.uint8), h_chain.view(torch.uint8))
        assert torch.equal(scales.view(torch.int32), s_chain.view(torch.int32))
