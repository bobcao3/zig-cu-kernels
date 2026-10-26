import math

import pytest
import torch

import fp8_mm_references as references
import pack_references
from zig_cu_kernels import fp8_mm, glu_quant, runtime

KINDS = {"mxfp8": "mxfp8", "fp8_b128_e8m0": "fp8_b128", "fp8_b128_fp32": "fp8_b128", "fp8_rowwise": "fp8_rowwise"}

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or fp8_mm.arch_of(runtime.compute_capability(0)) != 89,
                                reason="the GLU kernels run on compute capability 8.9")


def outputs(recipe, m, padded, hidden):
    h_scale = torch.uint8 if recipe != "fp8_b128_fp32" else torch.float32
    _, h, s_h = fp8_mm.glu_shapes(KINDS[recipe], m, padded, hidden, False, h_scale)
    return (torch.empty(h[0], dtype=h[1], device="cuda"), torch.zeros(s_h[0], dtype=s_h[1], device="cuda"))


@pytest.mark.parametrize("recipe", list(KINDS))
@pytest.mark.parametrize("activation, gate_first, clamp_limit", [("swiglu", False, None), ("swiglu", True, 3.0),
                                                                 ("geglu_tanh", True, None)])
@pytest.mark.parametrize("m, padded, hidden", [(300, 256, 256), (130, 384, 200), (1, 128, 128)])
def test_glu_packs_match_the_reference_where_the_bounds_agree(recipe, activation, gate_first, clamp_limit, m, padded,
                                                              hidden):
    if recipe != "mxfp8" and hidden % 128:
        pytest.skip("block128 and rowwise packs need hidden % 128 == 0")
    generator = torch.Generator().manual_seed(m + padded)
    z = (torch.randn((m, 2 * padded), generator=generator) * 4).bfloat16()
    z[0, :32] = 0
    h, s_h = outputs(recipe, m, padded, hidden)
    stream = torch.cuda.current_stream().cuda_stream
    glu_quant.glu_quant(z.cuda(), h, s_h, recipe, activation, gate_first, clamp_limit, hidden, stream)
    torch.cuda.synchronize()
    z_hidden = z if KINDS[recipe] == "mxfp8" else references.glu_z(z, hidden)
    (values, scales), _, agree = references.glu_pack(z_hidden, KINDS[recipe], activation, gate_first, clamp_limit,
                                                     hidden=hidden if KINDS[recipe] == "mxfp8" else None,
                                                     e8m0=recipe != "fp8_b128_fp32")
    assert agree.float().mean() > 0.5
    block = values.shape[1] // agree.shape[1]
    columns = agree.repeat_interleave(block, dim=1)
    assert torch.equal(h.cpu().view(torch.uint8)[columns], values.view(torch.uint8)[columns])
    if KINDS[recipe] == "fp8_rowwise":
        assert torch.equal(s_h.cpu()[agree], scales[agree])
    elif agree.all():
        assert torch.equal(s_h.cpu(), scales)



@pytest.mark.parametrize("hidden", [glu_quant.ONE_PASS_CHUNKS * 256 * glu_quant.CHUNK,
                                    glu_quant.ONE_PASS_CHUNKS * 256 * glu_quant.CHUNK + 128])
def test_rowwise_packs_of_wide_rows_match_the_bf16_chain(hidden):
    rows, stream = 4, torch.cuda.current_stream().cuda_stream
    z = (torch.randn((rows, 2 * hidden), generator=torch.Generator().manual_seed(hidden)) * 4).bfloat16().cuda()
    h, s_h = outputs("fp8_rowwise", rows, hidden, hidden)
    glu_quant.glu_quant(z, h, s_h, "fp8_rowwise", "swiglu", False, None, hidden, stream)
    h_bf16 = torch.empty((rows, hidden), dtype=torch.bfloat16, device="cuda")
    glu_quant.run("fp8_rowwise_glu_bf16_swiglu", (rows * hidden // glu_quant.CHUNK // 256, 1, 1),
                  glu_quant.ROW_SIGNATURE, (z.data_ptr(), h_bf16.data_ptr(), rows, hidden, hidden, 0, math.inf), stream)
    torch.cuda.synchronize()
    values, scales = pack_references.glu_h_quant(h_bf16.cpu(), "fp8_rowwise")
    assert torch.equal(h.cpu().view(torch.uint8), values.view(torch.uint8)) and torch.equal(s_h.cpu(), scales)
