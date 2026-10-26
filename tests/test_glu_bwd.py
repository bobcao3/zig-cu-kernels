import pytest
import torch

import fp8_mm_references as references
from zig_cu_kernels import glu_bwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the GLU backward kernels need a GPU")

VARIANTS = [("swiglu", False, None), ("swiglu", True, 3.0), ("swiglu", False, 0.5), ("geglu_tanh", True, None)]


def inputs(m, hidden, seed):
    generator = torch.Generator().manual_seed(seed)
    dh = torch.randn((m, hidden), generator=generator).bfloat16()
    z = (torch.randn((m, 2 * hidden), generator=generator) * 3).bfloat16()
    z[0, :32] = 0
    dh[-1, -32:] = 0
    if m >= 2:
        z[0, 0], z[0, hidden], z[1, 0], z[1, hidden] = 32.0, 32.0, -32.0, -32.0
    return dh, z


def as_bytes(x):
    return x.cpu().view(torch.uint8) if x.dtype != torch.uint8 else x.cpu()


@pytest.mark.parametrize("activation, gate_first, clamp_limit", VARIANTS)
@pytest.mark.parametrize("m, hidden", [(300, 256), (130, 192), (1, 64), (257, 128)])
def test_glu_backward_packs_match_the_reference_where_the_bounds_agree(activation, gate_first, clamp_limit, m, hidden):
    dh, z = inputs(m, hidden, m + hidden)
    outputs = glu_bwd.mxfp8_glu_backward(dh.cuda(), z.cuda(), gate_first, clamp_limit, activation)
    torch.cuda.synchronize()
    low, high = references.glu_backward(dh, z, activation, gate_first, clamp_limit)
    low_packs, high_packs = references.glu_backward_packs(*low), references.glu_backward_packs(*high)
    for name, output, low_pack, high_pack in zip(("dz_rw", "s_dz_rw", "dz_cw", "s_dz_cw", "h_cw", "s_h_cw"),
                                                 outputs, low_packs, high_packs):
        assert output.shape == low_pack.shape, name
        agree = as_bytes(low_pack) == as_bytes(high_pack)
        assert agree.float().mean() > 0.95, name
        assert torch.equal(as_bytes(output)[agree], as_bytes(low_pack)[agree]), name


def test_glu_backward_rejects_unsupported_arguments():
    dh, z = torch.zeros((4, 96), dtype=torch.bfloat16), torch.zeros((4, 192), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 64"):
        glu_bwd.check(dh, z, False, None, "swiglu")
    with pytest.raises(ValueError, match="gate_first=True"):
        glu_bwd.check(dh[:, :64], z[:, :128], False, None, "geglu_tanh")


def test_the_op_returns_the_launcher_packs_and_traces():
    import zig_cu_kernels.torch as ops

    dh, z = (x.cuda() for x in inputs(130, 192, 5))
    expected = glu_bwd.mxfp8_glu_backward(dh, z, True, 3.0, "swiglu")
    results = ops.mxfp8_glu_backward(dh, z, True, 3.0, "swiglu")
    compiled = torch.compile(ops.mxfp8_glu_backward, fullgraph=True)(dh, z, True, 3.0, "swiglu")
    for result, traced, packed in zip(results, compiled, expected):
        assert torch.equal(as_bytes(result), as_bytes(packed)) and torch.equal(as_bytes(traced), as_bytes(packed))
    torch.library.opcheck(ops.mxfp8_glu_backward_op, (dh, z, False, None, "swiglu"),
                          test_utils=("test_schema", "test_faketensor", "test_autograd_registration"))


@pytest.mark.parametrize("activation, gate_first, clamp_limit", VARIANTS)
@pytest.mark.parametrize("out_h, out_q", [(True, True), (False, False), (True, False)])
def test_bf16_outputs_match_the_operand_packs_and_the_reference(activation, gate_first, clamp_limit, out_h, out_q):
    import zig_cu_kernels.torch as ops

    dh, z = (x.cuda() for x in inputs(257, 192, 11))
    dz, dz_q, s_dz, h = ops.mxfp8_glu_backward_bf16(dh, z, gate_first, out_h, out_q, clamp_limit, activation)
    packed = glu_bwd.mxfp8_glu_backward(dh, z, gate_first, clamp_limit, activation)
    torch.cuda.synchronize()
    expected = references.glu_backward_packs(dz.cpu(), h.cpu() if out_h else dz.cpu()[:, :192])
    for output, reference in zip(packed[:4], expected[:4]):
        assert torch.equal(as_bytes(output), as_bytes(reference))
    low, high = references.glu_backward(dh, z, activation, gate_first, clamp_limit)
    agree = low[0].view(torch.int16) == high[0].view(torch.int16)
    assert torch.equal(dz.cpu()[agree].view(torch.int16), low[0][agree].view(torch.int16))
    assert (dz_q is not None) == out_q and (h is not None) == out_h
    if out_q:
        assert torch.equal(as_bytes(dz_q), as_bytes(packed[0])) and torch.equal(s_dz.cpu(), packed[1].cpu())
    if out_h:
        assert torch.equal(as_bytes(packed[4]), as_bytes(expected[4]))


@pytest.mark.parametrize("activation, gate_first", [("swiglu", False), ("geglu_tanh", True)])
@pytest.mark.parametrize("dh_scale, z_scale", [(2.0**-120, 3.0), (1.0, 2.0**-60)])
def test_swiglu_flushes_subnormal_gradients_and_geglu_keeps_them(activation, gate_first, dh_scale, z_scale):
    dh, z = inputs(257, 256, 3)
    dh, z = (dh.float() * dh_scale).bfloat16(), (z.float() * z_scale).bfloat16()
    dz, _, _, h = glu_bwd.mxfp8_glu_backward_bf16(dh.cuda(), z.cuda(), gate_first, True, False, None, activation)
    packed = glu_bwd.mxfp8_glu_backward(dh.cuda(), z.cuda(), gate_first, None, activation)
    torch.cuda.synchronize()
    low, high = references.glu_backward(dh, z, activation, gate_first)
    for output, low_value, high_value in zip((dz.cpu(), h.cpu()), low, high):
        agree = low_value.view(torch.int16) == high_value.view(torch.int16)
        assert agree.float().mean() > 0.95
        assert torch.equal(output[agree].view(torch.int16), low_value[agree].view(torch.int16))
    subnormal = [((x.float().abs() < 2.0**-126) & (x != 0)).any().item() for x in (dz.cpu(), h.cpu())]
    assert any(subnormal) == (activation == "geglu_tanh")
    expected = references.glu_backward_packs(dz.cpu(), h.cpu())
    for output, reference in zip(packed, expected):
        assert torch.equal(as_bytes(output), as_bytes(reference))


def test_geglu_accepts_an_infinite_clamp_limit_and_cpu_inputs_are_rejected():
    dh, z = inputs(130, 192, 7)
    finite = glu_bwd.mxfp8_glu_backward(dh.cuda(), z.cuda(), True, None, "geglu_tanh")
    infinite = glu_bwd.mxfp8_glu_backward(dh.cuda(), z.cuda(), True, float("inf"), "geglu_tanh")
    for a, b in zip(finite, infinite):
        assert torch.equal(as_bytes(a), as_bytes(b))
    with pytest.raises(ValueError, match="CUDA"):
        glu_bwd.mxfp8_glu_backward(dh, z, True, None, "geglu_tanh")
