"""Bitwise checks of the SM100 `mm_abt`, `mm_abt_low_rank` and `mm_atb` kernels with integer operands and power-of-two
scales on 90 tiles, more than the 74 clusters that the launcher starts on a B300, so each cluster of a persistent kernel
computes several tiles. `clusters` limits the grid of the persistent kernels to that many clusters."""

import pytest
import torch

import fp8_mm_references as references
from conftest import declare_gpu_work
from zig_cu_kernels import fp8_mm as launcher
from zig_cu_kernels import fp8_pack, mm_sm100, packs

CUDA = torch.cuda.is_available()
if CUDA:
    from zig_cu_kernels.torch import fp8_mm as ops

E8M0_ONE = 127
M, N = 2500, 2304


def generator(seed):
    return torch.Generator().manual_seed(seed)


def integers(shape, seed):
    return torch.randint(-4, 5, shape, generator=generator(seed)).float().to(torch.float8_e4m3fn)


def exponents(shape, seed):
    return torch.randint(-2, 3, shape, generator=generator(seed))


def require_sm100(recipe, clusters, monkeypatch, operation="mm_abt"):
    if not CUDA or torch.cuda.get_device_capability()[0] != 10 or not launcher.available(recipe, operation):
        pytest.skip(f"no SM100 {recipe} {operation} kernel on this device")
    if clusters is not None:
        monkeypatch.setattr(mm_sm100, "resident_clusters", lambda _: clusters)


walks = pytest.mark.parametrize("clusters", [None, 7])


def test_the_shapes_have_more_tiles_than_a_b300_has_clusters():
    assert packs.ceil_div(M, mm_sm100.TILE) * packs.ceil_div(N, mm_sm100.TILE) > 148 // mm_sm100.CLUSTER[0]


@walks
def test_fp8_b128_is_bitwise_with_integer_operands(clusters, monkeypatch):
    require_sm100("fp8_b128_e8m0", clusters, monkeypatch)
    declare_gpu_work(1)
    k = 640
    a, b = integers((M, k), 1), integers((N, k), 2)
    s_a = packs.scale_atoms(exponents((M, k // 128), 3).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    s_b = exponents((N // 128, 4 * packs.ceil_div(k, 512)), 4).add(E8M0_ONE).to(torch.uint8)
    exact = references.product(references.dequantized(a, s_a, "fp8_b128"), references.dequantized(b, s_b, "fp8_b128"))
    y = ops.fp8_b128_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda())
    assert torch.equal(y.cpu(), exact[0].float().bfloat16())


@walks
@pytest.mark.parametrize("bias", [False, True])
def test_fp8_rowwise_is_bitwise_with_integer_operands(bias, clusters, monkeypatch):
    require_sm100("fp8_rowwise", clusters, monkeypatch)
    declare_gpu_work(1)
    k = 528
    a, b = integers((M, k), 5), integers((N, k), 6)
    s_a, s_b = torch.pow(2.0, exponents((M, 1), 7).float()), torch.pow(2.0, exponents((N, 1), 8).float())
    terms = torch.randint(-64, 65, (N,), generator=generator(9)).float().bfloat16() if bias else None
    acc = (a.double() @ b.double().T) * s_a.double() * s_b.double().T
    y = ops.fp8_rowwise_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda(), None if terms is None else terms.cuda())
    assert torch.equal(y.cpu(), references.bf16_store(acc.float(), terms))


@walks
def test_mxfp8_is_bitwise_with_integer_operands(clusters, monkeypatch):
    require_sm100("mxfp8", clusters, monkeypatch)
    declare_gpu_work(1)
    k = 640
    a, b = integers((M, k), 10), integers((N, k), 11)
    s_a = packs.scale_atoms(exponents((M, k // 32), 12).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    s_b = packs.scale_atoms(exponents((N, k // 32), 13).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    exact = references.product(references.dequantized(a, s_a, "mxfp8"), references.dequantized(b, s_b, "mxfp8"))
    y = ops.mxfp8_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda())
    assert torch.equal(y.cpu(), exact[0].float().bfloat16())


def columnwise_pack(k, m, seed):
    """Returns the values `[k, m]` and the scales `[ceil(k/128), m/128, 32, 16]` of a columnwise MXFP8 pack, whose
    exponents in [-1, 1] keep every sum of the products exact in FP32 up to K = 16384."""
    values = integers((k, m), seed)
    grid = torch.randint(-1, 2, (m, packs.ceil_div(k, 32)), generator=generator(seed + 1)).add(E8M0_ONE)
    return values, packs.scale_atoms(grid.to(torch.uint8), E8M0_ONE).permute(1, 0, 2, 3).contiguous()


@pytest.mark.parametrize("k, m, n, width, clusters", [(200, 2432, 2304, 2296, None), (200, 2432, 2304, 2304, 7),
                                                      (12289, 384, 512, 448, None), (128, 256, 256, 256, None)])
def test_mxfp8_mm_atb_is_bitwise_with_integer_operands(k, m, n, width, clusters, monkeypatch):
    require_sm100("mxfp8", clusters, monkeypatch, "mm_atb")
    declare_gpu_work(1)
    a, s_a = columnwise_pack(k, m, 20)
    b, s_b = columnwise_pack(k, n, 22)
    exact = references.dequantized_columnwise(a, s_a).t() @ references.dequantized_columnwise(b, s_b)[:, :width]
    d = ops.mxfp8_mm_atb(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda(), width)
    assert torch.equal(d.cpu(), exact.float().bfloat16())


@walks
@pytest.mark.parametrize("rank", launcher.LOW_RANKS)
def test_mxfp8_low_rank_is_bitwise_with_integer_operands(rank, clusters, monkeypatch):
    require_sm100("mxfp8", clusters, monkeypatch)
    declare_gpu_work(1)
    k, width = 640, N - 8
    a, b = integers((M, k), 30), integers((N, k), 31)
    s_a = packs.scale_atoms(exponents((M, k // 32), 32).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    s_b = packs.scale_atoms(exponents((N, k // 32), 33).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    low, up = integers((M, rank), 34).bfloat16(), integers((width, rank), 35).bfloat16()
    bias = torch.randint(-64, 65, (width,), generator=generator(36)).float().bfloat16()
    exact = references.product(references.dequantized(a, s_a, "mxfp8"), references.dequantized(b, s_b, "mxfp8"))
    exact = exact[0][:, :width] + low.double() @ up.double().T
    y = ops.mxfp8_mm_abt_low_rank(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda(), low.cuda(), up.cuda(), bias.cuda(), width)
    assert torch.equal(y.cpu(), references.bf16_store(exact.float(), bias))


@pytest.mark.parametrize(("activation", "gate_first", "clamp_limit"), [("swiglu", False, None), ("swiglu", True, 7.0),
                                                                       ("geglu_tanh", True, None)])
@pytest.mark.parametrize(("m", "hidden", "clusters"), [(300, 256, None), (130, 200, None), (1300, 1000, 7)])
@pytest.mark.parametrize("out_z", [False, True])
def test_mxfp8_fused_glu_is_bitwise_with_the_composed_chain(m, hidden, clusters, activation, gate_first, clamp_limit,
                                                            out_z, monkeypatch):
    require_sm100("mxfp8", clusters, monkeypatch, "mm_glu")
    declare_gpu_work(2)
    k, padded = 512, packs.round_up(hidden, 128)
    x = torch.randn((m, k), generator=generator(40)).cuda().bfloat16()
    w = (torch.randn((2 * padded, k), generator=generator(41)) * 0.05).cuda().bfloat16()
    a, s_a = fp8_pack.mxfp8_quant(x)
    w1, s_w1 = fp8_pack.mxfp8_quant(w)
    arguments = (a, s_a, w1, s_w1, gate_first, out_z, clamp_limit, activation, hidden)
    z, h, s_h = ops.mxfp8_mm_glu(*arguments)
    monkeypatch.setattr(mm_sm100, "FUSED_GLU", frozenset())
    z_chain, h_chain, s_h_chain = ops.mxfp8_mm_glu(*arguments)
    assert torch.equal(h.view(torch.uint8), h_chain.view(torch.uint8))
    assert torch.equal(s_h, s_h_chain)
    assert (z is None) == (not out_z)
    if out_z:
        assert torch.equal(z.view(torch.int16), z_chain.view(torch.int16))
