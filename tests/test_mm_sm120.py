"""Bitwise checks of the SM120 `mm_abt` kernels with integer operands and power-of-two scales, whose sums FP32 holds
exactly, on shapes whose last row tile is partial. The tile `split` computes the last row band with 128x128 tiles
after 128x256 tiles, as `mm_abt` does when the wide tiles leave a nearly empty last wave. The tile `splitk` computes
the last row band in 3 K ranges whose FP32 partials the reduce kernel adds."""

import pytest
import torch

import fp8_mm_references as references
from conftest import declare_gpu_work
from zig_cu_kernels import fp8_mm as launcher
from zig_cu_kernels import decode, fp8_mm, fp8_pack, mm_sm120, packs

CUDA = torch.cuda.is_available()
if CUDA:
    from zig_cu_kernels.torch import fp8_mm as ops

E8M0_ONE = 127
SPLIT = "split"
SPLITK = "splitk"


def generator(seed):
    return torch.Generator().manual_seed(seed)


def integers(shape, seed):
    return torch.randint(-4, 5, shape, generator=generator(seed)).float().to(torch.float8_e4m3fn)


def exponents(shape, seed):
    return torch.randint(-2, 3, shape, generator=generator(seed))


def require_sm120(recipe, tile, monkeypatch):
    if not CUDA or torch.cuda.get_device_capability()[0] != 12 or not launcher.available(recipe, "mm_abt"):
        pytest.skip(f"no SM120 {recipe} mm_abt kernel on this device")
    monkeypatch.setitem(mm_sm120.TILES, recipe, mm_sm120.WIDE if tile in (SPLIT, SPLITK) else tile)
    monkeypatch.setattr(mm_sm120, "multiprocessors", lambda _: 7 if tile in (SPLIT, SPLITK) else 188)
    monkeypatch.setattr(mm_sm120, "SPLIT_RECIPES", frozenset({recipe} if tile == SPLITK else ()))
    launches = []
    if tile == SPLITK:
        monkeypatch.setattr(mm_sm120, "split_count", lambda m, n, padded_n, k, sms: 3)
        launch_split = mm_sm120.launch_split
        monkeypatch.setattr(mm_sm120, "launch_split", lambda *arguments: launches.append(launch_split(*arguments)))
    return launches


def check_launches(launches, tile):
    assert len(launches) == (tile == SPLITK)


tiles = pytest.mark.parametrize("tile", [mm_sm120.SQUARE, mm_sm120.WIDE, SPLIT, SPLITK])


@pytest.mark.parametrize(("m", "n", "sms", "tail"), [
    (8192, 8192, 188, 0), (4096, 12288, 188, 1), (4096, 16384, 188, 0), (16384, 5376, 188, 3), (390, 384, 188, 0),
    (390, 384, 7, 1), (1, 384, 7, 0),
])
def test_tail_bands_fill_the_last_wave(m, n, sms, tail):
    assert mm_sm120.tail_bands(m, n, sms) == tail


@pytest.mark.parametrize(("m", "n", "k", "splits"), [
    (8192, 8192, 8192, 0), (4096, 12288, 4096, 4), (4096, 16384, 5376, 0), (16384, 5376, 21504, 0), (4096, 12288, 4000, 0),
])
def test_split_count_splits_only_a_nearly_empty_last_wave(m, n, k, splits):
    assert mm_sm120.split_count(m, n, n, k, 188) == splits


def scales(shift, e8m0):
    return (shift + E8M0_ONE).to(torch.uint8) if e8m0 else torch.pow(2.0, shift.float())


@tiles
@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32"])
def test_fp8_b128_is_bitwise_with_integer_operands(recipe, tile, monkeypatch):
    launches = require_sm120(recipe, tile, monkeypatch)
    declare_gpu_work(1)
    m, n, k = 390, 384, 640
    e8m0 = recipe.endswith("e8m0")
    a, b = integers((m, k), 1), integers((n, k), 2)
    s_a = packs.scale_atoms(scales(exponents((m, k // 128), 3), e8m0), E8M0_ONE if e8m0 else 1.0)
    s_b = scales(exponents((n // 128, 4 * packs.ceil_div(k, 512)), 4), e8m0)
    exact = references.product(references.dequantized(a, s_a, "fp8_b128"), references.dequantized(b, s_b, "fp8_b128"))
    y = ops.fp8_b128_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda())
    assert torch.equal(y.cpu(), exact[0].float().bfloat16())
    check_launches(launches, tile)


@tiles
@pytest.mark.parametrize("bias", [False, True])
def test_fp8_rowwise_is_bitwise_with_integer_operands(bias, tile, monkeypatch):
    launches = require_sm120("fp8_rowwise", tile, monkeypatch)
    declare_gpu_work(1)
    m, n, k = (390, 384, 640) if tile == SPLITK else (390, 264, 528)
    a, b = integers((m, k), 5), integers((n, k), 6)
    s_a, s_b = scales(exponents((m, 1), 7), False), scales(exponents((n, 1), 8), False)
    terms = torch.randint(-64, 65, (n,), generator=generator(9)).float().bfloat16() if bias else None
    acc = (a.double() @ b.double().T) * s_a.double() * s_b.double().T
    y = ops.fp8_rowwise_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda(), None if terms is None else terms.cuda())
    assert torch.equal(y.cpu(), references.bf16_store(acc.float(), terms))
    check_launches(launches, tile)


@tiles
@pytest.mark.parametrize("m", [1, 390])
def test_mxfp8_is_bitwise_with_integer_operands(m, tile, monkeypatch):
    launches = require_sm120("mxfp8", tile, monkeypatch)
    declare_gpu_work(1)
    n, k = 384, 640
    a, b = integers((m, k), 10), integers((n, k), 11)
    s_a = packs.scale_atoms(exponents((m, k // 32), 12).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    s_b = packs.scale_atoms(exponents((n, k // 32), 13).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    exact = references.product(references.dequantized(a, s_a, "mxfp8"), references.dequantized(b, s_b, "mxfp8"))
    y = ops.mxfp8_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda())
    assert torch.equal(y.cpu(), exact[0].float().bfloat16())
    check_launches(launches, tile if m > decode.MAX_M else None)


@tiles
@pytest.mark.parametrize("rank", fp8_mm.LOW_RANKS)
def test_mxfp8_low_rank_is_bitwise_with_integer_operands(rank, tile, monkeypatch):
    require_sm120("mxfp8", tile, monkeypatch)
    declare_gpu_work(1)
    m, n, width, k = 390, 384, 376, 640
    a, b = integers((m, k), 14), integers((n, k), 15)
    s_a = packs.scale_atoms(exponents((m, k // 32), 16).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    s_b = packs.scale_atoms(exponents((n, k // 32), 17).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    low, up, bias = integers((m, rank), 18).bfloat16(), integers((width, rank), 19).bfloat16(), integers((width,), 20)
    exact = references.product(references.dequantized(a, s_a, "mxfp8"), references.dequantized(b, s_b, "mxfp8"))[0]
    expected = references.bf16_store((exact[:, :width] + low.double() @ up.double().T).float(), None, bias.bfloat16())
    y = ops.mxfp8_mm_abt_low_rank(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda(), low.cuda(), up.cuda(),
                                  bias.bfloat16().cuda(), width)
    assert torch.equal(y.cpu(), expected)


@pytest.mark.parametrize(("activation", "gate_first", "clamp_limit"), [("swiglu", False, None), ("swiglu", True, 7.0),
                                                                       ("geglu_tanh", True, None)])
@pytest.mark.parametrize(("m", "hidden"), [(300, 256), (130, 200)])
@pytest.mark.parametrize("out_z", [False, True])
def test_mxfp8_fused_glu_is_bitwise_with_the_composed_chain(m, hidden, activation, gate_first, clamp_limit, out_z,
                                                            monkeypatch):
    require_sm120("mxfp8", mm_sm120.WIDE, monkeypatch)
    declare_gpu_work(2)
    k, padded = 512, packs.round_up(hidden, 128)
    x = torch.randn((m, k), generator=generator(21)).cuda().bfloat16()
    w = (torch.randn((2 * padded, k), generator=generator(22)) * 0.05).cuda().bfloat16()
    a, s_a = fp8_pack.mxfp8_quant(x)
    w1, s_w1 = fp8_pack.mxfp8_quant(w)
    arguments = (a, s_a, w1, s_w1, gate_first, out_z, clamp_limit, activation, hidden)
    z, h, s_h = ops.mxfp8_mm_glu(*arguments)
    monkeypatch.setattr(mm_sm120, "FUSED_GLU", frozenset())
    z_chain, h_chain, s_h_chain = ops.mxfp8_mm_glu(*arguments)
    assert torch.equal(h.view(torch.uint8), h_chain.view(torch.uint8))
    assert torch.equal(s_h, s_h_chain)
    assert (z is None) == (not out_z)
    if out_z:
        assert torch.equal(z.view(torch.int16), z_chain.view(torch.int16))


def pack(recipe, x, weight):
    if recipe == "fp8_rowwise":
        return fp8_pack.fp8_rowwise_quant(x)
    quant = fp8_pack.fp8_b128_quant_weight if weight else fp8_pack.fp8_b128_quant
    return quant(x, e8m0=recipe == "fp8_b128_e8m0")


@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise"])
@pytest.mark.parametrize("activation", ["swiglu", "geglu_tanh"])
@pytest.mark.parametrize(("m", "hidden"), [(300, 256), (130, 640)])
@pytest.mark.parametrize("out_z", [False, True])
def test_fused_glu_is_bitwise_with_the_composed_chain(recipe, m, hidden, activation, out_z, monkeypatch):
    require_sm120(recipe, mm_sm120.WIDE, monkeypatch)
    declare_gpu_work(2)
    k = 512
    x = torch.randn((m, k), generator=generator(27)).cuda().bfloat16()
    w = (torch.randn((2 * hidden, k), generator=generator(28)) * 0.05).cuda().bfloat16()
    a, s_a = pack(recipe, x, False)
    w1, s_w1 = pack(recipe, w, True)
    operator = ops.fp8_rowwise_mm_glu if recipe == "fp8_rowwise" else ops.fp8_b128_mm_glu
    monkeypatch.setattr(mm_sm120, "FUSED_GLU", frozenset({(recipe, out_z)}))
    z, h, s_h = operator(a, s_a, w1, s_w1, activation, out_z)
    monkeypatch.setattr(mm_sm120, "FUSED_GLU", frozenset())
    z_chain, h_chain, s_h_chain = operator(a, s_a, w1, s_w1, activation, out_z)
    assert torch.equal(h.view(torch.uint8), h_chain.view(torch.uint8))
    assert torch.equal(s_h.view(torch.uint8), s_h_chain.view(torch.uint8))
    assert (z is None) == (not out_z)
    if out_z:
        assert torch.equal(z.view(torch.int16), z_chain.view(torch.int16))


@pytest.mark.parametrize(("m", "heads", "rope_pairs", "bf16_scales", "save"), [
    (300, 2, 32, False, True), (130, 4, 64, True, False), (257, 2, 48, False, False)])
def test_mxfp8_fused_qkv_prep_is_bitwise_with_the_composed_chain(m, heads, rope_pairs, bf16_scales, save,
                                                                 monkeypatch):
    require_sm120("mxfp8", mm_sm120.WIDE, monkeypatch)
    declare_gpu_work(2)
    dim, k = mm_sm120.QKV_HEAD_DIM, 512
    x = torch.randn((m, k), generator=generator(23)).cuda().bfloat16()
    w = (torch.randn((3 * heads * dim, k), generator=generator(24)) * 0.05).cuda().bfloat16()
    a, s_a = fp8_pack.mxfp8_quant(x)
    b, s_b = fp8_pack.mxfp8_quant(w)
    rope = torch.randn((1, m, 1, rope_pairs, 2), generator=generator(25)).cuda()
    qk_scales = torch.rand(heads, generator=generator(26)).add(0.5).cuda()
    qk_scales = qk_scales.bfloat16() if bf16_scales else qk_scales
    arguments = (a, s_a, b, s_b, qk_scales, rope, 0, 1e-6, heads, dim, save)
    monkeypatch.setattr(mm_sm120, "FUSED_QKV", True)
    fused = ops.mxfp8_mm_qkv_prep(*arguments)
    monkeypatch.setattr(mm_sm120, "FUSED_QKV", False)
    chain = ops.mxfp8_mm_qkv_prep(*arguments)
    for got, expected in zip(fused, chain, strict=True):
        if expected is None:
            assert got is None
        else:
            assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
