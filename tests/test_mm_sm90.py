"""Bitwise checks of the SM90 `mm_abt` kernels with integer operands and power-of-two scales, whose sums FP32 holds
exactly, on shapes with partial tiles and an odd number of row tiles."""

from types import SimpleNamespace

import pytest
import torch

import fp8_mm_references as references
from conftest import declare_gpu_work
from zig_cu_kernels import fp8_mm as launcher
from zig_cu_kernels import fp8_pack, mm_sm90, packs

CUDA = torch.cuda.is_available()
if CUDA:
    from zig_cu_kernels.torch import fp8_mm as ops

E8M0_ONE = 127


def generator(seed):
    return torch.Generator().manual_seed(seed)


def integers(shape, seed):
    return torch.randint(-4, 5, shape, generator=generator(seed)).float().to(torch.float8_e4m3fn)


def exponents(shape, seed):
    return torch.randint(-2, 3, shape, generator=generator(seed))


def require_sm90(recipe):
    if not CUDA or torch.cuda.get_device_capability()[0] != 9 or not launcher.available(recipe, "mm_abt"):
        pytest.skip(f"no SM90 {recipe} mm_abt kernel on this device")


def scales(shift, e8m0):
    return (shift + E8M0_ONE).to(torch.uint8) if e8m0 else torch.pow(2.0, shift.float())


@pytest.mark.parametrize("n", [384, 640])
@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32"])
def test_fp8_b128_is_bitwise_with_integer_operands(recipe, n):
    require_sm90(recipe)
    declare_gpu_work(1)
    m, k = 390, 640
    e8m0 = recipe.endswith("e8m0")
    a, b = integers((m, k), 1), integers((n, k), 2)
    s_a = packs.scale_atoms(scales(exponents((m, k // 128), 3), e8m0), E8M0_ONE if e8m0 else 1.0)
    s_b = scales(exponents((n // 128, 4 * packs.ceil_div(k, 512)), 4), e8m0)
    exact = references.product(references.dequantized(a, s_a, "fp8_b128"), references.dequantized(b, s_b, "fp8_b128"))
    y = ops.fp8_b128_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda())
    assert torch.equal(y.cpu(), exact[0].float().bfloat16())


@pytest.mark.parametrize(("m", "n", "k", "sms", "offset"),
                         [(4200, 640, 640, None, 0), (390, 384, 384, None, 1), (1000, 640, 1536, 4, 0)])
@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32"])
def test_fp8_b128_staged_scales_are_bitwise_with_integer_operands(recipe, m, n, k, sms, offset, monkeypatch):
    """Covers a tile below the last row, K below one scale atom, the last column tile, tiles that wrap the stage ring
    and activation scales at an address that is not a multiple of 16 bytes."""
    require_sm90(recipe)
    declare_gpu_work(1)
    if sms is not None:
        monkeypatch.setattr(mm_sm90.runtime, "num_sm", lambda device: sms)
    e8m0 = recipe.endswith("e8m0")
    a, b = integers((m, k), 21), integers((n, k), 22)
    s_a = packs.scale_atoms(scales(exponents((m, k // 128), 23), e8m0), E8M0_ONE if e8m0 else 1.0)
    s_b = scales(exponents((n // 128, 4 * packs.ceil_div(k, 512)), 24), e8m0)
    exact = references.product(references.dequantized(a, s_a, "fp8_b128"), references.dequantized(b, s_b, "fp8_b128"))
    storage = torch.empty(s_a.numel() + offset, dtype=s_a.dtype, device="cuda")
    shifted = storage[offset:].view(s_a.shape).copy_(s_a)
    y = ops.fp8_b128_mm_abt(a.cuda(), shifted, b.cuda(), s_b.cuda())
    assert torch.equal(y.cpu(), exact[0].float().bfloat16())


@pytest.mark.parametrize(("m", "n", "k", "sms"), [(1000, 640, 1536, 10), (1000, 640, 1408, 6), (4200, 640, 1024, 22)])
@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32"])
def test_fp8_b128_split_tail_is_bitwise_with_integer_operands(recipe, m, n, k, sms, monkeypatch):
    """Covers K ranges that start inside a scale atom, a split tile below the last row and a second product that reuses
    the launcher's workspace."""
    tiles = packs.ceil_div(m, 2 * mm_sm90.TILE_M) * packs.ceil_div(n, 192)
    assert mm_sm90.split_plan(tiles, sms // 2, k // 128)[1] > 1
    require_sm90(recipe)
    declare_gpu_work(2)
    monkeypatch.setattr(mm_sm90.runtime, "num_sm", lambda device: sms)
    monkeypatch.setattr(mm_sm90, "SPLIT_TAIL", "_split")
    monkeypatch.setattr(mm_sm90, "MIN_SPLIT_GAIN", 0.0)
    e8m0 = recipe.endswith("e8m0")
    a, b = integers((m, k), 31), integers((n, k), 32)
    s_a = packs.scale_atoms(scales(exponents((m, k // 128), 33), e8m0), E8M0_ONE if e8m0 else 1.0)
    s_b = scales(exponents((n // 128, 4 * packs.ceil_div(k, 512)), 34), e8m0)
    exact = references.product(references.dequantized(a, s_a, "fp8_b128"), references.dequantized(b, s_b, "fp8_b128"))
    for _ in range(2):
        y = ops.fp8_b128_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda())
        assert torch.equal(y.cpu(), exact[0].float().bfloat16())


@pytest.mark.parametrize(("tiles", "k_blocks", "expected"), [(352, 96, (330, 3)), (1792, 168, None), (1376, 64, None),
                                                            (1320, 64, None), (60, 96, None)])
def test_the_tail_is_split_where_the_idle_sms_cost_enough(tiles, k_blocks, expected):
    """The shapes are the 66 two-SM slots of an H100 with Qwen-Image down, Gemma 4 down, an 8192 cube and a product with
    whole rounds only."""
    assert mm_sm90.tail_split(tiles, 66, k_blocks) == expected


def launched_kernels(monkeypatch, three_consumers):
    """Lets every rowwise product without a bias that fits the tile rule run the three-consumer kernel when
    `three_consumers` is set, and records the names of the launched kernels."""
    names = []
    launch = mm_sm90.launch
    monkeypatch.setattr(mm_sm90, "THREE_CONSUMERS_MIN_K", 0 if three_consumers else 1 << 62)
    monkeypatch.setattr(mm_sm90, "launch", lambda problem, name: (names.append(name), launch(problem, name)))
    return names


@pytest.mark.parametrize(("bias", "three_consumers"), [(False, False), (True, False), (False, True)])
def test_fp8_rowwise_is_bitwise_with_integer_operands(bias, three_consumers, monkeypatch):
    require_sm90("fp8_rowwise")
    declare_gpu_work(1)
    names = launched_kernels(monkeypatch, three_consumers)
    m, n, k = 390, 264, 528
    a, b = integers((m, k), 5), integers((n, k), 6)
    s_a, s_b = scales(exponents((m, 1), 7), False), scales(exponents((n, 1), 8), False)
    terms = torch.randint(-64, 65, (n,), generator=generator(9)).float().bfloat16() if bias else None
    acc = (a.double() @ b.double().T) * s_a.double() * s_b.double().T
    y = ops.fp8_rowwise_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda(), None if terms is None else terms.cuda())
    assert torch.equal(y.cpu(), references.bf16_store(acc.float(), terms))
    assert names[-1].endswith(mm_sm90.THREE_CONSUMERS) == three_consumers


def test_fp8_rowwise_fast_accumulation_is_bitwise_with_integer_operands():
    require_sm90("fp8_rowwise")
    declare_gpu_work(1)
    m, n, k = 390, 264, 528
    a, b = integers((m, k), 10), integers((n, k), 11)
    s_a, s_b = scales(exponents((m, 1), 12), False), scales(exponents((n, 1), 13), False)
    acc = (a.double() @ b.double().T) * s_a.double() * s_b.double().T
    y = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    problem = launcher.Problem("mm_abt", "fp8_rowwise", launcher.Pack(a.cuda(), s_a.cuda()),
                               launcher.Pack(b.cuda(), s_b.cuda()), m, n, n, k, launcher.Epilogue("store"), {"y": y},
                               torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
    mm_sm90.fp8_rowwise_mm_abt_fast(problem)
    assert torch.equal(y.cpu(), acc.float().bfloat16())


@pytest.mark.parametrize("k", [640, 768])
@pytest.mark.parametrize(("recipe", "three_consumers"),
                         [("fp8_b128_e8m0", False), ("fp8_b128_fp32", False), ("fp8_rowwise", False),
                          ("fp8_rowwise", True)])
def test_each_cluster_computes_several_tiles_in_turn(recipe, three_consumers, k, monkeypatch):
    require_sm90(recipe)
    declare_gpu_work(1)
    monkeypatch.setattr(mm_sm90.runtime, "num_sm", lambda device: 2 * mm_sm90.CLUSTER)
    names = launched_kernels(monkeypatch, three_consumers)
    m, n = 1000, 640
    a, b = integers((m, k), 14), integers((n, k), 15)
    if recipe == "fp8_rowwise":
        s_a, s_b = scales(exponents((m, 1), 16), False), scales(exponents((n, 1), 17), False)
        exact = ((a.double() @ b.double().T) * s_a.double() * s_b.double().T).float().bfloat16()
        y = ops.fp8_rowwise_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda(), None)
    else:
        e8m0 = recipe.endswith("e8m0")
        s_a = packs.scale_atoms(scales(exponents((m, k // 128), 16), e8m0), E8M0_ONE if e8m0 else 1.0)
        s_b = scales(exponents((n // 128, 4 * packs.ceil_div(k, 512)), 17), e8m0)
        exact = references.product(references.dequantized(a, s_a, "fp8_b128"),
                                   references.dequantized(b, s_b, "fp8_b128"))[0].float().bfloat16()
        y = ops.fp8_b128_mm_abt(a.cuda(), s_a.cuda(), b.cuda(), s_b.cuda())
    assert torch.equal(y.cpu(), exact)
    assert names[-1].endswith(mm_sm90.THREE_CONSUMERS) == three_consumers


@pytest.mark.parametrize(("m", "n", "k", "three_consumers"),
                         [(4096, 4096, 12288, True), (16384, 5376, 21504, True), (8192, 8192, 16384, False),
                          (4096, 16384, 5376, False), (4096, 4096, 4096, True), (4096, 4096, 2048, False)])
def test_three_consumers_run_at_large_k_where_each_cluster_walks_no_more_tiles(m, n, k, three_consumers,
                                                                               monkeypatch):
    try:
        mm_sm90.runtime.info(mm_sm90.kernel_name("fp8_rowwise", "store") + mm_sm90.THREE_CONSUMERS)
    except RuntimeError:
        pytest.skip("the SM90 kernels are not built")
    names = []
    monkeypatch.setattr(mm_sm90, "THREE_CONSUMERS_MIN_K", 4096)
    monkeypatch.setattr(mm_sm90.runtime, "num_sm", lambda device: 132)
    monkeypatch.setattr(mm_sm90, "launch", lambda problem, name: names.append(name))
    problem = SimpleNamespace(recipe="fp8_rowwise", epilogue=launcher.Epilogue("store"), m=m, padded_n=n, k=k,
                              device_index=0)
    mm_sm90.mm_abt(problem)
    assert names == [mm_sm90.kernel_name("fp8_rowwise", "store") + (mm_sm90.THREE_CONSUMERS if three_consumers else "")]


def test_rowwise_products_run_two_consumers_by_default(monkeypatch):
    names = []
    monkeypatch.setattr(mm_sm90, "launch", lambda problem, name: names.append(name))
    problem = SimpleNamespace(recipe="fp8_rowwise", epilogue=launcher.Epilogue("store"), m=4096, padded_n=4096,
                              k=12288, device_index=0)
    mm_sm90.mm_abt(problem)
    assert names == [mm_sm90.kernel_name("fp8_rowwise", "store")]


def pack(recipe, x, weight):
    if recipe == "fp8_rowwise":
        return fp8_pack.fp8_rowwise_quant(x)
    quant = fp8_pack.fp8_b128_quant_weight if weight else fp8_pack.fp8_b128_quant
    return quant(x, e8m0=recipe == "fp8_b128_e8m0")


@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise"])
@pytest.mark.parametrize("activation", ["swiglu", "geglu_tanh"])
@pytest.mark.parametrize(("m", "hidden", "k", "sms"), [(300, 256, 512, 132), (130, 640, 640, 132), (1000, 384, 768, 4)])
@pytest.mark.parametrize("out_z", [False, True])
def test_fused_glu_is_bitwise_with_the_composed_chain(recipe, activation, m, hidden, k, sms, out_z, monkeypatch):
    require_sm90(recipe)
    declare_gpu_work(2)
    monkeypatch.setattr(mm_sm90.runtime, "num_sm", lambda device: sms)
    x = torch.randn((m, k), generator=generator(27)).cuda().bfloat16()
    w = (torch.randn((2 * hidden, k), generator=generator(28)) * 0.05).cuda().bfloat16()
    a, s_a = pack(recipe, x, False)
    w1, s_w1 = pack(recipe, w, True)
    operator = ops.fp8_rowwise_mm_glu if recipe == "fp8_rowwise" else ops.fp8_b128_mm_glu
    monkeypatch.setattr(mm_sm90, "FUSED_GLU", frozenset({(recipe, out_z)}))
    z, h, s_h = operator(a, s_a, w1, s_w1, activation, out_z)
    monkeypatch.setattr(mm_sm90, "FUSED_GLU", frozenset())
    z_chain, h_chain, s_h_chain = operator(a, s_a, w1, s_w1, activation, out_z)
    assert torch.equal(h.view(torch.uint8), h_chain.view(torch.uint8))
    assert torch.equal(s_h.view(torch.uint8), s_h_chain.view(torch.uint8))
    assert (z is None) == (not out_z)
    if out_z:
        assert torch.equal(z.view(torch.int16), z_chain.view(torch.int16))
