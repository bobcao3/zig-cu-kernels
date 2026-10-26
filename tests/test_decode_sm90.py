"""Checks of the SM90 decode kernels. With integer operands and power-of-two scales, whose sums FP32 holds exactly, each
kernel gives the bits of the tiled SM90 kernel for schedules whose CTAs hold whole tiles and for schedules whose tiles
have 2 to 5 contributors. Two runs give the same bits, and random operands stay within the error bound."""

import pytest
import torch

import fp8_mm_references as references
import pack_references
from conftest import declare_gpu_work
from zig_cu_kernels import fp8_mm as launcher
from zig_cu_kernels import decode, decode_sm90, mm_sm90, packs

CUDA = torch.cuda.is_available()
E8M0_ONE = 127
GAMMA = 2.0**-9
BF16_UNIT = 2.0**-8
ROWWISE_FROBENIUS = 2.0**-8
CASES = [("fp8_b128_e8m0", "store"), ("fp8_b128_fp32", "store"), ("fp8_rowwise", "store"), ("fp8_rowwise", "bias")]
TOKENS = [1, 7, 16, 20, 64, 128]
# Each entry is (n, k, ctas, contributors), where `contributors` lists the contributor counts of the shared tiles.
SPLITS = [(640, 1024, 5, []), (640, 1024, 7, [2, 3]), (640, 1024, 13, [3, 4]), (640, 1024, 21, [4, 5]),
          (384, 640, 15, [5])]


def require_sm90():
    if not CUDA or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("no SM90 device")


def generator(seed):
    return torch.Generator().manual_seed(seed)


def integers(shape, seed):
    return torch.randint(-4, 5, shape, generator=generator(seed)).float().to(torch.float8_e4m3fn)


def shifts(shape, seed):
    return torch.randint(-2, 3, shape, generator=generator(seed))


def integer_packs(recipe, m, n, k, seed):
    a, b = integers((m, k), seed), integers((n, k), seed + 1)
    if recipe == "fp8_rowwise":
        s_a, s_b = torch.pow(2.0, shifts((m, 1), seed + 2).float()), torch.pow(2.0, shifts((n, 1), seed + 3).float())
        return a, s_a, b, s_b
    e8m0 = recipe.endswith("e8m0")

    def encoded(shift):
        return (shift + E8M0_ONE).to(torch.uint8) if e8m0 else torch.pow(2.0, shift.float())

    s_a = packs.scale_atoms(encoded(shifts((m, k // 128), seed + 2)), E8M0_ONE if e8m0 else 1.0)
    s_b = encoded(shifts((n // 128, 4 * packs.ceil_div(k, 512)), seed + 3))
    return a, s_a, b, s_b


def problem_of(recipe, kind, operands, m, n, k, bias=None):
    a, s_a, b, s_b = (t.cuda() for t in operands)
    tensors = {} if bias is None else {"bias": bias.cuda()}
    y = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    return launcher.Problem("mm_abt", recipe, launcher.Pack(a, s_a), launcher.Pack(b, s_b), m, n, n, k,
                            launcher.Epilogue(kind, tensors), {"y": y}, torch.cuda.current_device(),
                            torch.cuda.current_stream().cuda_stream)


def decoded(recipe, kind, operands, m, n, k, ctas, monkeypatch, bias=None):
    monkeypatch.setattr(decode, "ctas", lambda device: ctas)
    problem = problem_of(recipe, kind, operands, m, n, k, bias)
    decode_sm90.mm_abt_decode(problem, ctas)
    return problem.outputs["y"]


def test_the_splits_give_the_listed_contributor_counts():
    for n, k, ctas, contributors in SPLITS:
        schedule = decode.plan(90, "fp8_rowwise", 16, n, k, ctas)
        assert sorted({len(slots) for _, slots in schedule.shared_tiles}) == contributors


def test_the_partial_count_is_the_slot_count_of_the_shared_tiles():
    for tiles in range(1, 7):
        for blocks in range(1, 10):
            for ctas in range(1, tiles * blocks + 1):
                schedule = decode.plan(90, "fp8_rowwise", 8, tiles * decode.TILE, blocks * decode.BLOCK, ctas)
                slots = sum(len(slots) for _, slots in schedule.shared_tiles)
                assert decode_sm90.partial_count(tiles, blocks, ctas) == slots


@pytest.mark.parametrize("token_tile", [8, 16, 64, 128])
def test_the_cta_count_keeps_gemma_tiles_whole_or_split_evenly(token_tile):
    assert decode_sm90.choose_ctas(16384 // 128, 5376 // 128, token_tile, 132) == 128
    assert decode_sm90.choose_ctas(5376 // 128, 8192 // 128, token_tile, 132) == 126
    assert decode_sm90.choose_ctas(5376 // 128, 21504 // 128, token_tile, 132) == 126
    assert decode_sm90.partial_count(5376 // 128, 8192 // 128, 126) == 126


@pytest.mark.parametrize(("n", "k", "ctas", "contributors"), SPLITS)
@pytest.mark.parametrize("m", TOKENS)
@pytest.mark.parametrize(("recipe", "kind"), CASES)
def test_decode_is_bitwise_with_the_tiled_kernel(recipe, kind, m, n, k, ctas, contributors, monkeypatch):
    require_sm90()
    declare_gpu_work(3)
    operands = integer_packs(recipe, m, n, k, m + n + ctas)
    bias = torch.randint(-64, 65, (n,), generator=generator(n)).float().bfloat16() if kind == "bias" else None
    y = decoded(recipe, kind, operands, m, n, k, ctas, monkeypatch, bias)
    tiled = problem_of(recipe, kind, operands, m, n, k, bias)
    mm_sm90.mm_abt(tiled)
    assert torch.equal(y.view(torch.int16), tiled.outputs["y"].view(torch.int16))
    a, s_a, b, s_b = operands
    kind_of_pack = "fp8_rowwise" if recipe == "fp8_rowwise" else "fp8_b128"
    exact = references.product(references.dequantized(a, s_a, kind_of_pack),
                               references.dequantized(b, s_b, kind_of_pack))[0]
    assert torch.equal(y.cpu(), references.bf16_store(exact.float(), bias))


def random_packs(recipe, m, n, k, seed):
    x = torch.randn((m, k), generator=generator(seed))
    w = torch.randn((n, k), generator=generator(seed + 1)) * 0.05
    if recipe == "fp8_rowwise":
        return (*pack_references.fp8_rowwise_quant(x), *pack_references.fp8_rowwise_quant(w))
    e8m0 = recipe.endswith("e8m0")
    return (*pack_references.fp8_b128_quant(x, e8m0=e8m0), *pack_references.fp8_b128_quant_weight(w, e8m0=e8m0))


@pytest.mark.parametrize("ctas", [13, 132])
@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise"])
def test_two_runs_give_the_same_bits(recipe, ctas, monkeypatch):
    require_sm90()
    declare_gpu_work(2)
    m, n, k = 16, 640, 1024
    operands = random_packs(recipe, m, n, k, 30)
    first = decoded(recipe, "store", operands, m, n, k, ctas, monkeypatch)
    second = decoded(recipe, "store", operands, m, n, k, ctas, monkeypatch)
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))


@pytest.mark.parametrize("m", [1, 16, 128])
@pytest.mark.parametrize("recipe", ["fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise"])
def test_decode_within_the_error_bound(recipe, m, monkeypatch):
    require_sm90()
    declare_gpu_work(1)
    n, k = 640, 2048
    operands = random_packs(recipe, m, n, k, 40 + m)
    y = decoded(recipe, "store", operands, m, n, k, 13, monkeypatch)
    a, s_a, b, s_b = operands
    kind = "fp8_rowwise" if recipe == "fp8_rowwise" else "fp8_b128"
    exact, magnitude = references.product(references.dequantized(a, s_a, kind), references.dequantized(b, s_b, kind))
    if recipe == "fp8_rowwise":
        assert torch.linalg.norm(y.cpu().double() - exact) <= ROWWISE_FROBENIUS * torch.linalg.norm(exact)
        return
    bound = GAMMA * magnitude + BF16_UNIT * exact.abs() + 1e-30
    assert ((y.cpu().double() - exact).abs() <= bound).all()
