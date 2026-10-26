"""Checks of the SM100 decode kernels. Integer operands with power-of-two scales, whose sums FP32 holds exactly, give
the bits of the tiled kernel for every recipe and token count, with the stream-K split forced to few work groups so
that tiles have 1, 2, 3 or 5 contributors. When the reduce kernel is not built, a PyTorch reduce with the epilogue
order of `decode` adds the partials, which checks the partials that the decode kernel writes."""

import pytest
import torch

import fp8_mm_references as references
import pack_references
from conftest import declare_gpu_work
from zig_cu_kernels import decode, decode_sm100, fp8_mm, packs, runtime

CUDA = torch.cuda.is_available()
if CUDA:
    from zig_cu_kernels.torch import fp8_mm as ops

RECIPES = ("mxfp8", "fp8_b128_e8m0", "fp8_rowwise")
E8M0_ONE = 127
N, K = 384, 640
GAMMA = 2.0**-9
BF16_UNIT = 2.0**-8


def generator(seed):
    return torch.Generator().manual_seed(seed)


def integers(shape, seed):
    return torch.randint(-4, 5, shape, generator=generator(seed)).float().to(torch.float8_e4m3fn)


def exponents(shape, seed):
    return torch.randint(-2, 3, shape, generator=generator(seed))


def scales(shift, e8m0):
    return (shift + E8M0_ONE).to(torch.uint8) if e8m0 else torch.pow(2.0, shift.float())


def require(recipe, m):
    if not CUDA or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("no SM100 device")
    name = decode_sm100.kernel_name(recipe, decode.smallest_token_tile(decode_sm100.ARCH, m))
    if not runtime.is_available(name):
        pytest.skip(f"no kernel {name}")


def integer_operands(recipe, m, seed, n=N, k=K):
    a, b = integers((m, k), seed), integers((n, k), seed + 1)
    if recipe == "mxfp8":
        s_a = packs.scale_atoms(exponents((m, k // 32), seed + 2).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
        s_b = packs.scale_atoms(exponents((n, k // 32), seed + 3).add(E8M0_ONE).to(torch.uint8), E8M0_ONE)
    elif recipe == "fp8_rowwise":
        s_a, s_b = scales(exponents((m, 1), seed + 2), False), scales(exponents((n, 1), seed + 3), False)
    else:
        e8m0 = recipe.endswith("e8m0")
        s_a = packs.scale_atoms(scales(exponents((m, k // 128), seed + 2), e8m0), E8M0_ONE if e8m0 else 1.0)
        s_b = scales(exponents((n // 128, 4 * packs.ceil_div(k, 512)), seed + 3), e8m0)
    return [t.cuda() for t in (a, s_a, b, s_b)]


def normal_operands(recipe, m, seed):
    x = torch.randn((m, K), generator=generator(seed))
    w = torch.randn((N, K), generator=generator(seed + 1)) * 0.05
    if recipe == "mxfp8":
        packed = (*pack_references.mxfp8_quant(x), *pack_references.mxfp8_quant(w))
    elif recipe == "fp8_rowwise":
        packed = (*pack_references.fp8_rowwise_quant(x), *pack_references.fp8_rowwise_quant(w))
    else:
        e8m0 = recipe.endswith("e8m0")
        packed = (*pack_references.fp8_b128_quant(x, e8m0=e8m0),
                  *pack_references.fp8_b128_quant_weight(w, e8m0=e8m0))
    return [t.cuda() for t in packed]


def product(recipe, operands, bias=None):
    if recipe == "mxfp8":
        return ops.mxfp8_mm_abt(*operands, bias)
    if recipe == "fp8_rowwise":
        return ops.fp8_rowwise_mm_abt(*operands, bias)
    return ops.fp8_b128_mm_abt(*operands)


def torch_reduce(problem, schedule, partials):
    """The epilogue order of `decode`: FP32 adds in contributor order, row and column factors, residual, bias."""
    slots = partials.view(-1, schedule.token_tile, decode.TILE)[:, :problem.m]
    tensors, y = problem.epilogue.tensors, problem.outputs["y"]
    for tile, contributors in schedule.shared_tiles:
        columns = slice(tile * decode.TILE, min((tile + 1) * decode.TILE, problem.n))
        x = slots[contributors[0]].clone()
        for slot in contributors[1:]:
            x = x + slots[slot]
        x = x[:, :columns.stop - columns.start]
        if problem.recipe == "fp8_rowwise":
            x = x * problem.a.scales[:problem.m] * problem.b.scales[columns, 0][None, :]
        if tensors.get("residual") is not None:
            x = x + tensors["residual"][:problem.m, columns].float()
        if tensors.get("bias") is not None:
            x = x + tensors["bias"][columns].float()
        y[:, columns] = x.bfloat16()


def use_decode(recipe, ctas, monkeypatch):
    """Routes `mm_abt` of `recipe` to the decode kernel on `ctas` work groups."""
    if not runtime.is_available(decode.reduce_name(recipe, "store")):
        monkeypatch.setattr(decode, "reduce", torch_reduce)
    arch = fp8_mm.arch_of(runtime.compute_capability(torch.cuda.current_device()))
    operations = dict(fp8_mm.kernels[arch, recipe])
    operations["mm_abt"] = lambda problem: decode_sm100.mm_abt_decode(problem, ctas)
    monkeypatch.setitem(fp8_mm.kernels, (arch, recipe), operations)
    monkeypatch.setattr(decode, "kernels", {(decode_sm100.ARCH, recipe): (operations["mm_abt"], frozenset(
        decode.EPILOGUES))})


def use_tiled(monkeypatch):
    monkeypatch.setattr(decode, "kernels", {})


@pytest.mark.parametrize("ctas", [3, 7, 15])
@pytest.mark.parametrize("m", [1, 7, 16, 64, 128])
@pytest.mark.parametrize("recipe", RECIPES)
def test_decode_is_bitwise_with_the_tiled_kernel(recipe, m, ctas, monkeypatch):
    require(recipe, m)
    declare_gpu_work(3)
    operands = integer_operands(recipe, m, m + ctas)
    use_tiled(monkeypatch)
    tiled = product(recipe, operands)
    use_decode(recipe, ctas, monkeypatch)
    assert torch.equal(product(recipe, operands).view(torch.int16), tiled.view(torch.int16))


@pytest.mark.parametrize("m", [1, 16, 128])
@pytest.mark.parametrize("recipe", RECIPES)
def test_decode_on_every_multiprocessor_is_bitwise_with_the_tiled_kernel(recipe, m, monkeypatch):
    require(recipe, m)
    declare_gpu_work(3)
    operands = integer_operands(recipe, m, 30 + m, n=5376, k=8192)
    use_tiled(monkeypatch)
    tiled = product(recipe, operands)
    use_decode(recipe, None, monkeypatch)
    assert torch.equal(product(recipe, operands).view(torch.int16), tiled.view(torch.int16))


@pytest.mark.parametrize("ctas", [3, 15])
@pytest.mark.parametrize("recipe", ["mxfp8", "fp8_rowwise"])
def test_decode_bias_is_bitwise_with_the_tiled_kernel(recipe, ctas, monkeypatch):
    require(recipe, 7)
    declare_gpu_work(3)
    operands = integer_operands(recipe, 7, 40 + ctas)
    terms = torch.randint(-64, 65, (N,), generator=generator(41)).float().bfloat16().cuda()
    use_tiled(monkeypatch)
    tiled = product(recipe, operands, terms)
    use_decode(recipe, ctas, monkeypatch)
    assert torch.equal(product(recipe, operands, terms).view(torch.int16), tiled.view(torch.int16))


@pytest.mark.parametrize("ctas", [3, 15])
def test_decode_residual_is_bitwise_with_the_tiled_kernel(ctas, monkeypatch):
    require("mxfp8", 16)
    declare_gpu_work(3)
    operands = integer_operands("mxfp8", 16, 50 + ctas)
    residual = torch.randint(-64, 65, (16, N), generator=generator(51)).float().bfloat16().cuda()
    use_tiled(monkeypatch)
    tiled = ops.mxfp8_mm_abt_add(*operands, residual)
    use_decode("mxfp8", ctas, monkeypatch)
    assert torch.equal(ops.mxfp8_mm_abt_add(*operands, residual).view(torch.int16), tiled.view(torch.int16))


@pytest.mark.parametrize("recipe", RECIPES)
def test_decode_runs_give_the_same_bits_and_hold_the_error_bound(recipe, monkeypatch):
    require(recipe, 64)
    declare_gpu_work(2)
    operands = normal_operands(recipe, 64, 60)
    use_decode(recipe, 15, monkeypatch)
    first, second = product(recipe, operands), product(recipe, operands)
    assert torch.equal(first.view(torch.int16), second.view(torch.int16))
    kind = "fp8_b128" if recipe.startswith("fp8_b128") else recipe
    a, s_a, b, s_b = operands
    exact, magnitude = references.product(references.dequantized(a, s_a, kind), references.dequantized(b, s_b, kind))
    bound = GAMMA * magnitude + BF16_UNIT * exact.abs() + 1e-30
    assert ((first.cpu().double() - exact).abs() <= bound).all()


@pytest.mark.parametrize("tiles, multiprocessors, ctas", [(42, 148, 126), (128, 148, 128), (336, 148, 148),
                                                          (5, 7, 7), (5, 23, 20), (3, 3, 3)])
def test_the_grid_keeps_tiles_whole_when_few_multiprocessors_idle(tiles, multiprocessors, ctas):
    assert decode_sm100.grid_ctas(tiles, multiprocessors) == ctas
