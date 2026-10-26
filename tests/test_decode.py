"""Checks of the decode schedule against a brute-force assignment of units, bitwise checks of the reduce kernels against
the same operations in PyTorch, and bitwise checks of each registered decode kernel against the tiled kernel with
integer operands and power-of-two scales, whose sums FP32 holds exactly."""

import pytest
import torch

from conftest import declare_gpu_work
from zig_cu_kernels import decode, fp8_mm, packs, runtime

CUDA = torch.cuda.is_available()
DECODE_M = (1, 7, 16, 64, 128)
REDUCE_CASES = [(512, 1024, 7), (384, 640, 11), (256, 1280, 9), (1024, 512, 5), (640, 1280, 23)]
E8M0_ONE = 127
SENTINEL = 0x7FC1


def brute_force(tiles, blocks, ctas):
    """Returns the CTA of each unit and, for each tile, its contributors and their slots, from the ranges alone."""
    units = tiles * blocks
    owners = [c for c in range(ctas) for _ in range(c * units // ctas, (c + 1) * units // ctas)]
    ranges = [range(c * units // ctas, (c + 1) * units // ctas) for c in range(ctas)]
    tile_owners = {}
    for unit, c in enumerate(owners):
        tile_owners.setdefault(unit // blocks, [])
        if c not in tile_owners[unit // blocks]:
            tile_owners[unit // blocks].append(c)
    slots = {}
    for tile, contributors in tile_owners.items():
        for c in contributors:
            first_tile = ranges[c][0] // blocks
            slots[c, tile] = 2 * c + (0 if tile == first_tile else 1)
    return owners, tile_owners, slots, ranges


def schedule_of(tiles, blocks, ctas, m=1):
    return decode.plan(89, "fp8_rowwise", m, tiles * decode.TILE, blocks * decode.BLOCK, ctas)


@pytest.mark.parametrize("tiles", range(1, 7))
@pytest.mark.parametrize("blocks", range(1, 10))
def test_schedule_matches_brute_force(tiles, blocks):
    units = tiles * blocks
    for ctas in range(1, units + 4):
        schedule = schedule_of(tiles, blocks, ctas)
        effective = min(ctas, units)
        assert schedule.ctas == effective
        owners, tile_owners, slots, ranges = brute_force(tiles, blocks, effective)
        assert len(owners) == units and all(len(r) > 0 for r in ranges)
        assert [schedule.owner(u) for u in range(units)] == owners
        assert [schedule.range_of(c) for c in range(effective)] == [(r.start, r.stop) for r in ranges]
        for tile in range(tiles):
            assert schedule.contributors(tile) == tuple(tile_owners[tile])
        expected_shared = tuple((tile, tuple(slots[c, tile] for c in tile_owners[tile]))
                                for tile in range(tiles) if len(tile_owners[tile]) > 1)
        assert schedule.shared_tiles == expected_shared
        used = [slots[c, tile] for tile, contributors in tile_owners.items() if len(contributors) > 1
                for c in contributors]
        assert len(used) == len(set(used)), "two partials share a slot"
        for (c, tile), slot in slots.items():
            assert schedule.slot(c, tile) == slot and slot // 2 == c
            if len(tile_owners[tile]) > 1:
                assert tile in (schedule.first_tile(c), schedule.last_tile(c)), "a middle tile is shared"


@pytest.mark.parametrize(("n", "k"), [(16384, 5376), (5376, 8192), (43008, 5376), (5376, 21504)])
@pytest.mark.parametrize("ctas", [132, 142, 148, 188])
def test_gemma_schedules_cover_every_unit_once(n, k, ctas):
    schedule = schedule_of(n // decode.TILE, k // decode.BLOCK, ctas)
    assert schedule.units == n * k // (decode.TILE * decode.BLOCK)
    sizes = [hi - lo for lo, hi in map(schedule.range_of, range(schedule.ctas))]
    assert sum(sizes) == schedule.units and max(sizes) - min(sizes) <= 1
    assert all(len(slots) >= 2 for _, slots in schedule.shared_tiles)
    assert schedule.workspace_shape == (ctas, 2, 8, decode.TILE)


def test_reduce_cases_have_two_to_five_contributors():
    counts = {len(slots) for n, k, ctas in REDUCE_CASES
              for _, slots in schedule_of(n // decode.TILE, k // decode.BLOCK, ctas).shared_tiles}
    assert {2, 3, 4, 5} <= counts


@pytest.mark.parametrize(("arch", "m", "tile"), [(89, 1, 8), (89, 9, 16), (120, 128, 128), (90, 9, 16), (90, 33, 64),
                                                 (100, 1, 16), (100, 65, 128)])
def test_token_tile_is_the_smallest_that_holds_m(arch, m, tile):
    assert decode.smallest_token_tile(arch, m) == tile


@pytest.mark.parametrize(("m", "n", "k", "expected"), [(1, 128, 128, True), (128, 5376, 8192, True),
                                                       (129, 5376, 8192, False), (0, 128, 128, False),
                                                       (16, 136, 128, False), (16, 128, 160, False)])
def test_eligible(m, n, k, expected):
    assert decode.eligible(m, n, k) == expected


def device_arch():
    return fp8_mm.arch_of(runtime.compute_capability(torch.cuda.current_device()))


def problem_of(recipe, a, b, m, n, padded_n, k, epilogue, y):
    device_index = torch.cuda.current_device()
    return fp8_mm.Problem("mm_abt", recipe, fp8_mm.Pack(*a), fp8_mm.Pack(*b), m, n, padded_n, k, epilogue, {"y": y},
                          device_index, torch.cuda.current_stream(device_index).cuda_stream)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA device")
@pytest.mark.parametrize("m", [16, 128, 129])
def test_mm_abt_routes_eligible_problems_to_the_decode_kernel(m, monkeypatch):
    arch, calls = device_arch(), []
    monkeypatch.setitem(fp8_mm.kernels, (arch, "fp8_rowwise"), {"mm_abt": lambda p: calls.append("tiled")})
    monkeypatch.setitem(decode.kernels, (arch, "fp8_rowwise"),
                        (lambda p: calls.append("decode"), frozenset(("store", "bias"))))
    empty = torch.empty(0)
    for kind, n, k in (("store", 256, 512), ("residual", 256, 512), ("store", 256, 528), ("bias", 136, 512)):
        problem = problem_of("fp8_rowwise", (empty, empty), (empty, empty), m, n, n, k, fp8_mm.Epilogue(kind), empty)
        fp8_mm.launch(problem)
    assert calls == (["decode", "tiled", "tiled", "tiled"] if m <= decode.MAX_M else ["tiled"] * 4)


def require_reduce(recipe, epilogue):
    if not CUDA:
        pytest.skip("needs a CUDA device")
    reason = runtime.unavailable_reason(decode.reduce_name(recipe, epilogue))
    if reason is not None:
        pytest.skip(reason)


def reduce_reference(partials, schedule, m, n, recipe, kind, s_a, s_w, bias, residual, y):
    """Adds the partials of each shared tile in CTA order and runs the epilogue, one PyTorch operation at a time."""
    expected = y.clone()
    slots = partials.view(-1, schedule.token_tile, decode.TILE)
    for tile, tile_slots in schedule.shared_tiles:
        columns = slice(tile * decode.TILE, (tile + 1) * decode.TILE)
        total = slots[tile_slots[0], :m]
        for slot in tile_slots[1:]:
            total = total + slots[slot, :m]
        if recipe == "fp8_rowwise":
            total = total * s_a[:m] * s_w[columns, 0][None, :]
        if kind == "residual":
            total = total + residual[:m, columns].float()
        if kind == "bias":
            total = total + bias[columns].float()
        width = min(decode.TILE, n - tile * decode.TILE)
        if width > 0:
            expected[:m, tile * decode.TILE:tile * decode.TILE + width] = total[:, :width].bfloat16()
    return expected


@pytest.mark.parametrize("case", REDUCE_CASES)
@pytest.mark.parametrize("m", DECODE_M)
@pytest.mark.parametrize("kind", decode.EPILOGUES)
@pytest.mark.parametrize("recipe", fp8_mm.RECIPES)
def test_reduce_is_bitwise_with_torch(recipe, kind, m, case):
    require_reduce(recipe, kind)
    declare_gpu_work(2)
    padded_n, k, ctas = case
    n = padded_n if kind == "residual" else padded_n - 8
    schedule = decode.plan(runtime.compute_capability(torch.cuda.current_device()), recipe, m, padded_n, k, ctas)
    generator = torch.Generator(device="cuda").manual_seed(m * 1000 + padded_n + ctas)
    partials = torch.randn(schedule.workspace_shape, generator=generator, device="cuda")
    s_a = torch.rand((m, 1), generator=generator, device="cuda") + 0.5
    s_w = torch.rand((padded_n, 1), generator=generator, device="cuda") + 0.5
    bias = torch.randn(padded_n, generator=generator, device="cuda").bfloat16() if kind == "bias" else None
    residual = torch.randn((m, padded_n), generator=generator, device="cuda").bfloat16() if kind == "residual" \
        else None
    y = torch.full((m, n), SENTINEL, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    expected = reduce_reference(partials, schedule, m, n, recipe, kind, s_a, s_w, bias, residual, y)
    tensors = {name: tensor for name, tensor in (("bias", bias), ("residual", residual)) if tensor is not None}
    empty = torch.empty(0, device="cuda")
    problem = problem_of(recipe, (empty, s_a), (empty, s_w), m, n, padded_n, k, fp8_mm.Epilogue(kind, tensors), y)
    decode.reduce(problem, schedule, partials)
    assert torch.equal(y.view(torch.int16), expected.view(torch.int16))


def integers(shape, generator):
    return torch.randint(-4, 5, shape, generator=generator).float().to(torch.float8_e4m3fn)


def powers_of_two(shape, generator, e8m0):
    shift = torch.randint(-2, 3, shape, generator=generator)
    return (shift + E8M0_ONE).to(torch.uint8) if e8m0 else torch.pow(2.0, shift.float())


def integer_packs(recipe, m, n, k, generator):
    a, b = integers((m, k), generator), integers((n, k), generator)
    if recipe == "mxfp8":
        s_a = powers_of_two(packs.mxfp8_scale_shape(m, k), generator, True)
        s_b = powers_of_two(packs.mxfp8_scale_shape(n, k), generator, True)
    elif recipe == "fp8_rowwise":
        s_a, s_b = powers_of_two((m, 1), generator, False), powers_of_two((n, 1), generator, False)
    else:
        e8m0 = recipe.endswith("e8m0")
        s_a = packs.scale_atoms(powers_of_two((m, k // 128), generator, e8m0), E8M0_ONE if e8m0 else 1.0)
        s_b = powers_of_two((n // 128, 4 * packs.ceil_div(k, 512)), generator, e8m0)
    return (a.cuda(), s_a.cuda()), (b.cuda(), s_b.cuda())


@pytest.mark.parametrize("ctas", [7, 23])
@pytest.mark.parametrize("m", DECODE_M)
@pytest.mark.parametrize("kind", decode.EPILOGUES)
@pytest.mark.parametrize("recipe", fp8_mm.RECIPES)
def test_decode_kernel_is_bitwise_with_the_tiled_kernel(recipe, kind, m, ctas, monkeypatch):
    if not CUDA:
        pytest.skip("needs a CUDA device")
    n, k = 640, 1280
    capability = runtime.compute_capability(torch.cuda.current_device())
    probe = problem_of(recipe, (None, None), (None, None), m, n, n, k, fp8_mm.Epilogue(kind), None)
    kernel = decode.lookup(capability, probe)
    if kernel is None:
        pytest.skip(f"no decode kernel of {recipe} with the {kind!r} epilogue on this device")
    monkeypatch.setattr(decode, "ctas", lambda _: ctas)
    declare_gpu_work(3)
    generator = torch.Generator().manual_seed(m * 100 + ctas)
    a, b = integer_packs(recipe, m, n, k, generator)
    terms = torch.randint(-64, 65, (m, n), generator=generator).float().bfloat16().cuda()
    tensors = {"bias": terms[0]} if kind == "bias" else {"residual": terms} if kind == "residual" else {}
    outputs = [torch.empty((m, n), dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    tiled = fp8_mm.lookup(capability, recipe, "mm_abt")
    tiled(problem_of(recipe, a, b, m, n, n, k, fp8_mm.Epilogue(kind, tensors), outputs[0]))
    for y in outputs[1:]:
        kernel(problem_of(recipe, a, b, m, n, n, k, fp8_mm.Epilogue(kind, tensors), y))
    assert torch.equal(outputs[1], outputs[0])
    assert torch.equal(outputs[2].view(torch.int16), outputs[1].view(torch.int16))
