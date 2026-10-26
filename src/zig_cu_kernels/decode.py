"""The stream-K schedule of the decode GEMMs over units of one weight tile and one K block, K fastest. A CTA that holds
part of a tile writes an FP32 partial to its slot of the workspace, and the reduce kernel adds the partials of each
shared tile in CTA order before it runs the epilogue."""

import functools
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import runtime
from .packs import ceil_div

MAX_M = 128
TILE = 128
BLOCK = 128
SLOTS = 2
TOKEN_TILES = {
    89: tuple(range(8, MAX_M + 1, 8)),
    90: (8, 16, 32, 64, 128),
    100: (16, 32, 64, 128),
    120: tuple(range(8, MAX_M + 1, 8)),
}
EPILOGUES = ("store", "bias", "residual")
REDUCE_SIGNATURE = "ppppppuuuuuu"
REDUCE_ROWS = 4
PDL_ARCH = 90


def eligible(m, padded_n, k):
    return 0 < m <= MAX_M and k % BLOCK == 0 and padded_n % TILE == 0


def ctas(device_index):
    return runtime.num_sm(device_index)


@dataclass(frozen=True)
class Schedule:
    arch: int
    recipe: str
    m: int
    padded_n: int
    k: int
    tiles: int
    blocks: int
    units: int
    ctas: int
    token_tile: int

    def start(self, c):
        return c * self.units // self.ctas

    def range_of(self, c):
        return self.start(c), self.start(c + 1)

    def owner(self, unit):
        return ((unit + 1) * self.ctas - 1) // self.units

    def first_tile(self, c):
        return self.start(c) // self.blocks

    def last_tile(self, c):
        return (self.start(c + 1) - 1) // self.blocks

    def contributors(self, tile):
        first = self.owner(tile * self.blocks)
        return tuple(range(first, self.owner((tile + 1) * self.blocks - 1) + 1))

    def slot(self, c, tile):
        assert self.first_tile(c) <= tile <= self.last_tile(c), f"CTA {c} holds no unit of the tile {tile}"
        return SLOTS * c + (0 if tile == self.first_tile(c) else 1)

    @functools.cached_property
    def shared_tiles(self):
        shared = []
        for tile in sorted({self.first_tile(c) for c in range(self.ctas)} | {self.last_tile(c)
                                                                               for c in range(self.ctas)}):
            contributors = self.contributors(tile)
            if len(contributors) > 1:
                shared.append((tile, tuple(self.slot(c, tile) for c in contributors)))
        return tuple(shared)

    @property
    def workspace_shape(self):
        return self.ctas, SLOTS, self.token_tile, TILE

    @property
    def workspace_bytes(self):
        return self.ctas * SLOTS * self.token_tile * TILE * 4


def smallest_token_tile(arch, m):
    return next(tile for tile in TOKEN_TILES[arch] if tile >= m)


def plan(capability, recipe, m, padded_n, k, ctas, token_tile=None):
    assert eligible(m, padded_n, k), f"no decode schedule for M = {m}, N = {padded_n}, K = {k}"
    assert recipe in fp8_mm.RECIPES, f"recipe {recipe!r} is none of {fp8_mm.RECIPES}"
    arch = fp8_mm.arch_of(capability)
    assert arch in TOKEN_TILES, f"compute capability {capability} has no decode token tiles"
    tile = token_tile or smallest_token_tile(arch, m)
    assert m <= tile <= MAX_M and tile % 8 == 0, f"the token tile {tile} does not hold {m} rows"
    tiles, blocks = padded_n // TILE, k // BLOCK
    units = tiles * blocks
    assert units * min(ctas, units) < 2**32, "the reduce kernel computes unit · CTA products in 32 bits"
    return Schedule(arch, recipe, m, padded_n, k, tiles, blocks, units, max(1, min(ctas, units)), tile)


def workspace(problem, schedule):
    return problem.workspace(schedule.workspace_bytes).view(torch.float32).view(schedule.workspace_shape)


def reduce_name(recipe, epilogue):
    return f"{recipe}_decode_reduce_{epilogue}"


def reduce(problem, schedule, partials):
    kind = problem.epilogue.kind
    kernel, info = runtime.load_for(problem.device_index, reduce_name(problem.recipe, kind))
    tensors = problem.epilogue.tensors
    bias, residual = tensors.get("bias"), tensors.get("residual")
    rowwise = problem.recipe == "fp8_rowwise"
    values = (partials.data_ptr(), problem.outputs["y"].data_ptr(), 0 if bias is None else bias.data_ptr(),
              0 if residual is None else residual.data_ptr(), problem.a.scales.data_ptr() if rowwise else 0,
              problem.b.scales.data_ptr() if rowwise else 0, problem.m, problem.n, problem.padded_n, problem.k,
              schedule.ctas, schedule.token_tile)
    grid = (schedule.tiles, ceil_div(problem.m, REDUCE_ROWS), 1)
    runtime.run(kernel, grid, info.threads, info.shared_bytes, REDUCE_SIGNATURE, values, problem.stream,
                pdl=schedule.arch >= PDL_ARCH)


def launch(problem, kernel_name, schedule, signature, values, partials, pdl=False, cluster=None):
    """Launches the decode kernel `kernel_name` on `C` work groups and then the reduce kernel when tiles are shared.
    `values` holds the arguments of `signature`, among them the pointer of `partials`."""
    kernel, info = runtime.load_for(problem.device_index, kernel_name)
    runtime.run(kernel, (schedule.ctas, 1, 1), info.threads, info.shared_bytes, signature, values, problem.stream,
                cluster=cluster, pdl=pdl)
    if schedule.shared_tiles:
        reduce(problem, schedule, partials)


kernels: dict = {}


def register(arch: int, recipe: str, kernel: Callable, epilogues=EPILOGUES):
    assert arch in TOKEN_TILES, f"arch {arch} is none of {tuple(TOKEN_TILES)}"
    kernels[arch, recipe] = kernel, frozenset(epilogues)


def lookup(capability, problem) -> Optional[Callable]:
    if not kernels or not eligible(problem.m, problem.padded_n, problem.k):
        return None
    entry = kernels.get((fp8_mm.arch_of(capability), problem.recipe))
    return entry[0] if entry is not None and problem.epilogue.kind in entry[1] else None


from . import fp8_mm  # noqa: E402
