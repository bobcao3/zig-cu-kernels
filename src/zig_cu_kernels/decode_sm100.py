"""The SM100 decode kernels of `mm_abt` for at most `decode.MAX_M` rows of `A` and the recipes `mxfp8`, `fp8_b128_e8m0`
and `fp8_rowwise`. Each kernel computes `Yᵀ = B Aᵀ` with `tcgen05.mma.cta_group::1`, whose M operand is a tile of 128
weight rows and whose N operand is the token tile `T` of `A`, over the stream-K schedule of `decode.plan`. The
block-scaled recipes scale inside the MMA; the rowwise recipe scales the sums in the epilogue."""

import torch

from . import decode, fp8_mm, runtime
from .packs import ceil_div

ARCH = 100
SIGNATURE = "m" * 4 + "p" * 7 + "u" * 7
SCALE_ATOM_WORDS = 128
SCALE_ATOM_BLOCKS = 4
KERNEL_RECIPES = {"mxfp8": "mxfp8", "fp8_b128_e8m0": "fp8_b128", "fp8_rowwise": "fp8_rowwise"}
WHOLE_TILE_SHARE = 0.85
TILED_ABOVE_M = 64


def kernel_name(recipe, token_tile):
    return f"{KERNEL_RECIPES[recipe]}_decode_mm_abt_sm{ARCH}_t{token_tile}"


def scale_atoms(scales):
    return scales.view(torch.float32).view(-1, SCALE_ATOM_WORDS)


def grid_ctas(tiles, multiprocessors):
    """Returns the largest multiple of `tiles` up to `multiprocessors` when it keeps `WHOLE_TILE_SHARE` of them busy,
    so that each work group holds part of one tile only, and `multiprocessors` otherwise."""
    multiple = tiles * (multiprocessors // tiles)
    return multiple if multiple >= WHOLE_TILE_SHARE * multiprocessors else multiprocessors


def mm_abt_decode(problem: fp8_mm.Problem, ctas=None, token_tile=None):
    """Launches the decode kernel on `ctas` work groups (by default `grid_ctas` of the SM count), then the reduce
    kernel when the schedule shares tiles. Without `ctas`, more than `TILED_ABOVE_M` rows of `A` and a weight with
    at least one tile per busy SM go to the tiled kernel, which is faster there."""
    capability = runtime.compute_capability(problem.device_index)
    if ctas is None:
        multiprocessors = decode.ctas(problem.device_index)
        tiles = problem.padded_n // decode.TILE
        if problem.m > TILED_ABOVE_M and tiles >= WHOLE_TILE_SHARE * multiprocessors:
            return fp8_mm.lookup(capability, problem.recipe, problem.operation)(problem)
        ctas = grid_ctas(tiles, multiprocessors)
    schedule = decode.plan(capability, problem.recipe, problem.m, problem.padded_n, problem.k, ctas, token_tile)
    tokens = schedule.token_tile
    partials = decode.workspace(problem, schedule)
    a, b, y = problem.a, problem.b, problem.outputs["y"]
    value_maps = (problem.tensor_map("b_decode", b.values, (decode.TILE, decode.BLOCK)),
                  problem.tensor_map(f"a_decode_{tokens}", a.values, (tokens, decode.BLOCK)))
    weight_words = row_scales = column_scales = 0
    if problem.recipe == "fp8_rowwise":
        scale_maps = value_maps
        scale_blocks = schedule.blocks
        row_scales, column_scales = a.scales.data_ptr(), b.scales.data_ptr()
    else:
        a_atoms = problem.tensor_map("s_a_decode", scale_atoms(a.scales), (1, SCALE_ATOM_WORDS), swizzle=0)
        if problem.recipe == "mxfp8":
            b_atoms = problem.tensor_map("s_b_decode", scale_atoms(b.scales), (1, SCALE_ATOM_WORDS), swizzle=0)
            scale_maps = (b_atoms, a_atoms)
            scale_blocks = schedule.blocks
        else:
            scale_maps = (a_atoms, a_atoms)
            scale_blocks = ceil_div(problem.k, SCALE_ATOM_BLOCKS * decode.BLOCK)
            weight_words = b.scales.data_ptr()
    tensors = problem.epilogue.tensors
    bias, residual = tensors.get("bias"), tensors.get("residual")
    values = (*value_maps, *scale_maps, weight_words, row_scales, column_scales,
              0 if bias is None else bias.data_ptr(), 0 if residual is None else residual.data_ptr(), y.data_ptr(),
              partials.data_ptr(), problem.m, problem.n, schedule.blocks, schedule.tiles, scale_blocks, y.stride(0),
              0 if residual is None else residual.stride(0))
    decode.launch(problem, kernel_name(problem.recipe, tokens), schedule, SIGNATURE, values, partials, pdl=True)
    return schedule, partials


for recipe in KERNEL_RECIPES:
    decode.register(ARCH, recipe, mm_abt_decode)
