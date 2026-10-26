"""The SM120 `mma.sync` kernels of `mm_abt` for every recipe, with the store, bias and residual epilogues. One work group
computes one 128x256 tile of `Y`. Every recipe also has a 128x128 kernel; `TILES` maps each recipe to the tile of the
kernel that `mm_abt` launches.

One work group fits on a multiprocessor at a time, so a grid of `T` wide tiles runs in `ceil(T / SMs)` waves. When the
last wave is nearly empty, `mm_abt` computes the last `tail_bands` row bands of 128 rows with 128x128 tiles in a second
launch. A 128x128 tile takes about `SQUARE_TIME` of the time of a wide tile, and a second launch costs about
`LAUNCH_COST` waves. For the recipes in `SPLIT_RECIPES`, `mm_abt` instead splits the K blocks of the last row band into
`split_count` ranges when a schedule model predicts a shorter time: the work groups of the extra grid rows write FP32
partials, and the decode reduce kernel adds them in split order and runs the epilogue, so the result does not depend on
the timing of the work groups.

The `low_rank` kernels add `low · upᵀ` to the FP32 sums with BF16 MMAs before the epilogue. `mm_glu` runs one kernel
whose epilogue writes `Z` when asked and the pack of `h` for each pair `(recipe, out_z)` in `FUSED_GLU`. For the rowwise
recipe that kernel writes the BF16 `h`, and `fp8_rowwise_quant_bf16` then writes its pack, because a row scale needs the
whole row. With `Z`, that path moves as many bytes as the composed chain and needs a BF16 `h` buffer, so the rowwise
`mm_glu` with `Z` runs the composed chain of `glu_quant`, like every case outside `FUSED_GLU`. With `FUSED_QKV` set
and heads of `QKV_HEAD_DIM`, `mm_qkv_prep` runs one kernel whose epilogue normalizes and rotates whole heads; the other
cases run the composed chain of `qkv_prep`."""

import functools
import heapq
import math

import torch

from . import decode, fp8_mm, glu_quant, qkv_prep, runtime
from .packs import ceil_div

ARCH = 120
SQUARE = "128x128x128"
WIDE = "128x256x128"
TILES = {recipe: WIDE for recipe in fp8_mm.RECIPES}
SIGNATURE = "mmpppppuuuu"
LOW_RANK_SIGNATURE = SIGNATURE + "ppu"
GLU_SIGNATURE = "mmpppppuuuuuf"
GLU_BOX = 32
FUSED_GLU = frozenset((recipe, out_z) for recipe in fp8_mm.RECIPES for out_z in (False, True)) - {("fp8_rowwise", True)}
QKV_SIGNATURE = "mmpppppppuuuuuf"
QKV_HEAD_DIM = 128
FUSED_QKV = True
EPILOGUES = ("store", "bias", "residual", "low_rank")
BAND = 128
SQUARE_TIME = 0.56
LAUNCH_COST = 0.1
SPLIT_RECIPES = frozenset({"mxfp8", "fp8_rowwise"})
SPLIT_SIGNATURE = SIGNATURE + "pu"
SPLIT_EPILOGUES = decode.EPILOGUES
MAX_SPLITS = 8
GROUP_COST = 0.05
PARTIAL_COST = 0.03
REDUCE_COST = 0.24


def kernel_name(recipe, tile=None, low_rank=False):
    operation = "mm_abt_low_rank" if low_rank else "mm_abt"
    return f"{fp8_mm.kernel_name(operation, recipe, ARCH)}_{tile or TILES[recipe]}"


@functools.cache
def multiprocessors(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def tail_cost(m, n, sms, tail):
    bands, wide_columns, square_columns = ceil_div(m, BAND), ceil_div(n, 256), ceil_div(n, 128)
    square_waves = SQUARE_TIME * ceil_div(tail * square_columns, sms) + (LAUNCH_COST if tail else 0)
    return math.ceil((bands - tail) * wide_columns / sms) + square_waves


def tail_bands(m, n, sms):
    """The number of trailing row bands that minimizes the modeled time in waves of wide tiles."""
    bands, wide_columns = ceil_div(m, BAND), ceil_div(n, 256)
    if bands * wide_columns <= sms:
        return 0
    return min(range(bands), key=lambda tail: (tail_cost(m, n, sms, tail), tail))


def split_waves(m, n, k, sms, splits):
    """The modeled time in waves of wide tiles when the last row band runs in `splits` K ranges: the work groups start
    in grid order on the first free multiprocessor, a split work group takes its share of the K blocks of a wide tile
    plus `GROUP_COST` and `PARTIAL_COST`, and the reduce kernel adds `REDUCE_COST`."""
    bands, columns, blocks = ceil_div(m, BAND), ceil_div(n, 256), ceil_div(k, 128)
    durations = [1.0] * ((bands - 1) * columns)
    for split in range(splits):
        share = ((split + 1) * blocks // splits - split * blocks // splits) / blocks
        durations += [(1 - GROUP_COST) * share + GROUP_COST + PARTIAL_COST] * columns
    free = [0.0] * sms
    for duration in durations:
        heapq.heappush(free, heapq.heappop(free) + duration)
    return max(free) + REDUCE_COST


@functools.cache
def split_count(m, n, padded_n, k, sms):
    """The number of K ranges of the last row band, or 0 when the 128x128 tail bands model faster."""
    bands, columns, blocks = ceil_div(m, BAND), ceil_div(n, 256), ceil_div(k, 128)
    if bands * columns <= sms or k % 128 or padded_n % 128 or n % 8:
        return 0
    tiles = padded_n // 128
    best, count = tail_cost(m, n, sms, tail_bands(m, n, sms)), 0
    for splits in range(2, min(blocks, MAX_SPLITS) + 1):
        if tiles * blocks * tiles * splits >= 2**32:
            break
        waves = split_waves(m, n, k, sms, splits)
        if waves < best:
            best, count = waves, splits
    return count


def rows_from(tensor, row, rows_per_index=1):
    return None if tensor is None else tensor[row // rows_per_index:]


def launch(problem, tile, row, end):
    low_rank = problem.epilogue.kind == "low_rank"
    kernel, info = runtime.load_for(problem.device_index, kernel_name(problem.recipe, tile, low_rank))
    a, b = problem.a, problem.b
    scale_a = rows_from(a.scales, row, BAND if a.scales.dim() == 4 else 1)
    a_map = problem.tensor_map(f"a{row}", rows_from(a.values, row), (info.tile_m, info.tile_k))
    b_map = problem.tensor_map(f"b{tile}", b.values, (info.tile_n, info.tile_k))
    tensors = problem.epilogue.tensors
    bias, residual = tensors.get("bias"), rows_from(tensors.get("residual"), row)
    m = end - row
    values = (a_map, b_map, rows_from(problem.outputs["y"], row).data_ptr(), 0 if bias is None else bias.data_ptr(),
              0 if residual is None else residual.data_ptr(), scale_a.data_ptr(), b.scales.data_ptr(), m,
              problem.n, problem.padded_n, problem.k)
    if low_rank:
        low = tensors["low"]
        values += (rows_from(low, row).data_ptr(), tensors["up"].data_ptr(), low.shape[1])
    grid = (ceil_div(problem.n, info.tile_n), ceil_div(m, info.tile_m), 1)
    runtime.run(kernel, grid, info.threads, info.shared_bytes, LOW_RANK_SIGNATURE if low_rank else SIGNATURE, values,
                problem.stream)


def launch_split(problem, splits):
    """Launches the wide tiles of all but the last row band and `splits` work groups per wide tile of the last band,
    then the reduce kernel on the last band."""
    recipe, a, b = problem.recipe, problem.a, problem.b
    kernel, info = runtime.load_for(problem.device_index, f"{fp8_mm.kernel_name('mm_abt_split', recipe, ARCH)}_{WIDE}")
    tiles, bands = problem.padded_n // 128, ceil_div(problem.m, BAND)
    partials = problem.workspace(tiles * splits * 2 * BAND * BAND * 4)
    tensors = problem.epilogue.tensors
    bias, residual, y = tensors.get("bias"), tensors.get("residual"), problem.outputs["y"]
    values = (problem.tensor_map("a0", a.values, (info.tile_m, info.tile_k)),
              problem.tensor_map(f"b{WIDE}", b.values, (info.tile_n, info.tile_k)), y.data_ptr(),
              0 if bias is None else bias.data_ptr(), 0 if residual is None else residual.data_ptr(),
              a.scales.data_ptr(), b.scales.data_ptr(), problem.m, problem.n, problem.padded_n, problem.k,
              partials.data_ptr(), splits)
    grid = (ceil_div(problem.n, info.tile_n), bands - 1 + splits, 1)
    runtime.run(kernel, grid, info.threads, info.shared_bytes, SPLIT_SIGNATURE, values, problem.stream)
    row = (bands - 1) * BAND
    rows, rowwise = problem.m - row, recipe == "fp8_rowwise"
    reduce, reduce_info = runtime.load_for(problem.device_index, decode.reduce_name(recipe, problem.epilogue.kind))
    values = (partials.data_ptr(), rows_from(y, row).data_ptr(), 0 if bias is None else bias.data_ptr(),
              0 if residual is None else rows_from(residual, row).data_ptr(),
              rows_from(a.scales, row).data_ptr() if rowwise else 0, b.scales.data_ptr() if rowwise else 0, rows,
              problem.n, problem.padded_n, problem.k, tiles * splits, BAND)
    grid = (tiles, ceil_div(rows, decode.REDUCE_ROWS), 1)
    runtime.run(reduce, grid, reduce_info.threads, reduce_info.shared_bytes, decode.REDUCE_SIGNATURE, values,
                problem.stream, pdl=True)


def mm_abt(problem: fp8_mm.Problem):
    kind = problem.epilogue.kind
    if kind not in EPILOGUES or (kind == "low_rank" and problem.recipe != "mxfp8"):
        raise NotImplementedError(f"{kernel_name(problem.recipe)} has no {kind!r} epilogue")
    tile = TILES[problem.recipe]
    sms = multiprocessors(problem.device_index)
    if tile == WIDE and problem.recipe in SPLIT_RECIPES and kind in SPLIT_EPILOGUES:
        splits = split_count(problem.m, problem.n, problem.padded_n, problem.k, sms)
        if splits:
            launch_split(problem, splits)
            return
    tail = tail_bands(problem.m, problem.n, sms) if tile == WIDE else 0
    split = (ceil_div(problem.m, BAND) - tail) * BAND
    if split > 0:
        launch(problem, tile, 0, min(split, problem.m))
    if tail > 0:
        launch(problem, SQUARE, split, problem.m)


def mm_glu(problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    half, hidden, z, recipe = problem.n // 2, parameters["hidden"], outputs["z"], problem.recipe
    if problem.m == 0 or (recipe, z is not None) not in FUSED_GLU:
        glu_quant.mm_glu(problem)
        return
    name = f"{recipe}_mm_glu_{parameters['activation']}_sm{ARCH}_{WIDE}"
    kernel, info = runtime.load_for(problem.device_index, name)
    a, b = problem.a, problem.b
    h, s_h = outputs["h"], outputs["s_h"]
    if recipe == "fp8_rowwise":
        h = problem.workspace(problem.m * hidden * 2).view(torch.bfloat16).view(problem.m, hidden)
    values = (problem.tensor_map("a", a.values, (info.tile_m, info.tile_k)),
              problem.tensor_map("b_glu", b.values, (GLU_BOX, info.tile_k)), h.data_ptr(),
              0 if recipe == "fp8_rowwise" else s_h.data_ptr(), 0 if z is None else z.data_ptr(),
              a.scales.data_ptr(), b.scales.data_ptr(), problem.m, half, hidden, problem.k,
              int(parameters["gate_first"]), glu_quant.bf16_limit(parameters["clamp_limit"]))
    grid = (ceil_div(half, info.tile_n // 2), ceil_div(problem.m, info.tile_m), 1)
    runtime.run(kernel, grid, info.threads, info.shared_bytes, GLU_SIGNATURE, values, problem.stream)
    if recipe == "fp8_rowwise":
        quant = "fp8_rowwise_quant_bf16"
        rows_per_group = runtime.info(quant).threads // 32
        glu_quant.run(quant, (ceil_div(problem.m, rows_per_group), 1, 1), glu_quant.QUANT_SIGNATURE,
                      (h.data_ptr(), outputs["h"].data_ptr(), s_h.data_ptr(), problem.m, hidden), problem.stream)


def mm_qkv_prep(problem):
    parameters, outputs, tensors = problem.epilogue.parameters, problem.outputs, problem.epilogue.tensors
    if problem.m == 0 or parameters["head_dim"] != QKV_HEAD_DIM or not FUSED_QKV:
        qkv_prep.mm_qkv_prep(problem)
        return
    qk_scales, rope = tensors["qk_scales"], tensors["rope"]
    assert rope.dtype == torch.float32 and rope.is_contiguous() and qk_scales.is_contiguous()
    z = outputs["saved_qkv"] if parameters["save_qkv"] else None
    kernel, info = runtime.load_for(problem.device_index, f"mxfp8_mm_qkv_prep_d{QKV_HEAD_DIM}_sm{ARCH}_{WIDE}")
    a, b = problem.a, problem.b
    values = (problem.tensor_map("a", a.values, (info.tile_m, info.tile_k)),
              problem.tensor_map("b", b.values, (info.tile_n, info.tile_k)), outputs["q"].data_ptr(),
              outputs["kv"].data_ptr(), 0 if z is None else z.data_ptr(), a.scales.data_ptr(), b.scales.data_ptr(),
              qk_scales.data_ptr(), rope.data_ptr(), int(qk_scales.dtype == torch.bfloat16), problem.m,
              problem.n // 3, problem.k, rope.numel() // problem.m, parameters["eps"])
    grid = (ceil_div(problem.n, info.tile_n), ceil_div(problem.m, info.tile_m), 1)
    runtime.run(kernel, grid, info.threads, info.shared_bytes, QKV_SIGNATURE, values, problem.stream)


for recipe in fp8_mm.RECIPES:
    fp8_mm.register(ARCH, recipe, "mm_abt", mm_abt)
    fp8_mm.register(ARCH, recipe, "mm_glu", mm_glu)
fp8_mm.register(ARCH, "mxfp8", "mm_qkv_prep", mm_qkv_prep)
