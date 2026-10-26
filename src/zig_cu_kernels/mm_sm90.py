"""The SM90 `wgmma` kernels of `mm_abt` for the recipes `fp8_b128_e8m0`, `fp8_b128_fp32` and `fp8_rowwise`. A cluster of
two work groups computes two tiles of `Y` that share their columns. The grid holds at most one work group per SM, and each
cluster computes every tile pair whose index it reaches in steps of the number of clusters. When `THREE_CONSUMERS_MIN_K`
is set, a promoted rowwise product without a bias runs the kernel with three consumer warpgroups and 192 x 128 tiles
instead of 128 x 192 tiles when K is at least `THREE_CONSUMERS_MIN_K` and its clusters walk no more tiles each. It is
None by default, so every product runs a kernel with two consumer warpgroups. For each pair
`(recipe, out_z)` in `FUSED_GLU`, `mm_glu` runs a kernel whose 256-column tile holds the `lo` rows of `w1` from one work
group of the cluster and the `hi` rows from the other, and whose epilogue writes `Z` when asked and the pack of `h`. For
the rowwise recipe that kernel writes the BF16 `h`, and `fp8_rowwise_quant_bf16` then writes its pack; `FUSED_GLU`
leaves the rowwise recipe out, because that path was not faster than the chain. Every other case, including a `hidden`
smaller than half of the rows of `w1`, runs the composed chain of `glu_quant`. The block-scaled store kernels copy the
activation scales with bulk copies, which need addresses that are multiples of 16 bytes, so `mm_abt` first copies
activation scales at any other address into a new tensor. When the idle SMs of the last wave of tiles would cost at
least `MIN_SPLIT_GAIN` of the run time, a block-scaled store product runs the split kernel, which divides the K blocks of
the tiles of that wave among work groups and adds their FP32 sums in a fixed order."""

import torch

from . import fp8_mm, glu_quant, runtime
from .packs import ceil_div

ARCH = 90
TILE_M = 128
K_BLOCK = 128
CLUSTER = 2
SCALE_ATOM_BLOCKS = 4
BULK_ALIGNMENT = 16
SIGNATURE = "mmppppuuuuu"
SPLIT_SIGNATURE = "mmpppppuuuuuuu"
SPLIT_TAIL = "_split"
MAX_SPLITS = 8
MIN_SPLIT_BLOCKS = 4
MIN_SPLIT_GAIN = 0.06
RECIPES = ("fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise")
EPILOGUES = ("store", "bias")
GLU_SIGNATURE = "mmpppppuuuuuuf"
FUSED_GLU = frozenset((recipe, out_z) for recipe in RECIPES[:2] for out_z in (False, True))
THREE_CONSUMERS = "_m192"
THREE_CONSUMERS_MIN_K = None


def kernel_name(recipe, epilogue):
    return fp8_mm.kernel_name("mm_abt", recipe, ARCH) + ("_bias" if epilogue == "bias" else "")


def mm_abt(problem: fp8_mm.Problem):
    kind = problem.epilogue.kind
    assert kind in EPILOGUES, f"{kernel_name(problem.recipe, 'store')} has no {kind!r} epilogue"
    name = kernel_name(problem.recipe, kind)
    if problem.recipe != "fp8_rowwise" and kind == "store" and problem.a.scales.data_ptr() % BULK_ALIGNMENT:
        problem.a = fp8_mm.Pack(problem.a.values, problem.a.scales.clone())
    if SPLIT_TAIL is not None and problem.recipe != "fp8_rowwise" and kind == "store":
        split_name = name + SPLIT_TAIL
        info = runtime.info(split_name)
        tiles = ceil_div(problem.m, CLUSTER * info.tile_m) * ceil_div(problem.padded_n, info.tile_n)
        slots = runtime.num_sm(problem.device_index) // CLUSTER
        plan = tail_split(tiles, slots, ceil_div(problem.k, K_BLOCK))
        if plan is not None:
            launch(problem, split_name, plan)
            return
    three_consumers = THREE_CONSUMERS_MIN_K is not None and problem.k >= THREE_CONSUMERS_MIN_K
    if problem.recipe == "fp8_rowwise" and kind == "store" and three_consumers:
        three = name + THREE_CONSUMERS
        if tiles_per_cluster(problem, runtime.info(three)) <= tiles_per_cluster(problem, runtime.info(name)):
            name = three
    launch(problem, name)


def split_plan(tiles, slots, k_blocks):
    """Returns the number of cluster tiles that run over all K blocks and the number of K ranges of each tile of a last
    round that holds at most half of `slots` tiles."""
    tail = tiles % slots
    if tiles <= slots or tail == 0:
        return tiles, 1
    return tiles - tail, max(1, min(slots // tail, k_blocks // MIN_SPLIT_BLOCKS, MAX_SPLITS))


def tail_split(tiles, slots, k_blocks):
    """Returns `split_plan` when the idle SMs of the last round of tiles would cost at least `MIN_SPLIT_GAIN` of the run
    time, and None otherwise."""
    whole, splits = split_plan(tiles, slots, k_blocks)
    if splits > 1 and (1 - 1 / splits) / ceil_div(tiles, slots) >= MIN_SPLIT_GAIN:
        return whole, splits
    return None


def tiles_per_cluster(problem: fp8_mm.Problem, info):
    tiles = ceil_div(problem.m, CLUSTER * info.tile_m) * ceil_div(problem.padded_n, info.tile_n)
    return ceil_div(tiles, runtime.num_sm(problem.device_index) // CLUSTER)


def fp8_rowwise_mm_abt_fast(problem: fp8_mm.Problem):
    """Computes a rowwise product without a bias in one `wgmma` accumulator over all of K, as
    `torch._scaled_mm(use_fast_accum=True)` does."""
    assert problem.recipe == "fp8_rowwise" and problem.epilogue.kind == "store", (problem.recipe, problem.epilogue)
    launch(problem, kernel_name("fp8_rowwise", "store") + "_fast")


def launch(problem: fp8_mm.Problem, name, split=None):
    kernel, info = runtime.load_for(problem.device_index, name)
    a, b = problem.a.values, problem.b.values
    a_map = problem.tensor_map("a", a, (info.tile_m, K_BLOCK))
    b_map = problem.tensor_map("b", b, (info.tile_n // CLUSTER, K_BLOCK))
    pairs_m = ceil_div(problem.m, CLUSTER * info.tile_m)
    tiles_n = ceil_div(problem.padded_n, info.tile_n)
    bias = problem.epilogue.tensors.get("bias")
    values = (a_map, b_map, problem.outputs["y"].data_ptr(), problem.a.scales.data_ptr(), problem.b.scales.data_ptr(),
              0 if bias is None else bias.data_ptr(), problem.m, problem.n, ceil_div(problem.k, K_BLOCK),
              SCALE_ATOM_BLOCKS * ceil_div(problem.k, SCALE_ATOM_BLOCKS * K_BLOCK), tiles_n)
    clusters = min(pairs_m * tiles_n, runtime.num_sm(problem.device_index) // CLUSTER)
    signature = SIGNATURE
    if split is not None:
        whole, splits = split
        groups = (pairs_m * tiles_n - whole) * CLUSTER * (info.tile_m // 64)
        partials = problem.workspace(groups * splits * info.tile_n // 8 * 128 * 16)
        counters = torch.zeros(groups, dtype=torch.int32, device=partials.device)
        problem.workspaces.append(counters)
        values = values[:5] + (partials.data_ptr(), counters.data_ptr()) + values[6:] + (whole, splits)
        signature = SPLIT_SIGNATURE
    runtime.run(kernel, (CLUSTER * clusters, 1, 1), info.threads, info.shared_bytes, signature, values,
                problem.stream, cluster=(CLUSTER, 1, 1))


def mm_glu(problem: fp8_mm.Problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    half, hidden, z, recipe = problem.n // 2, parameters["hidden"], outputs["z"], problem.recipe
    name = f"{recipe}_mm_glu_{parameters['activation']}_sm{ARCH}"
    if problem.m == 0 or (recipe, z is not None) not in FUSED_GLU or half % K_BLOCK or hidden != half:
        glu_quant.mm_glu(problem)
        return
    kernel, info = runtime.load_for(problem.device_index, name)
    a, b = problem.a, problem.b
    h, s_h = outputs["h"], outputs["s_h"]
    if recipe == "fp8_rowwise":
        h = problem.workspace(problem.m * hidden * 2).view(torch.bfloat16).view(problem.m, hidden)
    values = (problem.tensor_map("a", a.values, (TILE_M, K_BLOCK)),
              problem.tensor_map("b_glu", b.values, (info.tile_n // CLUSTER, K_BLOCK)), h.data_ptr(),
              0 if recipe == "fp8_rowwise" else s_h.data_ptr(), 0 if z is None else z.data_ptr(),
              a.scales.data_ptr(), b.scales.data_ptr(), problem.m, half, hidden, ceil_div(problem.k, K_BLOCK),
              SCALE_ATOM_BLOCKS * ceil_div(problem.k, SCALE_ATOM_BLOCKS * K_BLOCK), int(parameters["gate_first"]),
              glu_quant.bf16_limit(parameters["clamp_limit"]))
    tiles = ceil_div(problem.m, CLUSTER * TILE_M) * (half // (info.tile_n // 2))
    clusters = min(tiles, runtime.num_sm(problem.device_index) // CLUSTER)
    runtime.run(kernel, (CLUSTER * clusters, 1, 1), info.threads, info.shared_bytes, GLU_SIGNATURE, values,
                problem.stream, cluster=(CLUSTER, 1, 1))
    if recipe == "fp8_rowwise":
        quant = "fp8_rowwise_quant_bf16"
        rows_per_group = runtime.info(quant).threads // 32
        glu_quant.run(quant, (ceil_div(problem.m, rows_per_group), 1, 1), glu_quant.QUANT_SIGNATURE,
                      (h.data_ptr(), outputs["h"].data_ptr(), s_h.data_ptr(), problem.m, hidden), problem.stream)


for recipe in RECIPES:
    fp8_mm.register(ARCH, recipe, "mm_abt", mm_abt)
    fp8_mm.register(ARCH, recipe, "mm_glu", mm_glu)
