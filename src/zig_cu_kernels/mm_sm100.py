"""The SM100 `mm_abt` and `mm_atb` kernels. A cluster of two CTAs computes 256x256 tiles of `Y = A Bᵀ`, or of
`D = Aᵀ B` over the columnwise MXFP8 packs `[K, M]` and `[K, N]`, with `tcgen05.mma.cta_group::2`. The `mxfp8` and
`fp8_rowwise` grids hold one cluster per two SMs, and each cluster walks the tiles; the `fp8_b128_e8m0` grid holds one
cluster per tile. The `mxfp8` `low_rank` kernel adds `low · upᵀ` with BF16 MMAs into the same accumulator after the K
blocks of each tile. The `mxfp8` `mm_glu` kernels let CTA 0 multiply the `lo` rows and CTA 1 the `hi` rows of `w1`, and
write the pack of `h` (and `Z` when asked) from the FP32 sums for each pair `(recipe, out_z)` in `FUSED_GLU`; every other
`mm_glu` and `mm_qkv_prep` run the composed chains of `glu_quant` and `qkv_prep`."""

import functools

import torch

from . import fp8_mm, glu_quant, packs, qkv_prep, runtime

ARCH = 100
TILE = 256
HALF = TILE // 2
K_BLOCK = 128
STORE_BOX = (32, 64)
SCALE_ATOM_WORDS = 128
CLUSTER = (2, 1, 1)
ALIGNMENT = 16
WEIGHT_SPANS = 88
SIGNATURE = "m" * 5 + "p" * 5 + "u" * 8
LOW_RANK_SIGNATURE = "m" * 7 + "p" * 2 + "u" * 7
RANK_BLOCK = 64
RANK_STEP = 16
KERNELS = {"mxfp8": "mxfp8_mm_abt_sm100", "fp8_b128_e8m0": "fp8_b128_mm_abt_sm100",
           "fp8_rowwise": "fp8_rowwise_mm_abt_sm100"}
EPILOGUES = ("store", "bias", "residual")
ATB_KERNEL = "mxfp8_mm_atb_sm100"
LOW_RANK_KERNEL = "mxfp8_mm_abt_low_rank_sm100"
GLU_SIGNATURE = "m" * 5 + "p" * 3 + "u" * 7 + "f"
FUSED_GLU = frozenset({("mxfp8", False), ("mxfp8", True)})


def scale_atoms(scales):
    return scales.view(torch.float32).view(-1, SCALE_ATOM_WORDS)


@functools.cache
def resident_clusters(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count // CLUSTER[0]


def pointer(tensor):
    if tensor is None:
        return 0
    assert tensor.data_ptr() % ALIGNMENT == 0, "the SM100 mm_abt kernels need 16-byte aligned epilogue tensors"
    return tensor.data_ptr()


def mm_abt(problem: fp8_mm.Problem):
    kind = problem.epilogue.kind
    low_rank = kind == "low_rank" and problem.recipe == "mxfp8"
    if kind not in EPILOGUES and not low_rank:
        raise NotImplementedError(f"the kernel {KERNELS[problem.recipe]} has no {kind} epilogue")
    kernel, info = runtime.load_for(problem.device_index, LOW_RANK_KERNEL if low_rank else KERNELS[problem.recipe])
    a, b, y = problem.a, problem.b, problem.outputs["y"]
    rows, k = y.shape[0], problem.k
    k_blocks = packs.ceil_div(k, K_BLOCK)
    tiles_m, tiles_n = packs.ceil_div(rows, TILE), packs.ceil_div(problem.n, TILE)
    value_maps = (problem.tensor_map("a", a.values, (HALF, K_BLOCK)), problem.tensor_map("b", b.values, (HALF, K_BLOCK)))
    y_map = problem.tensor_map("y", y, STORE_BOX)
    weight_scales = row_scales = column_scales = 0
    scale_blocks = k_blocks
    if problem.recipe == "fp8_rowwise":
        scale_maps = value_maps
        row_scales, column_scales = pointer(a.scales), pointer(b.scales)
    else:
        a_atoms = problem.tensor_map("s_a", scale_atoms(a.scales), (1, SCALE_ATOM_WORDS), swizzle=0)
        if problem.recipe == "mxfp8":
            scale_maps = (a_atoms, problem.tensor_map("s_b", scale_atoms(b.scales), (1, SCALE_ATOM_WORDS), swizzle=0))
        else:
            scale_maps = (a_atoms, a_atoms)
            scale_blocks = a.scales.shape[1]
            assert scale_blocks <= WEIGHT_SPANS, f"the kernel {KERNELS[problem.recipe]} takes K <= {WEIGHT_SPANS * 512}"
            weight_scales = b.scales.data_ptr()
    tensors = problem.epilogue.tensors
    residual = tensors.get("residual")
    sizes = (rows, problem.n, k_blocks, tiles_m, tiles_n, scale_blocks)
    if low_rank:
        low, up = tensors["low"], tensors["up"]
        rank_maps = (problem.tensor_map("low", low, (HALF, RANK_BLOCK)), problem.tensor_map("up", up, (HALF, RANK_BLOCK)))
        values = (*value_maps, *scale_maps, y_map, *rank_maps, pointer(tensors.get("bias")), 0, *sizes,
                  low.shape[1] // RANK_STEP)
    else:
        values = (*value_maps, *scale_maps, y_map, weight_scales, row_scales, column_scales,
                  pointer(tensors.get("bias")), pointer(residual), *sizes, problem.padded_n // HALF,
                  0 if residual is None else residual.stride(0))
    clusters = tiles_m * tiles_n
    if problem.recipe != "fp8_b128_e8m0":
        clusters = min(clusters, resident_clusters(problem.device_index))
    runtime.run(kernel, (2 * clusters, 1, 1), info.threads, info.shared_bytes,
                LOW_RANK_SIGNATURE if low_rank else SIGNATURE, values, problem.stream, cluster=CLUSTER)


def mm_atb(problem: fp8_mm.Problem):
    if problem.epilogue.kind != "store":
        raise NotImplementedError(f"the kernel {ATB_KERNEL} has no {problem.epilogue.kind} epilogue")
    kernel, info = runtime.load_for(problem.device_index, ATB_KERNEL)
    a, b, y = problem.a, problem.b, problem.outputs["y"]
    rows = y.shape[0]
    tiles_m, tiles_n = packs.ceil_div(rows, TILE), packs.ceil_div(problem.n, TILE)
    maps = (problem.tensor_map("a", a.values, (K_BLOCK, HALF)), problem.tensor_map("b", b.values, (K_BLOCK, HALF)),
            problem.tensor_map("s_a", scale_atoms(a.scales), (1, SCALE_ATOM_WORDS), swizzle=0),
            problem.tensor_map("s_b", scale_atoms(b.scales), (1, SCALE_ATOM_WORDS), swizzle=0),
            problem.tensor_map("y", y, STORE_BOX))
    values = (*maps, 0, 0, 0, 0, 0, rows, problem.n, packs.ceil_div(problem.k, K_BLOCK), tiles_m, tiles_n,
              rows // HALF, problem.padded_n // HALF, 0)
    clusters = min(tiles_m * tiles_n, resident_clusters(problem.device_index))
    runtime.run(kernel, (2 * clusters, 1, 1), info.threads, info.shared_bytes, SIGNATURE, values, problem.stream,
                cluster=CLUSTER)


def mm_glu(problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    hidden, z = parameters["hidden"], outputs["z"]
    fused = (problem.recipe, z is not None) in FUSED_GLU and (z is None or hidden % 8 == 0)
    if problem.m == 0 or not fused:
        glu_quant.mm_glu(problem)
        return
    kernel, info = runtime.load_for(problem.device_index, f"mxfp8_mm_glu_{parameters['activation']}_sm{ARCH}")
    a, b = problem.a, problem.b
    half_blocks = problem.n // 2 // HALF
    k_blocks = packs.ceil_div(problem.k, K_BLOCK)
    tiles_m = packs.ceil_div(problem.m, TILE)
    maps = (problem.tensor_map("a", a.values, (HALF, K_BLOCK)), problem.tensor_map("b", b.values, (HALF, K_BLOCK)),
            problem.tensor_map("s_a", scale_atoms(a.scales), (1, SCALE_ATOM_WORDS), swizzle=0),
            problem.tensor_map("s_b", scale_atoms(b.scales), (1, SCALE_ATOM_WORDS), swizzle=0))
    z_map = maps[0] if z is None else problem.tensor_map("z", z.view(problem.m, 2, hidden), (STORE_BOX[0], 1, STORE_BOX[1]))
    values = (*maps, z_map, pointer(outputs["h"]), pointer(outputs["s_h"]), pointer(z), problem.m, hidden, k_blocks, tiles_m,
              half_blocks, half_blocks, int(parameters["gate_first"]), glu_quant.bf16_limit(parameters["clamp_limit"]))
    clusters = min(tiles_m * half_blocks, resident_clusters(problem.device_index))
    runtime.run(kernel, (2 * clusters, 1, 1), info.threads, info.shared_bytes, GLU_SIGNATURE, values, problem.stream,
                cluster=CLUSTER)


def mm_qkv_prep(problem):
    qkv_prep.mm_qkv_prep(problem)


for recipe in KERNELS:
    fp8_mm.register(ARCH, recipe, "mm_abt", mm_abt)
    fp8_mm.register(ARCH, recipe, "mm_glu", mm_glu)
fp8_mm.register(ARCH, "mxfp8", "mm_qkv_prep", mm_qkv_prep)
fp8_mm.register(ARCH, "mxfp8", "mm_atb", mm_atb)
