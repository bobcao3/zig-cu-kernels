"""Launchers of `zig/glu_quant.zig` and the GLU operation `mm_glu` composed from `mm_abt`. `mm_glu` writes the
product `Z` to memory in BF16, and a second kernel computes the GLU product `H` of `Z` and writes its pack."""

import math

import torch

from . import fp8_mm, packs, runtime

ARCH = 89
CHUNK = 16
BLOCK_SIGNATURE = "pppuuuuuuf"
ROW_SIGNATURE = "ppuuuuf"
ONE_PASS_SIGNATURE = "pppuuuf"
ONE_PASS_CHUNKS = 6
QUANT_SIGNATURE = "pppuu"
PACKS = {"mxfp8": "mxfp8_glu_quant", "fp8_b128_e8m0": "fp8_b128_glu_quant_e8m0",
         "fp8_b128_fp32": "fp8_b128_glu_quant_f32", "fp8_rowwise": "fp8_rowwise_glu_bf16"}


def run(name, grid, signature, values, stream):
    kernel, info = runtime.load(name)
    if grid[0] > 0:
        runtime.run(kernel, grid, info.threads, 0, signature, values, stream)


def bf16_limit(clamp_limit):
    if clamp_limit is None:
        return math.inf
    return torch.tensor(clamp_limit, dtype=torch.float32).bfloat16().item()


def glu_quant(z, h, s_h, recipe, activation, gate_first, clamp_limit, hidden, stream, workspace=None):
    """Writes the pack `(h, s_h)` of the GLU product of `z` `[M, 2·padded]`, whose halves `[lo | hi]` hold `hidden`
    columns each. `gate` is `lo` when `gate_first`, else `hi`. A rowwise pack of a row wider than one work group
    holds goes through a BF16 temporary from `workspace(byte_count)`."""
    rows, padded = z.shape[0], z.shape[1] // 2
    assert recipe == "mxfp8" or hidden % packs.GROUP == 0, hidden
    assert z.dtype == torch.bfloat16 and z.is_contiguous() and padded % CHUNK == 0
    name = f"{PACKS[recipe]}_{activation}"
    threads = runtime.info(name).threads
    common = (int(gate_first), bf16_limit(clamp_limit))
    if recipe == "fp8_rowwise" and activation == "swiglu" and hidden <= ONE_PASS_CHUNKS * threads * CHUNK:
        run(f"fp8_rowwise_glu_quant_{activation}", (rows, 1, 1), ONE_PASS_SIGNATURE,
            (z.data_ptr(), h.data_ptr(), s_h.data_ptr(), padded, hidden, *common), stream)
        return None
    if recipe == "fp8_rowwise":
        byte_count = rows * hidden * 2
        buffer = workspace(byte_count) if workspace else torch.empty(byte_count, dtype=torch.uint8, device=z.device)
        h_bf16 = buffer.view(torch.bfloat16).view(rows, hidden)
        run(name, (packs.ceil_div(rows * hidden // CHUNK, threads), 1, 1), ROW_SIGNATURE,
            (z.data_ptr(), h_bf16.data_ptr(), rows, padded, hidden, *common), stream)
        quant = "fp8_rowwise_quant_bf16"
        rows_per_group = runtime.info(quant).threads // 32
        run(quant, (packs.ceil_div(rows, rows_per_group), 1, 1), QUANT_SIGNATURE,
            (h_bf16.data_ptr(), h.data_ptr(), s_h.data_ptr(), rows, hidden), stream)
        return h_bf16
    code_width = h.shape[1]
    scale_width = s_h.shape[1] * packs.ATOM_BLOCKS * (packs.MXFP8_BLOCK if recipe == "mxfp8" else packs.GROUP)
    chunks = packs.round_up(rows, packs.ATOM_ROWS) * scale_width // CHUNK
    run(name, (packs.ceil_div(chunks, threads), 1, 1), BLOCK_SIGNATURE,
        (z.data_ptr(), h.data_ptr(), s_h.data_ptr(), rows, padded, hidden, code_width, scale_width, *common), stream)
    return None


def mm_glu(problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    padded, hidden = problem.n // 2, parameters["hidden"]
    z = outputs["z"]
    if z is not None and hidden == padded:
        z_full = z
    else:
        z_full = problem.workspace(problem.m * problem.n * 2).view(torch.bfloat16).view(problem.m, problem.n)
    product = fp8_mm.Problem("mm_abt", problem.recipe, problem.a, problem.b, problem.m, problem.n, problem.padded_n,
                             problem.k, fp8_mm.Epilogue("store"), {"y": z_full}, problem.device_index, problem.stream)
    fp8_mm.launch(product)
    with torch.cuda.stream(torch.cuda.ExternalStream(problem.stream)):
        if z is not None and z is not z_full:
            z[:, :hidden].copy_(z_full[:, :hidden])
            z[:, hidden:].copy_(z_full[:, padded:padded + hidden])
        glu_quant(z_full, outputs["h"], outputs["s_h"], problem.recipe, parameters["activation"],
                  parameters["gate_first"], parameters["clamp_limit"], hidden, problem.stream, problem.workspace)


for recipe in PACKS:
    fp8_mm.register(ARCH, recipe, "mm_glu", mm_glu)
