"""Launchers of `zig/qkv_prep.zig` and the operation `mm_qkv_prep`, which writes the BF16 product `Z` of `mm_abt`
to memory and then normalizes and rotates the q and k heads of `Z` into `Q` and `KV` with a second kernel."""

import torch

from . import fp8_mm, packs, runtime

ARCH = 89
CHUNK = 8
SIGNATURE = "ppppupuuuf"


def mxfp8_qkv_prep(z, q, kv, qk_scales, rope, eps, head_dim, stream):
    """Writes `q` `[M, H·D]` and `kv` `[M, 2·H·D]` from `z` `[M, 3·H·D]` with `D = head_dim`. `rope` holds the cosine
    and sine of each interleaved pair of each row."""
    rows, attn_dim = q.shape
    assert z.dtype == torch.bfloat16 and z.is_contiguous() and z.shape == (rows, 3 * attn_dim)
    assert rope.dtype == torch.float32 and rope.is_contiguous() and qk_scales.is_contiguous()
    assert attn_dim % fp8_mm.QKV_PLANE_COLUMNS == 0 and head_dim in fp8_mm.HEAD_DIMS
    rope_width = rope.numel() // rows if rows else 0
    name = f"mxfp8_qkv_prep_d{head_dim}"
    kernel, info = runtime.load(name)
    chunks = rows * 3 * attn_dim // CHUNK
    if chunks:
        runtime.run(kernel, (packs.ceil_div(chunks, info.threads), 1, 1), info.threads, 0, SIGNATURE,
                    (z.data_ptr(), q.data_ptr(), kv.data_ptr(), qk_scales.data_ptr(),
                     int(qk_scales.dtype == torch.bfloat16), rope.data_ptr(), rows, attn_dim, rope_width, eps),
                    stream)


def mm_qkv_prep(problem):
    parameters, outputs = problem.epilogue.parameters, problem.outputs
    if parameters["save_qkv"]:
        z = outputs["saved_qkv"]
    else:
        z = problem.workspace(problem.m * problem.n * 2).view(torch.bfloat16).view(problem.m, problem.n)
    product = fp8_mm.Problem("mm_abt", problem.recipe, problem.a, problem.b, problem.m, problem.n, problem.padded_n,
                             problem.k, fp8_mm.Epilogue("store"), {"y": z}, problem.device_index, problem.stream)
    fp8_mm.launch(product)
    tensors = problem.epilogue.tensors
    mxfp8_qkv_prep(z, outputs["q"], outputs["kv"], tensors["qk_scales"], tensors["rope"], parameters["eps"],
                   parameters["head_dim"], problem.stream)


fp8_mm.register(ARCH, "mxfp8", "mm_qkv_prep", mm_qkv_prep)
