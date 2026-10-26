"""Launcher of `zig/glu_bwd.zig`: the backward of the GLU product `h` of `z = [first | second]`, written as the MXFP8
packs that the gradient products read."""

import math

import torch

from . import packs, runtime

ACTIVATIONS = ("swiglu", "geglu_tanh")
SIGNATURE = "ppppppppuuuuf"
BF16_SIGNATURE = "ppppppuuuf"
BLOCK_ROWS = 32
HIDDEN_MULTIPLE = 64
MAX_U32 = (1 << 32) - 1


def require(condition, message):
    if not condition:
        raise ValueError(f"mxfp8_glu_backward: {message}")


def check(dh, z, gate_first, clamp_limit, activation):
    require(activation in ACTIVATIONS, f"activation must be one of {ACTIVATIONS}, got {activation!r}")
    require(activation == "swiglu" or (gate_first and clamp_limit in (None, math.inf)),
            "geglu_tanh takes gate_first=True and no clamp_limit")
    require(clamp_limit is None or clamp_limit > 0, f"clamp_limit must be positive, got {clamp_limit}")
    for name, x in (("dh", dh), ("z", z)):
        require(x.dtype == torch.bfloat16 and x.dim() == 2, f"{name} must be a 2-D bfloat16 tensor")
        require(x.is_contiguous(), f"{name} must be contiguous")
    rows, hidden = dh.shape
    require(rows > 0 and hidden > 0 and hidden % HIDDEN_MULTIPLE == 0,
            f"dh must be [M, F] with M > 0 and F a positive multiple of {HIDDEN_MULTIPLE}, got {tuple(dh.shape)}")
    require(tuple(z.shape) == (rows, 2 * hidden), f"z must be [{rows}, {2 * hidden}], got {tuple(z.shape)}")
    require(z.device == dh.device, "dh and z must be on one device")
    return rows, hidden


def bf16_limit(clamp_limit):
    return math.inf if clamp_limit is None else torch.tensor(clamp_limit, dtype=torch.float32).bfloat16().item()


def output_shapes(rows, hidden):
    """Returns the shapes and dtypes of `(dz_rw, s_dz_rw, dz_cw, s_dz_cw, h_cw, s_h_cw)`."""
    atoms, padded = packs.ceil_div(rows, packs.ATOM_ROWS), packs.round_up(hidden, packs.ATOM_ROWS)
    dz = ((rows, 2 * hidden), torch.float8_e4m3fn)
    dz_scales = ((atoms, 2 * hidden // packs.ATOM_ROWS, 32, 16), torch.uint8)
    return dz, dz_scales, dz, dz_scales, ((rows, padded), torch.float8_e4m3fn), \
        ((atoms, padded // packs.ATOM_ROWS, 32, 16), torch.uint8)


def mxfp8_glu_backward(dh, z, gate_first=False, clamp_limit=None, activation="swiglu"):
    """Returns `(dz_rw, s_dz_rw, dz_cw, s_dz_cw, h_cw, s_h_cw)` for `dh` `[M, F]` and `z` `[M, 2F]`: the rowwise and the
    columnwise MXFP8 packs of the BF16 gradient `dz` `[M, 2F]` in the half order of `z`, and the columnwise pack of the
    BF16 product `h` recomputed from `z`, `[M, round128(F)]` with zero pad columns. `gate` is the first half when
    `gate_first`, else the second."""
    rows, hidden = check(dh, z, gate_first, clamp_limit, activation)
    require(dh.is_cuda, "dh and z must be CUDA tensors")
    outputs = tuple(torch.empty(shape, dtype=dtype, device=dh.device) for shape, dtype in output_shapes(rows, hidden))
    padded = outputs[4].shape[1]
    kernel, info = runtime.load(f"mxfp8_glu_backward_{activation}")
    items = packs.round_up(rows, packs.ATOM_ROWS) // BLOCK_ROWS * padded // 2
    require(items <= MAX_U32, "the input is too large")
    limit = bf16_limit(clamp_limit) if activation == "swiglu" else math.inf
    values = (dh.data_ptr(), z.data_ptr(), *(tensor.data_ptr() for tensor in outputs), rows, hidden, padded,
              int(gate_first), limit)
    stream = torch.cuda.current_stream(dh.device).cuda_stream
    runtime.run(kernel, (packs.ceil_div(items, info.threads), 1, 1), info.threads, 0, SIGNATURE, values, stream)
    return outputs


def bf16_output_shapes(rows, hidden, out_h, out_q):
    """Returns the shapes and dtypes of `(dz, dz_q, s_dz, h)`, None for an output that is not asked for."""
    dz = ((rows, 2 * hidden), torch.bfloat16)
    dz_q, s_dz = output_shapes(rows, hidden)[:2] if out_q else (None, None)
    return dz, dz_q, s_dz, ((rows, hidden), torch.bfloat16) if out_h else None


def mxfp8_glu_backward_bf16(dh, z, gate_first=False, out_h=False, out_q=False, clamp_limit=None, activation="swiglu"):
    """Returns `(dz, dz_q, s_dz, h)`: the BF16 gradient `dz` `[M, 2F]` of `mxfp8_glu_backward`, its rowwise MXFP8 pack
    `(dz_q, s_dz)` with `out_q`, and the BF16 product `h` `[M, F]` with `out_h`, else None."""
    rows, hidden = check(dh, z, gate_first, clamp_limit, activation)
    require(dh.is_cuda, "dh and z must be CUDA tensors")
    outputs = tuple(None if shape is None else torch.empty(shape[0], dtype=shape[1], device=dh.device)
                    for shape in bf16_output_shapes(rows, hidden, out_h, out_q))
    kernel, info = runtime.load(f"mxfp8_glu_backward_bf16_{activation}")
    items = packs.round_up(rows, packs.ATOM_ROWS) // BLOCK_ROWS * hidden // 2
    require(items <= MAX_U32, "the input is too large")
    limit = bf16_limit(clamp_limit) if activation == "swiglu" else math.inf
    pointers = (0 if tensor is None else tensor.data_ptr() for tensor in outputs)
    values = (dh.data_ptr(), z.data_ptr(), *pointers, rows, hidden, int(gate_first), limit)
    stream = torch.cuda.current_stream(dh.device).cuda_stream
    runtime.run(kernel, (packs.ceil_div(items, info.threads), 1, 1), info.threads, 0, BF16_SIGNATURE, values, stream)
    return outputs
