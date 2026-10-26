import math
from typing import Optional

import torch

from .. import fp8_gemm as launcher
from ..tensors import ALIGNMENT_BYTES
from .common import (
    ELEMENTS,
    SCALE_DTYPES,
    activation_block_scales,
    current_stream_handle,
    leading_rows,
    require_choice,
    require_dtype,
    require_one_element,
    require_tensorwise_scale,
    row_layout,
    workspace_for,
    writes_in_place,
)
from .ops import Output, define

__all__ = [
    "quantize_fp8_rows",
    "quantize_fp8_tiles",
    "quantize_fp8_tensorwise",
    "fp8_gemm",
    "fp8_gated_gemm",
    "fp8_gemm_op",
    "fp8_gemm_out_op",
    "fp8_gated_gemm_op",
    "fp8_gated_gemm_out_op",
    "quantize_fp8_rows_op",
    "quantize_fp8_rows_out_op",
    "quantize_fp8_tiles_op",
    "quantize_fp8_tiles_out_op",
    "quantize_fp8_tensorwise_op",
    "quantize_fp8_tensorwise_out_op",
    "quantize_fp8_static_op",
    "quantize_fp8_static_out_op",
]

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
SCALE_GROUP = launcher.SCALE_GROUP
SIZE_LIMIT = 1 << 32
QUANTIZER_VALUE_LIMIT = 1 << 35
INPUT_DTYPES = tuple(ELEMENTS)


def rows_of(tensor):
    """Returns the row count and the row stride in elements, or None if the kernels cannot access `tensor` in place."""
    layout = row_layout(tensor)
    if layout is None or tensor.data_ptr() % ALIGNMENT_BYTES:
        return None
    rows, _, stride_bytes = layout
    stride = stride_bytes // tensor.element_size()
    if rows > 1 and (stride_bytes % ALIGNMENT_BYTES or stride >= SIZE_LIMIT):
        return None
    return rows, stride


def readable(tensor):
    """Returns `tensor`, or a contiguous copy that the kernels can read in place, and the row stride of the result."""
    rows = rows_of(tensor)
    if rows is None:
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        rows = rows_of(tensor)
    return tensor, rows[1]


def require_below(limit, **sizes):
    for name, size in sizes.items():
        if size >= limit:
            raise ValueError(f"{name} = {size} must be below {limit}")


def scales_e8m0(*scales):
    return any(isinstance(scale, torch.Tensor) and scale.dtype == E8M0 for scale in scales)


def product_scaling_type(a, weights, scale_a, weight_scales, use_fast_accum):
    """Checks the shapes and sizes of `a`, the dict `weights` and the scales, and returns the scaling type."""
    lead, k = tuple(a.shape[:-1]), a.shape[-1]
    columns = next(iter(weights.values())).shape[0]
    for name, weight in weights.items():
        if tuple(weight.shape) != (columns, k):
            raise ValueError(f"{name} must have the shape {(columns, k)}, got {tuple(weight.shape)}")
    e8m0 = scales_e8m0(scale_a, *weight_scales)
    scaling_type = launcher.scaling_type_of(lead, columns, k, scale_a.shape, [scale.shape for scale in weight_scales],
                                            e8m0)
    launcher.check_product(scaling_type, columns, k, e8m0, use_fast_accum)
    require_below(SIZE_LIMIT, m=math.prod(lead))
    return scaling_type


def checked_scaling_type(a, columns, scale_a, weight_scales):
    """Returns the scaling type of a product whose shapes `product_scaling_type` has checked."""
    e8m0 = scales_e8m0(scale_a, *weight_scales)
    return launcher.scaling_type_of(tuple(a.shape[:-1]), columns, a.shape[-1], scale_a.shape,
                                 [scale.shape for scale in weight_scales], e8m0)


def scale_strides(scale):
    """Returns the row stride and the scale group stride of a scale tensor. Returns None when the leading dimensions do
    not flatten into one row stride or when the storage of an E8M0 tensor does not start at a multiple of 4 bytes."""
    rows = leading_rows(scale)
    if rows is None or (scale.dtype == E8M0 and scale.untyped_storage().data_ptr() % 4 != 0):
        return None
    return rows[1], scale.stride(-1)


def check_output(out, shape):
    if tuple(out.shape) != tuple(shape):
        raise ValueError(f"out must have the shape {tuple(shape)}, got {tuple(out.shape)}")


def scale_operand(name, scale, scaling_type, lead, scale_groups):
    """Returns `scale`, or a dense copy when the kernels cannot read `scale`, and the row and scale group strides in
    elements, with the stride 0 for a dimension of size 1."""
    if scaling_type == "tensorwise":
        if scale.data_ptr() % launcher.WORD_BYTES:
            scale = scale.clone()
        return scale, (0, 0)
    strides = scale_strides(scale)
    if strides is None or (scale.dtype == torch.float32 and scale.data_ptr() % launcher.WORD_BYTES):
        rows = math.prod(lead)
        copy = scale.new_empty((scale_groups, rows)).t() if scale_groups > 1 and name == "scale_a" else \
            scale.new_empty((rows, scale_groups))
        copy.copy_(scale.reshape(rows, scale_groups))
        scale, strides = copy, scale_strides(copy)
    return scale, tuple(stride if size > 1 else 0 for stride, size in zip(strides, (math.prod(lead), scale_groups)))


def scale_operands(scaling_type, scale_a, weight_scales, lead, columns, k):
    """Returns the scale tensors that the kernels read for `scale_a` and the dict `weight_scales`, the strides of the
    scale tensors and the E8M0 flags. Bit i of the flags marks scale i as E8M0, where `scale_a` is scale 0."""
    scale_groups = 1 if scaling_type != "block128" else k // SCALE_GROUP
    weight_lead = (columns,) if scaling_type != "block128" else (-(-columns // SCALE_GROUP),)
    operands = [scale_operand("scale_a", scale_a, scaling_type, lead, scale_groups)]
    operands += [scale_operand(name, scale, scaling_type, weight_lead, scale_groups)
                 for name, scale in weight_scales.items()]
    scales = (scale_a, *weight_scales.values())
    e8m0 = sum(1 << index for index, scale in enumerate(scales) if scale.dtype == E8M0)
    return [operand[0] for operand in operands], [operand[1] for operand in operands], e8m0


def bias_operand(bias):
    if bias is None or (bias.is_contiguous() and bias.data_ptr() % launcher.WORD_BYTES == 0):
        return bias
    return bias.clone(memory_format=torch.contiguous_format)


def run(kind, scaling_type, use_fast_accum, device_index, m, columns, k, sizes, pointers):
    """Launches a product on the current stream, with a workspace of the stream when the work groups share tiles."""
    stream = current_stream_handle(device_index)
    family = launcher.family_of(scaling_type, use_fast_accum)
    launch = launcher.product_launch(family, kind, m, columns, k, sizes, device_index, True)
    if not launch.part_bytes:
        launch(pointers, launcher.NO_BUFFERS, stream)
        return
    with workspace_for(device_index, stream) as workspace:
        launch(pointers, workspace.addresses(launch.part_bytes, launch.counters), stream)


def output_target(out, inputs, residual):
    """Returns `out` when the kernels can write `out` in place, or else a contiguous temporary."""
    if rows_of(out) is not None and writes_in_place(out, inputs, aliases=(residual,)):
        return out
    return torch.empty(out.shape, dtype=out.dtype, device=out.device)


def residual_operand(residual):
    if residual is None:
        return None, 0
    rows = rows_of(residual)
    if rows is None:
        return readable(residual)
    return residual, rows[1]


def check_gemm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
               bias: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str, use_fast_accum: bool):
    require_dtype("a", a, (FP8,))
    require_dtype("b", b, (FP8,))
    require_dtype("scale_a", scale_a, SCALE_DTYPES)
    require_dtype("scale_b", scale_b, SCALE_DTYPES)
    require_dtype("bias", bias, (torch.bfloat16,))
    require_dtype("residual", residual, (torch.bfloat16,))
    require_choice("activation", activation, tuple(launcher.ACTIVATIONS))
    if a.dim() < 1 or b.dim() != 2:
        raise ValueError("a must be [..., k], and b must be [n, k]")


def check_gemm_sizes(a, b, scale_a, scale_b, bias, residual, activation, use_fast_accum):
    product_scaling_type(a, {"b": b}, scale_a, (scale_b,), use_fast_accum)
    n = b.shape[0]
    if bias is not None and tuple(bias.shape) != (n,):
        raise ValueError(f"bias must have the shape {(n,)}, got {tuple(bias.shape)}")
    if residual is not None and tuple(residual.shape) != (*a.shape[:-1], n):
        raise ValueError(f"residual must have the shape {(*a.shape[:-1], n)}, got {tuple(residual.shape)}")


def gemm_output(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
                bias: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str,
                use_fast_accum: bool) -> torch.Tensor:
    return a.new_empty((*a.shape[:-1], b.shape[0]), dtype=torch.bfloat16)


def gemm_into(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor,
              bias: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str, use_fast_accum: bool,
              out: torch.Tensor) -> None:
    lead, (n, k) = tuple(a.shape[:-1]), b.shape
    check_output(out, (*lead, n))
    m = math.prod(lead)
    if m == 0 or n == 0:
        return
    scaling_type = checked_scaling_type(a, n, scale_a, (scale_b,))
    target = output_target(out, (a, b, scale_a, scale_b, bias), residual)
    (a, a_stride), (b, b_stride) = readable(a), readable(b)
    scales, strides, e8m0 = scale_operands(scaling_type, scale_a, {"scale_b": scale_b}, lead, n, k)
    (a_rows, a_blocks), (b_rows, b_blocks) = strides
    bias = bias_operand(bias)
    residual, residual_stride = residual_operand(residual)
    sizes = (a_stride, b_stride, a_rows, a_blocks, b_rows, b_blocks, e8m0, rows_of(target)[1], residual_stride, m, n,
             k, launcher.ACTIVATIONS[activation])
    pointers = (a.data_ptr(), b.data_ptr(), *(scale.data_ptr() for scale in scales), target.data_ptr(),
                0 if bias is None else bias.data_ptr(), 0 if residual is None else residual.data_ptr())
    run("store", scaling_type, use_fast_accum, a.get_device(), m, n, k, sizes, pointers)
    if target is not out:
        out.copy_(target)


FP8_GEMM = define("fp8_gemm", run=gemm_into, outputs={"out": Output(gemm_output, torch.bfloat16)}, check=check_gemm,
                  check_sizes=check_gemm_sizes)
fp8_gemm_op, fp8_gemm_out_op = FP8_GEMM.functional, FP8_GEMM.out


def fp8_gemm(a, b, scale_a, scale_b, *, bias=None, activation="none", residual=None, use_fast_accum=False, out=None):
    """Returns the BF16 product of the E4M3 activation `a` and the transposed E4M3 weight `b`, scaled by `scale_a` and
    `scale_b`, with the epilogues of `gemm` (README sections "Scaling types", "Accumulation" and "Custom ops")."""
    return FP8_GEMM(a, b, scale_a, scale_b, bias, residual, activation, use_fast_accum, outputs={"out": out})


def check_gated(a: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, scale_a: torch.Tensor,
                scale_gate: torch.Tensor, scale_up: torch.Tensor, bias_gate: Optional[torch.Tensor],
                bias_up: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str,
                use_fast_accum: bool):
    for name, tensor in (("a", a), ("w_gate", w_gate), ("w_up", w_up)):
        require_dtype(name, tensor, (FP8,))
    for name, tensor in (("scale_a", scale_a), ("scale_gate", scale_gate), ("scale_up", scale_up)):
        require_dtype(name, tensor, SCALE_DTYPES)
    for name, tensor in (("bias_gate", bias_gate), ("bias_up", bias_up), ("residual", residual)):
        require_dtype(name, tensor, (torch.bfloat16,))
    require_choice("activation", activation, tuple(launcher.ACTIVATIONS))
    if a.dim() < 1 or w_gate.dim() != 2 or w_up.dim() != 2:
        raise ValueError("a must be [..., k], and w_gate and w_up must be [f, k]")


def check_gated_sizes(a, w_gate, w_up, scale_a, scale_gate, scale_up, bias_gate, bias_up, residual, activation,
                      use_fast_accum):
    product_scaling_type(a, {"w_gate": w_gate, "w_up": w_up}, scale_a, (scale_gate, scale_up), use_fast_accum)
    f = w_gate.shape[0]
    for name, bias in (("bias_gate", bias_gate), ("bias_up", bias_up)):
        if bias is not None and tuple(bias.shape) != (f,):
            raise ValueError(f"{name} must have the shape {(f,)}, got {tuple(bias.shape)}")
    if residual is not None and tuple(residual.shape) != (*a.shape[:-1], f):
        raise ValueError(f"residual must have the shape {(*a.shape[:-1], f)}, got {tuple(residual.shape)}")


def gated_output(a: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, scale_a: torch.Tensor,
                 scale_gate: torch.Tensor, scale_up: torch.Tensor, bias_gate: Optional[torch.Tensor],
                 bias_up: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str,
                 use_fast_accum: bool) -> torch.Tensor:
    return a.new_empty((*a.shape[:-1], w_gate.shape[0]), dtype=torch.bfloat16)


def gated_weights(w_gate, w_up):
    """Returns `w_gate` and `w_up` in a form that the kernels read with one row stride, and that row stride."""
    (gate, gate_stride), (up, up_stride) = readable(w_gate), readable(w_up)
    if gate_stride != up_stride and gate.shape[0] > 1:
        gate, up = gate.contiguous(), up.contiguous()
        gate_stride = gate.shape[1]
    return gate, up, gate_stride


def gated_into(a: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, scale_a: torch.Tensor,
               scale_gate: torch.Tensor, scale_up: torch.Tensor, bias_gate: Optional[torch.Tensor],
               bias_up: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str,
               use_fast_accum: bool, out: torch.Tensor) -> None:
    lead, (f, k) = tuple(a.shape[:-1]), w_gate.shape
    check_output(out, (*lead, f))
    m = math.prod(lead)
    if m == 0 or f == 0:
        return
    scaling_type = checked_scaling_type(a, f, scale_a, (scale_gate, scale_up))
    target = output_target(out, (a, w_gate, w_up, scale_a, scale_gate, scale_up, bias_gate, bias_up), residual)
    a, a_stride = readable(a)
    gate, up, b_stride = gated_weights(w_gate, w_up)
    weight_scales = {"scale_gate": scale_gate, "scale_up": scale_up}
    scales, strides, e8m0 = scale_operands(scaling_type, scale_a, weight_scales, lead, f, k)
    (a_rows, a_blocks), (gate_rows, gate_blocks), (up_rows, up_blocks) = strides
    biases = [bias_operand(bias) for bias in (bias_gate, bias_up)]
    residual, residual_stride = residual_operand(residual)
    sizes = (a_stride, b_stride, a_rows, a_blocks, gate_rows, gate_blocks, up_rows, up_blocks, e8m0,
             rows_of(target)[1], residual_stride, m, f, k, launcher.ACTIVATIONS[activation])
    pointers = (a.data_ptr(), gate.data_ptr(), up.data_ptr(), *(scale.data_ptr() for scale in scales),
                target.data_ptr(), *(0 if bias is None else bias.data_ptr() for bias in biases),
                0 if residual is None else residual.data_ptr())
    run("gated", scaling_type, use_fast_accum, a.get_device(), m, f, k, sizes, pointers)
    if target is not out:
        out.copy_(target)


FP8_GATED_GEMM = define("fp8_gated_gemm", run=gated_into, outputs={"out": Output(gated_output, torch.bfloat16)},
                        check=check_gated, check_sizes=check_gated_sizes)
fp8_gated_gemm_op, fp8_gated_gemm_out_op = FP8_GATED_GEMM.functional, FP8_GATED_GEMM.out


def fp8_gated_gemm(a, w_gate, w_up, scale_a, scale_gate, scale_up, *, bias_gate=None, bias_up=None, activation="silu",
                   residual=None, use_fast_accum=False, out=None):
    """Returns the gated linear unit of `gated_gemm` for the E4M3 operands `a`, `w_gate` and `w_up` with the scales
    `scale_a`, `scale_gate` and `scale_up`. See the README sections "BF16 GEMM", "Scaling types" and "Accumulation"."""
    arguments = (a, w_gate, w_up, scale_a, scale_gate, scale_up, bias_gate, bias_up, residual, activation,
                 use_fast_accum)
    return FP8_GATED_GEMM(*arguments, outputs={"out": out})


def check_quantizer_input(name, x):
    require_dtype(name, x, INPUT_DTYPES)
    if x.dim() < 1:
        raise ValueError(f"{name} needs at least one dimension")


def check_quantizer_sizes(name, x):
    require_below(QUANTIZER_VALUE_LIMIT, **{f"{name}.numel()": x.numel()})
    require_below(SIZE_LIMIT, **{f"the rows of {name}": math.prod(x.shape[:-1]), f"{name}.shape[-1]": x.shape[-1]})


def check_groups(name, x, scale_dtype):
    check_quantizer_input(name, x)
    require_choice("scale_dtype", scale_dtype, SCALE_DTYPES)


def check_group_sizes(name, x):
    check_quantizer_sizes(name, x)
    if x.shape[-1] % SCALE_GROUP:
        raise ValueError(f"{name} needs a last dimension that is a multiple of {SCALE_GROUP}, got {x.shape[-1]}")


def codes_target(q, inputs):
    """Returns `q` when the kernels can write the codes to `q` in place, or else a contiguous temporary."""
    if q.is_contiguous() and q.data_ptr() % launcher.CODE_ALIGNMENT_BYTES == 0 and writes_in_place(q, inputs):
        return q
    return torch.empty(q.shape, dtype=FP8, device=q.device)


def copy_back(target, output):
    if target is not output:
        output.copy_(target)


def check_rows(x: torch.Tensor, scale_dtype: torch.dtype):
    check_groups("x", x, scale_dtype)


def check_rows_sizes(x, scale_dtype):
    check_group_sizes("x", x)


def codes_like(x: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=FP8)


def row_scales_like(x: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    return activation_block_scales(x, scale_dtype)


def rows_into(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor) -> None:
    m, k = math.prod(x.shape[:-1]), x.shape[-1]
    if tuple(q.shape) != tuple(x.shape) or tuple(scales.shape) != (m, k // SCALE_GROUP):
        raise ValueError(f"q must have the shape {tuple(x.shape)} and scales {(m, k // SCALE_GROUP)}")
    if m * k == 0:
        return
    source, stride = readable(x)
    codes = codes_target(q, (x,))
    in_place = (m <= 1 or scales.stride(0) == 1) and writes_in_place(scales, (x,)) and \
        (scales.dtype == E8M0 or scales.data_ptr() % launcher.WORD_BYTES == 0)
    targets = scales if in_place else activation_block_scales(x, scales.dtype)
    scale_stride = targets.stride(1) if k > SCALE_GROUP else m
    suffix = launcher.SCALE_SUFFIXES[str(scales.dtype).removeprefix("torch.")]
    launcher.launch_rows(f"fp8_quantize_rows_{ELEMENTS[x.dtype]}{suffix}", source.data_ptr(), stride, codes.data_ptr(),
                         targets.data_ptr(), scale_stride, m, k, current_stream_handle(x.get_device()))
    copy_back(codes, q)
    copy_back(targets, scales)


QUANTIZE_FP8_ROWS = define("quantize_fp8_rows", run=rows_into, check=check_rows, check_sizes=check_rows_sizes,
                           outputs={"q": Output(codes_like, FP8, "out"),
                                    "scales": Output(row_scales_like, "scale_dtype", "scales_out")})
quantize_fp8_rows_op, quantize_fp8_rows_out_op = QUANTIZE_FP8_ROWS.functional, QUANTIZE_FP8_ROWS.out


def quantize_fp8_rows(x, *, scale_dtype=torch.float32, out=None, scales_out=None):
    """Quantizes `x` to E4M3 with one scale per row and scale group and returns `(q, scales)` to pass as the block128
    `a` and `scale_a` of `fp8_gemm`. See the README sections "Quantizers" and "Custom ops"."""
    return QUANTIZE_FP8_ROWS(x, scale_dtype, outputs={"out": out, "scales_out": scales_out})


def check_tiles(w: torch.Tensor, scale_dtype: torch.dtype):
    check_groups("w", w, scale_dtype)
    if w.dim() != 2:
        raise ValueError("w must be an [n, k] matrix")


def check_tiles_sizes(w, scale_dtype):
    check_group_sizes("w", w)


def tile_codes_like(w: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    return w.new_empty(w.shape, dtype=FP8)


def tile_scales_like(w: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    return w.new_empty(((w.shape[0] + SCALE_GROUP - 1) // SCALE_GROUP, w.shape[-1] // SCALE_GROUP), dtype=scale_dtype)


def tiles_into(w: torch.Tensor, q: torch.Tensor, scales: torch.Tensor) -> None:
    (n, k), tile_rows = w.shape, -(-w.shape[0] // SCALE_GROUP)
    if tuple(q.shape) != (n, k) or tuple(scales.shape) != (tile_rows, k // SCALE_GROUP):
        raise ValueError(f"q must have the shape {(n, k)} and scales {(tile_rows, k // SCALE_GROUP)}")
    if n * k == 0:
        return
    source, stride = readable(w)
    codes = codes_target(q, (w,))
    in_place = (k <= SCALE_GROUP or scales.stride(1) == 1) and writes_in_place(scales, (w,)) and \
        (scales.dtype == E8M0 or scales.data_ptr() % launcher.WORD_BYTES == 0)
    targets = scales if in_place else torch.empty(scales.shape, dtype=scales.dtype, device=scales.device)
    scale_stride = targets.stride(0) if tile_rows > 1 else k // SCALE_GROUP
    suffix = launcher.SCALE_SUFFIXES[str(scales.dtype).removeprefix("torch.")]
    launcher.launch_tiles(f"fp8_quantize_tiles_{ELEMENTS[w.dtype]}{suffix}", source.data_ptr(), stride,
                          codes.data_ptr(), targets.data_ptr(), scale_stride, n, k,
                          current_stream_handle(w.get_device()))
    copy_back(codes, q)
    copy_back(targets, scales)


QUANTIZE_FP8_TILES = define("quantize_fp8_tiles", run=tiles_into, check=check_tiles, check_sizes=check_tiles_sizes,
                            outputs={"q": Output(tile_codes_like, FP8, "out"),
                                     "scales": Output(tile_scales_like, "scale_dtype", "scales_out")})
quantize_fp8_tiles_op, quantize_fp8_tiles_out_op = QUANTIZE_FP8_TILES.functional, QUANTIZE_FP8_TILES.out


def quantize_fp8_tiles(w, *, scale_dtype=torch.float32, out=None, scales_out=None):
    """Quantizes the weight `w` to E4M3 with one scale per 128 x 128 tile and returns `(q, scales)` to pass as the
    block128 `b` and `scale_b` of `fp8_gemm`. See the README sections "Quantizers" and "Custom ops"."""
    return QUANTIZE_FP8_TILES(w, scale_dtype, outputs={"out": out, "scales_out": scales_out})


def check_tensorwise(x: torch.Tensor):
    require_dtype("x", x, INPUT_DTYPES)


def check_tensorwise_sizes(x):
    check_quantizer_sizes("x", x.reshape(-1) if x.dim() == 0 else x)
    if (x.dim() == 0 or x.shape[-1] % 8) and x.numel() % 8:
        raise ValueError(f"x needs a last dimension or a number of values that is a multiple of 8, got "
                         f"{tuple(x.shape)}")


def scalar_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((), dtype=torch.float32)


def tensorwise_codes_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=FP8)


def tensorwise_source(x):
    """Returns `x` or a copy as a matrix that the kernels can read, and the row count, width and row stride."""
    matrix = x.reshape(1, -1) if x.dim() == 0 or x.shape[-1] % 8 else x
    rows = rows_of(matrix)
    if rows is None:
        matrix = x.clone(memory_format=torch.contiguous_format)
        matrix = matrix.reshape(1, -1) if x.dim() == 0 or x.shape[-1] % 8 else matrix
        rows = rows_of(matrix)
    width = matrix.shape[-1]
    return matrix, math.prod(matrix.shape[:-1]), width, rows[1]


def tensorwise_into(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> None:
    if tuple(q.shape) != tuple(x.shape):
        raise ValueError(f"q must have the shape {tuple(x.shape)}")
    require_one_element("scale", scale)
    source, rows, width, stride = tensorwise_source(x) if x.numel() else (x, 0, 0, 0)
    codes = codes_target(q, (x,))
    scales = scale if scale.data_ptr() % launcher.WORD_BYTES == 0 and writes_in_place(scale, (x,)) else \
        torch.empty((), dtype=torch.float32, device=scale.device)
    amax = torch.zeros((), dtype=torch.int32, device=x.device)
    launcher.launch_tensorwise(ELEMENTS[x.dtype], source.data_ptr(), stride, codes.data_ptr(), rows, width,
                               scales.data_ptr(), amax.data_ptr(), current_stream_handle(x.get_device()))
    copy_back(codes, q)
    if scales is not scale:
        scale.copy_(scales.reshape(scale.shape))


QUANTIZE_FP8_TENSORWISE = define(
    "quantize_fp8_tensorwise", run=tensorwise_into, check=check_tensorwise, check_sizes=check_tensorwise_sizes,
    outputs={"q": Output(tensorwise_codes_like, FP8, "out"), "scale": Output(scalar_like, torch.float32, "scale_out")})
quantize_fp8_tensorwise_op = QUANTIZE_FP8_TENSORWISE.functional
quantize_fp8_tensorwise_out_op = QUANTIZE_FP8_TENSORWISE.out


def check_static(x: torch.Tensor, scale: torch.Tensor):
    require_dtype("x", x, INPUT_DTYPES)
    require_dtype("scale", scale, (torch.float32,))


def check_static_sizes(x, scale):
    check_tensorwise_sizes(x)
    require_one_element("scale", scale)


def static_codes_like(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=FP8)


def static_into(x: torch.Tensor, scale: torch.Tensor, q: torch.Tensor) -> None:
    if tuple(q.shape) != tuple(x.shape):
        raise ValueError(f"q must have the shape {tuple(x.shape)}")
    if x.numel() == 0:
        return
    source, rows, width, stride = tensorwise_source(x)
    codes = codes_target(q, (x, scale))
    if scale.data_ptr() % launcher.WORD_BYTES:
        scale = scale.clone()
    launcher.launch_tensorwise(ELEMENTS[x.dtype], source.data_ptr(), stride, codes.data_ptr(), rows, width,
                               scale.data_ptr(), 0, current_stream_handle(x.get_device()))
    copy_back(codes, q)


QUANTIZE_FP8_STATIC = define("quantize_fp8_static", run=static_into, check=check_static,
                             check_sizes=check_static_sizes, outputs={"q": Output(static_codes_like, FP8, "out")})
quantize_fp8_static_op, quantize_fp8_static_out_op = QUANTIZE_FP8_STATIC.functional, QUANTIZE_FP8_STATIC.out


def quantize_fp8_tensorwise(x, scale=None, *, out=None, scale_out=None):
    """Quantizes `x` to E4M3 with one FP32 scale, computed from `x` or given as `scale`, and returns `(q, scale)`. See
    the README sections "Quantizers" and "Custom ops"."""
    if scale is None:
        return QUANTIZE_FP8_TENSORWISE(x, outputs={"out": out, "scale_out": scale_out})
    require_tensorwise_scale("scale", scale)
    if scale_out is not None:
        raise ValueError("scale_out must be None when the call gets a scale")
    return QUANTIZE_FP8_STATIC(x, scale, outputs={"out": out}), scale.reshape(())
