import functools
import math
from typing import Optional

import torch

from .. import gemm as launcher
from ..tensors import ALIGNMENT_BYTES
from .common import current_stream_handle, leading_rows, require_choice, require_dtype, workspace_for, writes_in_place
from .ops import Output, define

__all__ = ["gemm", "gated_gemm", "gemm_op", "gemm_out_op", "gated_gemm_op", "gated_gemm_out_op"]

BF16 = (torch.bfloat16,)
OUT_DTYPES = (torch.bfloat16, torch.float32)


def require_dims(name, tensor, dims):
    if tensor is not None and tensor.dim() not in dims:
        raise ValueError(f"{name} must have {' or '.join(map(str, dims))} dimensions, got {tensor.dim()}")


def require_rows(name, tensor):
    if tensor.dim() < 1:
        raise ValueError(f"{name} must have at least one dimension")


def require_shape(name, tensor, shape):
    if tensor is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have the shape {tuple(shape)}, got {tuple(tensor.shape)}")


def require_product_sizes(a, columns, column_name):
    k = a.shape[-1]
    if k % 8 or columns % 8:
        raise ValueError(f"k and {column_name} must be multiples of 8, got k = {k} and {column_name} = {columns}")
    if max(math.prod(a.shape[:-1]), columns, k) > launcher.MAX_INDEX:
        raise ValueError(f"the row count, {column_name} and k must lie below 2^31")


def contiguous_row_stride(tensor):
    if tensor.numel() == 0 or tensor.numel() == tensor.shape[-1]:
        return 0
    return tensor.shape[-1]


def ready_row_stride(tensor):
    """Returns the row stride in elements that the launchers of `zig_cu_kernels.gemm` pass for `tensor`, 0 for a tensor
    without elements or with one row, or None when the kernels cannot use `tensor` in place."""
    if tensor.numel() == 0:
        return 0
    if tensor.data_ptr() % ALIGNMENT_BYTES:
        return None
    if tensor.is_contiguous():
        stride = contiguous_row_stride(tensor)
    else:
        rows = leading_rows(tensor)
        if rows is None or (tensor.shape[-1] > 1 and tensor.stride(-1) != 1):
            return None
        stride = 0 if rows[0] == 1 else rows[1]
    if stride and (stride * tensor.element_size() % ALIGNMENT_BYTES or stride > launcher.MAX_INDEX):
        return None
    return stride


def matrix_ready(tensor):
    if tensor is None:
        return None, 0
    stride = ready_row_stride(tensor)
    if stride is None:
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        stride = contiguous_row_stride(tensor)
    return tensor, stride


def vector_ready(tensor):
    if tensor is None or ((tensor.stride(0) == 1 or tensor.numel() <= 1) and tensor.data_ptr() % ALIGNMENT_BYTES == 0):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def output_target(out, inputs, residual):
    stride = ready_row_stride(out)
    if stride is not None and writes_in_place(out, inputs, (residual,)):
        return out, stride
    target = torch.empty_like(out, memory_format=torch.contiguous_format)
    return target, contiguous_row_stride(target)


def pointer_of(tensor):
    return 0 if tensor is None else tensor.data_ptr()


@functools.lru_cache(maxsize=4096)
def launch_for(kind, m, columns, k, sizes, activation, device_index):
    return launcher.Launch(kind, launcher.plan(device_index, kind, m, columns, k), sizes, activation, device_index)


def run(kind, m, columns, k, sizes, activation, pointers, device_index):
    """Launches the kernel `kind` on the current stream, with the workspace of the stream when the plan splits k. The
    caller passes operands that `matrix_ready`, `vector_ready` and `output_target` returned."""
    launch = launch_for(kind, m, columns, k, sizes, activation, device_index)
    stream = current_stream_handle(device_index)
    if not launch.partial_bytes:
        launch(pointers, launcher.NO_BUFFERS, stream)
        return
    with workspace_for(device_index, stream) as workspace:
        launch(pointers, workspace.addresses(launch.partial_bytes, launch.counter_count), stream)


def gemm_output(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], residual: Optional[torch.Tensor],
                activation: str, out_dtype: torch.dtype) -> torch.Tensor:
    return a.new_empty((*a.shape[:-1], b.shape[0]), dtype=out_dtype)


def check_gemm(a, b, bias, residual, activation, out_dtype):
    for name, tensor in (("a", a), ("b", b), ("bias", bias), ("residual", residual)):
        require_dtype(name, tensor, BF16)
    require_choice("activation", activation, launcher.ACTIVATIONS)
    require_choice("out_dtype", out_dtype, OUT_DTYPES)
    if out_dtype == torch.float32 and (activation != "none" or residual is not None):
        raise ValueError("an FP32 output takes no activation and no residual")
    require_rows("a", a)
    require_dims("b", b, (2,))
    require_dims("bias", bias, (1,))
    require_dims("residual", residual, (a.dim(),))


def check_gemm_sizes(a, b, bias, residual, activation, out_dtype):
    if b.shape[1] != a.shape[-1]:
        raise ValueError(f"b must be [n, {a.shape[-1]}] for a of the shape {tuple(a.shape)}, got {tuple(b.shape)}")
    require_product_sizes(a, b.shape[0], "n")
    require_shape("bias", bias, (b.shape[0],))
    require_shape("residual", residual, (*a.shape[:-1], b.shape[0]))


def gemm_into(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], residual: Optional[torch.Tensor],
              activation: str, out: torch.Tensor) -> None:
    require_shape("out", out, (*a.shape[:-1], b.shape[0]))
    if out.numel() == 0:
        return
    (a, a_stride), (b, b_stride), (residual, residual_stride) = map(matrix_ready, (a, b, residual))
    bias = vector_ready(bias)
    target, target_stride = output_target(out, (a, b, bias), residual)
    m, n, k = math.prod(a.shape[:-1]), b.shape[0], a.shape[-1]
    strides = (a_stride, b_stride, target_stride)
    pointers = (a.data_ptr(), b.data_ptr(), target.data_ptr(), pointer_of(bias))
    if target.dtype == torch.float32:
        run("store_f32", m, n, k, (*strides, m, n, k), (), pointers, a.get_device())
    else:
        run("store_bf16", m, n, k, (*strides, residual_stride, m, n, k), (launcher.activation_index(activation),),
            (*pointers, pointer_of(residual)), a.get_device())
    if target is not out:
        out.copy_(target)


GEMM = define("gemm", run=gemm_into, outputs={"out": Output(gemm_output, "out_dtype")}, check=check_gemm,
              check_sizes=check_gemm_sizes)
gemm_op, gemm_out_op = GEMM.functional, GEMM.out


def gemm(a: torch.Tensor, b: torch.Tensor, *, bias: Optional[torch.Tensor] = None, activation: str = "none",
         residual: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None,
         out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Returns `F.linear(a, b, bias)` followed by `activation` and the addition of `residual`, for BF16 `a` and `b`.
    See the README section "BF16 GEMM"."""
    return GEMM(a, b, bias, residual, activation, out_dtype, outputs={"out": out})


def gated_output(a: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, bias_gate: Optional[torch.Tensor],
                 bias_up: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str) -> torch.Tensor:
    return a.new_empty((*a.shape[:-1], w_gate.shape[0]), dtype=torch.bfloat16)


def check_gated(a, w_gate, w_up, bias_gate, bias_up, residual, activation):
    tensors = (("a", a), ("w_gate", w_gate), ("w_up", w_up), ("bias_gate", bias_gate), ("bias_up", bias_up),
               ("residual", residual))
    for name, tensor in tensors:
        require_dtype(name, tensor, BF16)
    require_choice("activation", activation, launcher.ACTIVATIONS)
    require_rows("a", a)
    require_dims("w_gate", w_gate, (2,))
    require_dims("w_up", w_up, (2,))
    require_dims("bias_gate", bias_gate, (1,))
    require_dims("bias_up", bias_up, (1,))
    require_dims("residual", residual, (a.dim(),))


def check_gated_sizes(a, w_gate, w_up, bias_gate, bias_up, residual, activation):
    f = w_gate.shape[0]
    if w_gate.shape[1] != a.shape[-1]:
        raise ValueError(f"w_gate must be [f, {a.shape[-1]}] for a of the shape {tuple(a.shape)}, got "
                         f"{tuple(w_gate.shape)}")
    require_shape("w_up", w_up, w_gate.shape)
    require_product_sizes(a, f, "f")
    require_shape("bias_gate", bias_gate, (f,))
    require_shape("bias_up", bias_up, (f,))
    require_shape("residual", residual, (*a.shape[:-1], f))


def gated_into(a: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, bias_gate: Optional[torch.Tensor],
               bias_up: Optional[torch.Tensor], residual: Optional[torch.Tensor], activation: str,
               out: torch.Tensor) -> None:
    require_shape("out", out, (*a.shape[:-1], w_gate.shape[0]))
    if out.numel() == 0:
        return
    (w_gate, weight_stride), (w_up, _) = matrix_ready(w_gate), matrix_ready(w_up)
    if w_gate.stride(0) != w_up.stride(0):
        w_gate, w_up = w_gate.contiguous(), w_up.contiguous()
        weight_stride = contiguous_row_stride(w_gate)
    (a, a_stride), (residual, residual_stride) = matrix_ready(a), matrix_ready(residual)
    bias_gate, bias_up = vector_ready(bias_gate), vector_ready(bias_up)
    target, target_stride = output_target(out, (a, w_gate, w_up, bias_gate, bias_up), residual)
    m, f, k = math.prod(a.shape[:-1]), w_gate.shape[0], a.shape[-1]
    sizes = (a_stride, weight_stride, target_stride, residual_stride, m, f, k)
    pointers = (a.data_ptr(), w_gate.data_ptr(), w_up.data_ptr(), target.data_ptr(),
                *map(pointer_of, (bias_gate, bias_up, residual)))
    run("gated", m, f, k, sizes, (launcher.activation_index(activation),), pointers, a.get_device())
    if target is not out:
        out.copy_(target)


GATED_GEMM = define("gated_gemm", run=gated_into, outputs={"out": Output(gated_output, torch.bfloat16)},
                    check=check_gated, check_sizes=check_gated_sizes)
gated_gemm_op, gated_gemm_out_op = GATED_GEMM.functional, GATED_GEMM.out


def gated_gemm(a: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, *, bias_gate: Optional[torch.Tensor] = None,
               bias_up: Optional[torch.Tensor] = None, activation: str = "silu",
               residual: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Returns the BF16 gated linear unit of `a` with the weights `w_gate` and `w_up`, plus `residual`. See the README
    section "BF16 GEMM"."""
    return GATED_GEMM(a, w_gate, w_up, bias_gate, bias_up, residual, activation, outputs={"out": out})
