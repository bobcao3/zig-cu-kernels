import torch

from .. import axpby as launcher
from .. import runtime
from .common import (
    ELEMENTS,
    current_stream_handle,
    fp32_scalar,
    is_kernel_ready,
    kernel_ready,
    require_dtype,
    writes_in_place,
)
from .ops import Output, define

__all__ = ["axpby", "axpby_", "axpby_op", "axpby_out_op"]


def check_axpby(alpha: float, x: torch.Tensor, beta: float, y: torch.Tensor):
    fp32_scalar("alpha", alpha)
    fp32_scalar("beta", beta)
    require_dtype("x", x, tuple(ELEMENTS))
    require_dtype("y", y, (x.dtype,))


def check_axpby_launch(alpha: float, x: torch.Tensor, beta: float, y: torch.Tensor):
    if y.shape != x.shape:
        raise ValueError(f"y must have the shape {tuple(x.shape)} of x, got {tuple(y.shape)}")
    runtime.load_for(x.get_device(), f"axpby_{ELEMENTS[x.dtype]}")


def axpby_like(alpha: float, x: torch.Tensor, beta: float, y: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)


def axpby_into(alpha: float, x: torch.Tensor, beta: float, y: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != x.shape:
        raise ValueError(f"out must have the shape {tuple(x.shape)} of x, got {tuple(out.shape)}")
    if out.numel() == 0:
        return
    alpha, beta = launcher.fp32("alpha", alpha), launcher.fp32("beta", beta)
    operands = [kernel_ready(tensor) if coefficient != 0 else tensor for tensor, coefficient in ((x, alpha), (y, beta))]
    read = [operand for operand, coefficient in zip(operands, (alpha, beta)) if coefficient != 0]
    in_place = is_kernel_ready(out) and writes_in_place(out, (), read)
    target = out if in_place else torch.empty_like(out, memory_format=torch.contiguous_format)
    device_index = x.get_device()
    pointers = (operands[0].data_ptr(), operands[1].data_ptr(), target.data_ptr())
    launch = launcher.plan(device_index, ELEMENTS[x.dtype], out.numel())
    launch.launch(pointers, alpha, beta, current_stream_handle(device_index))
    if not in_place:
        out.copy_(target)


AXPBY = define("axpby", run=axpby_into, outputs={"out": Output(axpby_like, "x")}, check=check_axpby,
               check_sizes=check_axpby_launch)
axpby_op, axpby_out_op = AXPBY.functional, AXPBY.out


def axpby(alpha: float, x: torch.Tensor, beta: float, y: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    """Returns `alpha * x + beta * y` in a new contiguous tensor or in `out` (README sections "axpby" and
    "Scalar arguments")."""
    return AXPBY(alpha, x, beta, y, outputs={"out": out})


def axpby_(alpha: float, x: torch.Tensor, beta: float, y: torch.Tensor) -> torch.Tensor:
    """Sets `y` to `axpby(alpha, x, beta, y)` and returns `y`."""
    return AXPBY(alpha, x, beta, y, outputs={"out": y})
