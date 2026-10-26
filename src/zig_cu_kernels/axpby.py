import functools
import math

from . import runtime
from .runtime import fp32
from .tensors import is_aligned, is_contiguous, require, require_device, require_dtype, view_of

SIGNATURE = "pppqff"
ELEMENT_BYTES = {"bf16": 2, "f32": 4}
DTYPE_NAMES = {"bf16": "bfloat16", "f32": "float32"}
CHUNK_BYTES = 16
MAX_PLANS = 4096
# A call whose tensors exceed the L2 cache lets one SM hold at most this many work groups, by the operands it reads.
RESIDENT_WORK_GROUPS = {1: 4, 2: 1}


@functools.lru_cache(maxsize=MAX_PLANS)
def plan(device_id, element, count):
    kernel, info = runtime.load_for(device_id, f"axpby_{element}")
    tensor_bytes = count * ELEMENT_BYTES[element]
    work_groups = max(1, -(-(tensor_bytes // CHUNK_BYTES) // (info.threads * info.chunks_per_thread)))
    shared_bytes = {(operands, tensors): capped_shared_bytes(kernel, device_id, operands, tensors * tensor_bytes)
                    for operands in range(3) for tensors in range(1, operands + 2)}
    return Launch(runtime.kernel_call(kernel, SIGNATURE), (work_groups, 1, 1), info.threads, count, shared_bytes)


def capped_shared_bytes(kernel, device_id, operands, byte_count):
    if byte_count <= runtime.l2_bytes(device_id):
        return 0
    return runtime.shared_bytes_for(kernel, device_id, RESIDENT_WORK_GROUPS[max(1, operands)])


class Launch:

    def __init__(self, kernel_call, grid, threads, count, shared_bytes):
        self.kernel_call = kernel_call
        self.grid = grid
        self.threads = threads
        self.count = count
        self.shared_bytes = shared_bytes

    def shared_bytes_of(self, pointers, alpha, beta):
        x, y, out = pointers
        read = [pointer for pointer, coefficient in ((x, alpha), (y, beta)) if coefficient != 0]
        return self.shared_bytes[len(read), len({out, *read})]

    def launch(self, pointers, alpha, beta, stream):
        shared_bytes = self.shared_bytes_of(pointers, alpha, beta)
        self.kernel_call(self.grid, self.threads, shared_bytes, (*pointers, self.count, alpha, beta), stream)


def axpby(alpha, x, beta, y, out, *, element, stream=None):
    """Queues `out = alpha * x + beta * y` on `stream`. `element` is "bf16" or "f32", the dtype of `x`, `y` and
    `out` (README sections "axpby" and "Launchers")."""
    alpha, beta = fp32("alpha", alpha), fp32("beta", beta)
    require(element in ELEMENT_BYTES, f"element must be one of {', '.join(ELEMENT_BYTES)}, got {element}")
    names, operands = ("x", "y", "out"), (x, y, out)
    views = [view_of(tensor, stream, name) for tensor, name in zip(operands, names)]
    count = math.prod(views[2].shape)
    for name, view in zip(names, views):
        require_dtype(view, name, DTYPE_NAMES[element])
        require(math.prod(view.shape) == count, f"{name} must have {count} elements")
    if not count:
        return
    device_id = runtime.current_device().device_id
    for name, operand, view, read in zip(names, operands, views, (alpha != 0, beta != 0, True)):
        require_device(name, operand, view.pointer, device_id)
        if read:
            require(is_contiguous(view) and is_aligned(view), f"{name} must be C-contiguous and 16-byte aligned")
        if read and name != "out":
            gap = abs(view.pointer - views[2].pointer)
            require(gap == 0 or gap >= count * ELEMENT_BYTES[element], f"out overlaps {name} without being {name}")
    pointers = tuple(view.pointer for view in views)
    plan(device_id, element, count).launch(pointers, alpha, beta, runtime.stream_handle(stream))
