import math
import threading

import torch
from cuda.bindings import driver

from ..tensors import ALIGNMENT_BYTES

ELEMENTS = {torch.bfloat16: "bf16", torch.float32: "f32"}
SCALE_DTYPES = (torch.float32, torch.float8_e8m0fnu)
SCALE_GROUP = 128
FP32_OVERFLOW = 2**128 - 2**103
FP32_SIGNIFICAND_BITS = 24
EAGER_WORKSPACES = 16
eager_workspaces = {}
workspaces_lock = threading.Lock()


def current_stream_handle(device_index=None):
    # The raw handle avoids the Python stream object that torch.cuda.current_stream() builds on every call.
    return torch._C._cuda_getCurrentRawStream(torch._C._cuda_getDevice() if device_index is None else device_index)


def is_kernel_ready(tensor):
    return tensor.is_contiguous() and tensor.data_ptr() % ALIGNMENT_BYTES == 0


def kernel_ready(tensor):
    return tensor if is_kernel_ready(tensor) else tensor.clone(memory_format=torch.contiguous_format)


def device_index_of(arguments, named_positions):
    """Returns the CUDA device index of every tensor of `arguments` at `named_positions`, or raises ValueError."""
    first_name, first_position = named_positions[0]
    first = arguments[first_position]
    if not first.is_cuda:
        raise ValueError(f"{first_name} must be a CUDA tensor, got a tensor on {first.device}")
    device_index = first.get_device()
    for name, position in named_positions:
        tensor = arguments[position]
        if tensor is not None and (not tensor.is_cuda or tensor.get_device() != device_index):
            raise ValueError(f"{name} must be on cuda:{device_index}, the device of {first_name}, got {tensor.device}")
    return device_index


def require_same_device(names, tensors):
    device = tensors[0].device
    for name, tensor in zip(names, tensors):
        if tensor is not None and tensor.device != device:
            raise ValueError(f"{name} must be on {device}, the device of {names[0]}, got {tensor.device}")


def require_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")


def require_dtype(name, tensor, dtypes):
    if tensor is not None and tensor.dtype not in dtypes:
        raise ValueError(f"{name} must be a {' or '.join(map(str, dtypes))} tensor, got {tensor.dtype}")


def require_choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(map(str, choices))}, got {value}")


def require_one_element(name, tensor):
    if tensor.numel() != 1:
        raise ValueError(f"{name} must hold one element, got the shape {tuple(tensor.shape)}")


def rounded_to_fp32(value):
    magnitude = abs(value)
    shift = magnitude.bit_length() - FP32_SIGNIFICAND_BITS
    if shift <= 0:
        return float(value)
    kept, dropped = magnitude >> shift, magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    kept += dropped > half or (dropped == half and kept & 1)
    return math.copysign(kept << shift, value)


def fp32_scalar(name, value):
    """Returns the Python number `value` as a float that rounds to the same FP32 value as `value`."""
    if type(value) is float:
        if abs(value) >= FP32_OVERFLOW and not math.isinf(value):
            raise OverflowError(f"{name} = {value} rounds to infinity in FP32")
        return value
    if isinstance(value, (torch.SymFloat, torch.SymInt)):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a Python number, got {type(value).__name__}")
    try:
        number = rounded_to_fp32(value) if isinstance(value, int) else float(value)
    except OverflowError:
        number = math.inf
    if abs(number) >= FP32_OVERFLOW and not (isinstance(value, float) and math.isinf(value)):
        raise OverflowError(f"{name} = {value} rounds to infinity in FP32")
    return number


def require_int(name, value):
    if type(value) is int or isinstance(value, torch.SymInt):
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    return value


def require_scale_dtype(name, dtype, scaling):
    require_choice(name, dtype, SCALE_DTYPES if scaling == "block128" else (torch.float32,))


def require_tensorwise_scale(name, scale):
    require_tensor(name, scale)
    require_dtype(name, scale, (torch.float32,))
    require_one_element(name, scale)


def activation_block_scales(x, dtype):
    return x.new_empty((x.shape[-1] // SCALE_GROUP, math.prod(x.shape[:-1])), dtype=dtype).t()


def has_scale_shape(scale, lead, scale_groups):
    return tuple(scale.shape) in ((math.prod(lead), scale_groups), (*lead, scale_groups))


def scale_layout(name, scale, lead, scale_groups, dtypes=SCALE_DTYPES):
    """Returns the row stride and the stride between scale groups of `scale`. The function returns None when the rows
    have no single stride, or for E8M0 storage at an address that is not a multiple of 4."""
    require_dtype(name, scale, dtypes)
    if not has_scale_shape(scale, lead, scale_groups):
        shapes = f"{(math.prod(lead), scale_groups)} or {(*lead, scale_groups)}"
        raise ValueError(f"{name} must have the shape {shapes}, got {tuple(scale.shape)}")
    rows = leading_rows(scale)
    if rows is None or (scale.dtype == torch.float8_e8m0fnu and scale.untyped_storage().data_ptr() % 4 != 0):
        return None
    return rows[1], scale.stride(-1)


def byte_span(tensor):
    if tensor.is_contiguous():
        return tensor.numel() * tensor.element_size()
    span = 1
    for size, stride in zip(tensor.shape, tensor.stride()):
        span += (size - 1) * stride
    return span * tensor.element_size()


def byte_range(tensor):
    count = tensor.numel()
    if count == 0:
        return None
    start = tensor.data_ptr()
    if tensor.is_contiguous():
        return start, start + count * tensor.element_size()
    return start, start + byte_span(tensor)


def overlaps(first, second):
    first_range = byte_range(first)
    return first_range is not None and overlaps_range(second, *first_range)


def overlaps_range(tensor, start, end):
    tensor_range = byte_range(tensor)
    return tensor_range is not None and tensor_range[0] < end and start < tensor_range[1]


def is_exact_alias(first, second):
    return (first.data_ptr() == second.data_ptr() and first.dtype == second.dtype and first.shape == second.shape
            and first.stride() == second.stride())


def leading_rows(tensor):
    """Returns the row count and row stride of the leading dimensions, or None when no single row stride exists."""
    shape = tensor.shape
    if shape and tensor.is_contiguous():
        count = tensor.numel()
        if count:
            rows = count // shape[-1]
            return (rows, shape[-1]) if rows > 1 else (1, 0)
    strides = tensor.stride()
    rows, row_stride = 1, 0
    for axis in range(len(shape) - 2, -1, -1):
        size = shape[axis]
        if size == 1:
            continue
        if rows == 1:
            row_stride = strides[axis]
        elif strides[axis] != row_stride * rows:
            return None
        rows *= size
    return rows, row_stride


def row_layout(tensor):
    """Returns the row count, row length and row stride in bytes if the elements form contiguous rows of one stride."""
    element_bytes = tensor.element_size()
    if tensor.dim() == 0:
        return 1, element_bytes, 0
    rows = leading_rows(tensor)
    if rows is None or (tensor.shape[-1] > 1 and tensor.stride(-1) != 1):
        return None
    return rows[0], tensor.shape[-1] * element_bytes, rows[1] * element_bytes


def shares_memory(first, second):
    """Returns whether a byte lies in an element of each tensor. The answer is exact for two tensors of contiguous rows
    with equal row strides or with a single row in one tensor, and otherwise tells whether the byte ranges overlap."""
    if not overlaps(first, second):
        return False
    first_rows, second_rows = row_layout(first), row_layout(second)
    if first_rows is None or second_rows is None:
        return True
    (rows, width, stride), (other_rows, other_width, other_stride) = first_rows, second_rows
    if rows > 1 and other_rows > 1 and stride != other_stride:
        return True
    stride = stride if rows > 1 else other_stride
    if stride <= 0:
        return True
    offset = second.data_ptr() - first.data_ptr()
    lowest = (-other_width - offset) // stride + 1
    highest = -((offset - width) // stride) - 1
    return max(lowest, 1 - rows) <= min(highest, other_rows - 1)


def shares_memory_within(tensor):
    """Returns whether two elements of `tensor` share memory. The answer is exact for at most two dimensions of size
    above 1 or for strides that each exceed the span of the dimensions with smaller strides, and True otherwise."""
    if tensor.is_contiguous():
        return False
    dimensions = sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride()) if size > 1)
    span = 0
    for stride, size in dimensions:
        if stride <= span:
            break
        span += (size - 1) * stride
    else:
        return False
    if len(dimensions) != 2 or dimensions[0][0] == 0:
        return True
    (stride, size), (other_stride, other_size) = dimensions
    divisor = math.gcd(stride, other_stride)
    return other_stride // divisor < size and stride // divisor < other_size


def writes_in_place(output, inputs, aliases=()):
    """Returns whether `output` overlaps no tensor of `inputs` and no tensor of `aliases` except an exact alias."""
    output_range = byte_range(output)
    if output_range is None:
        return True
    for tensor in inputs:
        if tensor is not None and overlaps_range(tensor, *output_range):
            return False
    for tensor in aliases:
        if tensor is not None and overlaps_range(tensor, *output_range) and not is_exact_alias(output, tensor):
            return False
    return True


def require_separate_outputs(names, outputs):
    for index, output in enumerate(outputs):
        if output is None:
            continue
        if shares_memory_within(output):
            raise ValueError(f"the elements of {names[index]} share memory")
        for other_index in range(index + 1, len(outputs)):
            other = outputs[other_index]
            if other is not None and shares_memory(output, other):
                raise ValueError(f"{names[index]} and {names[other_index]} share memory")


class PartialSumTensors:
    """A thread holds the workspace in a `with` block until the thread has queued every launch with the addresses."""

    def __init__(self, device_index, own_pools=False):
        self.lock = threading.Lock()
        self.owner = None
        self.device_index = device_index
        self.own_pools = own_pools
        self.parts = torch.empty(0, dtype=torch.uint8, device=device_index)
        self.counters = torch.empty(0, dtype=torch.int32, device=device_index)

    def __enter__(self):
        self.lock.acquire()
        self.owner = threading.get_ident()
        return self

    def __exit__(self, *exception):
        self.owner = None
        self.lock.release()

    def addresses(self, part_bytes, counter_count):
        if self.owner != threading.get_ident():
            raise RuntimeError("a thread must hold the workspace while it uses the addresses")
        if self.parts.numel() < part_bytes or self.counters.numel() < counter_count:
            if not self.own_pools:
                self.grow(part_bytes, counter_count)
            else:
                with torch.cuda.device(self.device_index):
                    pool = torch.cuda.MemPool()
                with torch.cuda.use_mem_pool(pool, self.device_index):
                    self.grow(part_bytes, counter_count)
        return self.parts.data_ptr(), self.counters.data_ptr()

    def grow(self, part_bytes, counter_count):
        if self.parts.numel() < part_bytes:
            size = max(part_bytes, 2 * self.parts.numel())
            self.parts = torch.empty(size, dtype=torch.uint8, device=self.device_index)
        if self.counters.numel() < counter_count:
            size = max(counter_count, 2 * self.counters.numel())
            self.counters = torch.zeros(size, dtype=torch.int32, device=self.device_index)


def workspace_for(device_index, stream):
    """Returns the workspace for a call on `stream`. During a CUDA graph capture, each call gets a new workspace in the
    pool of the graph. Eager calls on one stream share a workspace that allocates each growth in a new memory pool,
    outside any pool to which PyTorch routes the thread, such as that of `torch.compile(mode="reduce-overhead")`."""
    status, capturing = driver.cuStreamIsCapturing(stream)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuStreamIsCapturing failed: {status}")
    if capturing != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
        return PartialSumTensors(device_index)
    key = device_index, stream
    with workspaces_lock:
        workspace = eager_workspaces.get(key)
        if workspace is None:
            if len(eager_workspaces) >= EAGER_WORKSPACES:
                eager_workspaces.pop(next(iter(eager_workspaces)))
            workspace = eager_workspaces[key] = PartialSumTensors(device_index, own_pools=True)
        return workspace
