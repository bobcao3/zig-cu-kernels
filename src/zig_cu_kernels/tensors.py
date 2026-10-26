import functools
from collections import namedtuple

from cuda.bindings import driver

ALIGNMENT_BYTES = 16
# DLPack forbids the stream 0 of the driver API and names the legacy default stream 1.
LEGACY_DEFAULT_STREAM = 1
DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "float8_e4m3fn": 1, "float8_e8m0fnu": 1, "int32": 4, "uint32": 4}
DEVICE_ORDINAL = driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
MEMORY_ATTRIBUTES = [driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, DEVICE_ORDINAL]

View = namedtuple("View", "pointer shape strides itemsize dtype", defaults=(None,))


@functools.cache
def dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def view_of(tensor, stream=-1, name="the operand"):
    """Returns the view of a PyTorch tensor or of an object that exports DLPack or the CUDA Array Interface. The
    function passes `stream` to the producer of an exported object for the ordering of README section "Launchers",
    and the value -1 skips that ordering."""
    if hasattr(tensor, "data_ptr"):
        # `tuple.__new__` skips the constructor that `namedtuple` generates in Python.
        return tuple.__new__(View, (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.element_size(),
                                    dtype_name(tensor.dtype)))
    from cuda.core.utils import StridedMemoryView

    try:
        memory = StridedMemoryView.from_any_interface(tensor, stream_ptr=producer_stream(stream))
    except Exception as error:
        if exports_pageable_memory(tensor):
            raise ValueError(f"{name} must be in device memory, not in pageable host memory") from error
        raise
    strides = memory.strides or contiguous_strides(memory.shape)
    dtype = memory.dtype
    return View(memory.ptr, tuple(memory.shape), tuple(strides), dtype.itemsize,
                None if dtype.kind == "V" else dtype.name)


def exports_pageable_memory(exported):
    interface = getattr(exported, "__cuda_array_interface__", None)
    if interface is None or hasattr(exported, "__dlpack__"):
        return False
    address = interface["data"][0]
    return address != 0 and pointer_device(address) < 0


def pointer_device(address):
    status, ordinal = driver.cuPointerGetAttribute(DEVICE_ORDINAL, address)
    return ordinal if status == driver.CUresult.CUDA_SUCCESS else -1


def memory_device(address):
    """The driver reports a device ordinal for pinned host memory too, so the function also checks the memory type."""
    status, values = driver.cuPointerGetAttributes(len(MEMORY_ATTRIBUTES), MEMORY_ATTRIBUTES, address)
    if status != driver.CUresult.CUDA_SUCCESS or values[0] != driver.CUmemorytype.CU_MEMORYTYPE_DEVICE:
        return -1
    return values[1]


def device_of(operand, pointer=None, name="the operand"):
    """Returns the device of `operand`, -1 outside device memory, or None for an exported object without elements."""
    if hasattr(operand, "data_ptr"):
        return operand.get_device()
    if pointer is None:
        view = view_of(operand, name=name)
        if 0 in view.shape:
            return None
        pointer = view.pointer
    return memory_device(pointer)


def require_device(name, operand, pointer, device_id):
    found = device_of(operand, pointer, name)
    if found is not None and found != device_id:
        place = "in host memory" if found < 0 else f"on device {found}"
        raise ValueError(f"{name} must be on device {device_id}, the current device, not {place}")


def producer_stream(stream):
    if isinstance(stream, int) and stream == -1:
        return -1
    from .runtime import stream_handle

    return stream_handle(stream) or LEGACY_DEFAULT_STREAM


def require_dtype(view, name, *dtypes):
    if view.dtype in dtypes:
        return view.dtype
    if view.dtype is None:
        match = next((dtype for dtype in dtypes if DTYPE_BYTES[dtype] == view.itemsize), None)
        if match is not None:
            return match
    found = f", not {view.dtype}" if view.dtype else f", not {view.itemsize}-byte values"
    raise ValueError(f"{name} must hold {' or '.join(dtypes)} values{found}")


def contiguous_strides(shape):
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def is_contiguous(view):
    if 0 in view.shape:
        return True
    expected = 1
    for size, stride in zip(reversed(view.shape), reversed(view.strides)):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def is_aligned(view):
    return view.pointer % ALIGNMENT_BYTES == 0


def rows_view(view):
    shape, strides = view.shape, view.strides
    if len(shape) == 0:
        raise ValueError("the tensor needs at least one dimension")
    width = shape[-1]
    if width > 1 and strides[-1] != 1:
        raise ValueError("the last dimension must be contiguous")
    if 0 in shape[:-1]:
        return 0, width, width
    rows, stride = 1, width
    for axis in range(len(shape) - 2, -1, -1):
        size = shape[axis]
        if size == 1:
            continue
        if rows == 1:
            stride = strides[axis]
        elif strides[axis] != stride * rows:
            raise ValueError("the leading dimensions must flatten into rows of one stride")
        rows *= size
    return rows, width, stride


def require(condition, message):
    if not condition:
        raise ValueError(message)
