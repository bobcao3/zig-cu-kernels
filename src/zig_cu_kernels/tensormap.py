import functools
import struct

import torch
from cuda.bindings import driver

CACHE_LIMIT = 1024
RANKS = (2, 3)
DATA_TYPES = {
    torch.float8_e4m3fn: driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint8: driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.bfloat16: driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float32: driver.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
}
SWIZZLES = {
    0: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: driver.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}
L2_PROMOTIONS = {
    0: driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
    64: driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
    128: driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    256: driver.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
}
OPAQUE = struct.Struct("<16Q")


def tensor_map(tensor, box, swizzle=128, l2_promotion=256):
    """Returns the 128 bytes of a tiled `CUtensorMap` of the 2-D or 3-D `tensor`, whose last dimension has stride 1.
    `box` gives the extent of a tile in each dimension of `tensor`, in the order of `tensor.shape`. `swizzle` and
    `l2_promotion` are sizes in bytes. The driver encodes a map only on devices of compute capability 9.0 or higher."""
    return encoded(tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tuple(box), swizzle,
                   l2_promotion)


@functools.lru_cache(maxsize=CACHE_LIMIT)
def encoded(pointer, shape, strides, dtype, box, swizzle, l2_promotion):
    if len(shape) not in RANKS or len(box) != len(shape):
        raise ValueError(f"a tensor map needs a tensor of rank 2 or 3 and a box of the same rank, got the shape "
                         f"{shape} and the box {box}")
    if dtype not in DATA_TYPES:
        raise ValueError(f"no tensor map data type holds {dtype}; use one of {', '.join(map(str, DATA_TYPES))}")
    if strides[-1] != 1:
        raise ValueError(f"the last dimension of a tensor map must have stride 1, got the strides {strides}")
    element_bytes = torch.empty((), dtype=dtype).element_size()
    status, encoding = driver.cuTensorMapEncodeTiled(
        DATA_TYPES[dtype],
        len(shape),
        pointer,
        [driver.cuuint64_t(extent) for extent in reversed(shape)],
        [driver.cuuint64_t(stride * element_bytes) for stride in reversed(strides[:-1])],
        [driver.cuuint32_t(extent) for extent in reversed(box)],
        [driver.cuuint32_t(1)] * len(shape),
        driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        SWIZZLES[swizzle],
        L2_PROMOTIONS[l2_promotion],
        driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if status != driver.CUresult.CUDA_SUCCESS:
        raise ValueError(f"cuTensorMapEncodeTiled failed with {status} for the shape {shape}, strides {strides}, "
                         f"{dtype}, box {box} and swizzle {swizzle}")
    return OPAQUE.pack(*(int(word) for word in encoding.opaque))
