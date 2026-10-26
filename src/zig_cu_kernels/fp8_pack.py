"""Launchers of `zig/fp8_pack.zig`, which writes the packs of `tests/pack_references.py` on the current stream."""

import torch

from . import runtime
from .packs import ATOM_BLOCKS, ATOM_ROWS, GROUP, MXFP8_BLOCK, ceil_div, round_up, span_count
from .tensors import require

ELEMENTS = {torch.bfloat16: "bf16", torch.float32: "f32"}
CHUNK = 16
ALIGNMENT_BYTES = 16
MAX_U32 = (1 << 32) - 1
MAX_GRID_Y = (1 << 16) - 1
BLOCK_SIGNATURE = "pppuuuuu"
TILE_SIGNATURE = "pppuu"
ROW_SIGNATURE = "pppuu"
COLUMN_SIGNATURE = "pppppuuu"
HELD_ROW_SIGNATURE = "pppu"
HELD_ROW_CHUNKS = 4


def element_of(x, name, dims):
    """Returns the element name of `x` after the checks that read only the shape and dtype."""
    require(x.dim() in dims, f"{name} needs {' or '.join(map(str, dims))} dimensions, got {tuple(x.shape)}")
    require(x.dtype in ELEMENTS, f"{name} must be bfloat16 or float32, got {x.dtype}")
    require(x.shape[-1] % CHUNK == 0, f"the rows of {name} must hold a multiple of {CHUNK} values")
    require(x.numel() <= MAX_U32, f"{name} must have fewer than 2^32 values")
    return ELEMENTS[x.dtype]


def source(x, name, dims):
    element = element_of(x, name, dims)
    require(x.is_cuda and x.is_contiguous(), f"{name} must be a contiguous CUDA tensor")
    require(x.data_ptr() % ALIGNMENT_BYTES == 0, f"{name} must start at a multiple of {ALIGNMENT_BYTES} bytes")
    return element


def require_outputs(x, names, outputs, expected):
    for name, output, (shape, dtype) in zip(names, outputs, expected):
        require(tuple(output.shape) == tuple(shape) and output.dtype == dtype,
                f"{name} must be a {dtype} tensor of the shape {tuple(shape)}, got {output.dtype} {tuple(output.shape)}")
        require(output.is_contiguous(), f"{name} must be contiguous")
        require(output.device == x.device, f"{name} must be on {x.device}, got {output.device}")


def launch(kernel_name, grid, signature, values, device):
    kernel, info = runtime.load(kernel_name)
    runtime.run(kernel, grid, info.threads, 0, signature, values, torch.cuda.current_stream(device).cuda_stream)


def scale_dtype(e8m0):
    return torch.uint8 if e8m0 else torch.float32


def launch_blocks(kernel_name, x, values, scales, rows, width, code_rows, code_width, scale_width, matrices):
    chunks = round_up(rows, ATOM_ROWS) * scale_width // CHUNK
    require(chunks <= MAX_U32 and matrices <= MAX_GRID_Y, "the input is too large")
    if chunks * matrices == 0:
        return
    arguments = (x.data_ptr(), values.data_ptr(), scales.data_ptr(), rows, width, code_rows, code_width, scale_width)
    launch(kernel_name, (ceil_div(chunks, runtime.info(kernel_name).threads), matrices, 1), BLOCK_SIGNATURE,
           arguments, x.device)


def mxfp8_pack_shapes(x):
    """Returns the shapes and dtypes of the values and scales of the MXFP8 pack of `x` `[rows, K]` or `[L, rows, K]`."""
    element_of(x, "x", (2, 3))
    *lead, rows, width = x.shape
    code_rows = rows if x.dim() == 2 else round_up(rows, ATOM_ROWS)
    code_width = round_up(width, GROUP)
    return ((*lead, code_rows, code_width), torch.float8_e4m3fn), \
        ((*lead, ceil_div(rows, ATOM_ROWS), code_width // GROUP, 32, 16), torch.uint8)


def empty_outputs(x, shapes):
    return tuple(torch.empty(shape, dtype=dtype, device=x.device) for shape, dtype in shapes)


def mxfp8_quant_(x, values, scales):
    """Writes the MXFP8 pack of `x` into `values` and `scales`, which have the shapes of `mxfp8_pack_shapes`."""
    element = source(x, "x", (2, 3))
    require_outputs(x, ("values", "scales"), (values, scales), mxfp8_pack_shapes(x))
    require(values.data_ptr() % ALIGNMENT_BYTES == 0, f"values must start at a multiple of {ALIGNMENT_BYTES} bytes")
    matrices, rows, width = (1, *x.shape) if x.dim() == 2 else x.shape
    code_rows, code_width = values.shape[-2:]
    launch_blocks(f"mxfp8_quant_{element}", x, values, scales, rows, width, code_rows, code_width, code_width,
                  matrices)


def mxfp8_quant(x):
    """Returns the MXFP8 pack of `x` `[rows, K]`, or of each matrix of `x` `[L, rows, K]` with rows padded to 128."""
    values, scales = empty_outputs(x, mxfp8_pack_shapes(x))
    mxfp8_quant_(x, values, scales)
    return values, scales


def mxfp8_columnwise_shapes(x):
    """Returns the shapes and dtypes of the values and scales of the columnwise MXFP8 pack of `x` `[rows, K]`."""
    element_of(x, "x", (2,))
    return mxfp8_pack_shapes(x)


def launch_columns(kernel_name, x, row_pack, column_pack):
    rows, width = x.shape
    code_width = column_pack[0].shape[1]
    items = round_up(rows, ATOM_ROWS) // MXFP8_BLOCK * code_width // 2
    require(items <= MAX_U32, "the input is too large")
    if items == 0:
        return
    arguments = (x.data_ptr(), *(tensor.data_ptr() for tensor in (*row_pack, *column_pack)), rows, width, code_width)
    launch(kernel_name, (ceil_div(items, runtime.info(kernel_name).threads), 1, 1), COLUMN_SIGNATURE, arguments,
           x.device)


def mxfp8_quant_columnwise(x):
    """Returns the columnwise MXFP8 pack of `x` `[rows, K]`, an operand of `mxfp8_mm_atb`. The values
    `[rows, round128(K)]` keep the layout of `x`, and each 32 rows of a column share a scale. The scales
    `[ceil(rows/128), round128(K)/128, 32, 16]` hold atoms of 128 columns by 4 row blocks."""
    element = source(x, "x", (2,))
    values, scales = empty_outputs(x, mxfp8_columnwise_shapes(x))
    launch_columns(f"mxfp8_quant_columnwise_{element}", x, (values, scales), (values, scales))
    return values, scales


def mxfp8_quant_dual(x):
    """Returns the rowwise and the columnwise MXFP8 packs of `x` `[rows, K]` from one read of `x`."""
    element = source(x, "x", (2,))
    row_pack = empty_outputs(x, mxfp8_columnwise_shapes(x))
    column_pack = empty_outputs(x, mxfp8_columnwise_shapes(x))
    launch_columns(f"mxfp8_quant_dual_{element}", x, row_pack, column_pack)
    return (*row_pack, *column_pack)


def fp8_b128_pack_shapes(x, e8m0):
    element_of(x, "x", (2,))
    rows, width = x.shape
    require(width % GROUP == 0, f"the rows of x must hold a multiple of {GROUP} values")
    return ((rows, width), torch.float8_e4m3fn), \
        ((ceil_div(rows, ATOM_ROWS), span_count(width), 32, 16), scale_dtype(e8m0))


def fp8_b128_quant(x, *, e8m0):
    """Returns the block128 activation pack of `x` `[rows, K]` with one scale per 128 values of a row."""
    element = source(x, "x", (2,))
    values, scales = empty_outputs(x, fp8_b128_pack_shapes(x, e8m0))
    rows, width = x.shape
    blocks = ATOM_BLOCKS * span_count(width)
    suffix = "e8m0" if e8m0 else "f32"
    launch_blocks(f"fp8_b128_quant_{element}_{suffix}", x, values, scales, rows, width, rows, width, blocks * GROUP, 1)
    return values, scales


def fp8_b128_weight_pack_shapes(w, e8m0):
    element_of(w, "w", (2,))
    n, k = w.shape
    require(n % GROUP == 0 and k % GROUP == 0, f"both dimensions of w must be multiples of {GROUP}")
    return ((n, k), torch.float8_e4m3fn), ((n // GROUP, ATOM_BLOCKS * span_count(k)), scale_dtype(e8m0))


def fp8_b128_quant_weight(w, *, e8m0):
    """Returns the block128 weight pack of `w` `[N, K]` with one scale per 128 by 128 tile."""
    element = source(w, "w", (2,))
    values, scales = empty_outputs(w, fp8_b128_weight_pack_shapes(w, e8m0))
    n, k = w.shape
    scale_width = scales.shape[1]
    require(n // GROUP <= MAX_GRID_Y, "w must have fewer than 2^16 tiles in a column")
    if n * k > 0:
        suffix = "e8m0" if e8m0 else "f32"
        launch(f"fp8_b128_quant_weight_{element}_{suffix}", (scale_width, n // GROUP, 1), TILE_SIGNATURE,
               (w.data_ptr(), values.data_ptr(), scales.data_ptr(), k, scale_width), w.device)
    return values, scales


def fp8_rowwise_pack_shapes(x):
    element_of(x, "x", range(2, 9))
    return (x.shape, torch.float8_e4m3fn), ((*x.shape[:-1], 1), torch.float32)


def fp8_rowwise_quant(x):
    """Returns the rowwise pack of torchao `PerRow` for `x` `[..., K]`: E4M3 values and FP32 scales `[..., 1]`."""
    element = source(x, "x", range(2, 9))
    values, scales = empty_outputs(x, fp8_rowwise_pack_shapes(x))
    width = x.shape[-1]
    rows = x.numel() // width if width else 0
    if width == 0:
        scales.zero_()
        return values, scales
    held = f"fp8_rowwise_quant_held_{element}"
    if rows > 0 and width <= runtime.info(held).threads * HELD_ROW_CHUNKS * CHUNK:
        launch(held, (rows, 1, 1), HELD_ROW_SIGNATURE, (x.data_ptr(), values.data_ptr(), scales.data_ptr(), width),
               x.device)
    elif rows > 0:
        name = f"fp8_rowwise_quant_{element}"
        rows_per_group = runtime.info(name).threads // 32
        launch(name, (ceil_div(rows, rows_per_group), 1, 1), ROW_SIGNATURE,
               (x.data_ptr(), values.data_ptr(), scales.data_ptr(), rows, width), x.device)
    return values, scales
