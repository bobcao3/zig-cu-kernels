"""Layouts of the FP8 packs, E4M3 values and their scales. A 4-D scale tensor holds atoms of 128 rows by 4 blocks
that `tcgen05.cp ... 32x128b.warpx4` copies to tensor memory unchanged (PTX ISA 9.7.18.9.2)."""

import torch

E4M3_MAX = 448.0
E8M0_BIAS = 127
E8M0_NAN = 255
ATOM_ROWS = 128
ATOM_BLOCKS = 4
ATOM_ELEMENTS = 512
MXFP8_BLOCK = 32
GROUP = 128
RECIPES = ("mxfp8", "fp8_b128", "fp8_rowwise")

FP32_MANTISSA_BITS = 23
FP32_MANTISSA_MASK = (1 << FP32_MANTISSA_BITS) - 1
SMALLEST_E8M0_BITS = 1 << (FP32_MANTISSA_BITS - 1)
CANONICAL_NAN_BITS = 0x7FC0_0000


def ceil_div(a, b):
    return -(-a // b)


def round_up(a, b):
    return ceil_div(a, b) * b


def span_count(cols):
    return ceil_div(cols, GROUP * ATOM_BLOCKS)


def mxfp8_values_shape(rows, cols):
    return rows, round_up(cols, GROUP)


def mxfp8_scale_shape(rows, cols):
    return ceil_div(rows, ATOM_ROWS), round_up(cols, GROUP) // GROUP, 32, 16


def fp8_b128_activation_scale_shape(rows, cols):
    assert cols % GROUP == 0, cols
    return ceil_div(rows, ATOM_ROWS), span_count(cols), 32, 16


def fp8_b128_weight_scale_shape(n, k):
    assert n % GROUP == 0 and k % GROUP == 0, (n, k)
    return n // GROUP, ATOM_BLOCKS * span_count(k)


def fp8_rowwise_scale_shape(shape):
    return *shape[:-1], 1


def scale_atom_index(row, block, blocks_per_row):
    """Returns the index of `(row, block)` in a 4-D scale tensor with `blocks_per_row` blocks in each row."""
    assert blocks_per_row % ATOM_BLOCKS == 0, blocks_per_row
    atom = (row // ATOM_ROWS) * (blocks_per_row // ATOM_BLOCKS) + block // ATOM_BLOCKS
    return atom * ATOM_ELEMENTS + (row % 32) * 16 + ((row % ATOM_ROWS) // 32) * ATOM_BLOCKS + block % ATOM_BLOCKS


def scale_atoms(grid, fill):
    """Returns the 4-D scale tensor of the `[rows, blocks]` grid, with `fill` in the padding of the last atoms."""
    rows, blocks = grid.shape
    row_atoms, column_atoms = ceil_div(rows, ATOM_ROWS), ceil_div(blocks, ATOM_BLOCKS)
    padded = grid.new_full((row_atoms * ATOM_ROWS, column_atoms * ATOM_BLOCKS), fill)
    padded[:rows, :blocks] = grid
    atoms = padded.view(row_atoms, ATOM_ROWS // 32, 32, column_atoms, ATOM_BLOCKS)
    return atoms.permute(0, 3, 2, 1, 4).reshape(row_atoms, column_atoms, 32, 16).contiguous()


def scale_grid(atoms, rows, blocks):
    row_atoms, column_atoms = atoms.shape[:2]
    assert atoms.shape[2:] == (32, 16), atoms.shape
    assert rows <= row_atoms * ATOM_ROWS and blocks <= column_atoms * ATOM_BLOCKS, (atoms.shape, rows, blocks)
    grid = atoms.view(row_atoms, column_atoms, 32, ATOM_ROWS // 32, ATOM_BLOCKS).permute(0, 3, 2, 1, 4)
    return grid.reshape(row_atoms * ATOM_ROWS, column_atoms * ATOM_BLOCKS)[:rows, :blocks]


def e8m0_ceil(scale):
    """Returns the E8M0 bytes of `cvt.rp.satfinite.ue8m0x2.f32` for positive normal FP32 `scale` (PTX ISA 9.7.10.24)."""
    assert scale.dtype == torch.float32, scale.dtype
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> FP32_MANTISSA_BITS) & 0xFF
    byte = exponent + ((bits & FP32_MANTISSA_MASK) != 0).to(torch.int32)
    return byte.clamp(0, 254).to(torch.uint8)


def e8m0_decode(byte):
    assert byte.dtype == torch.uint8, byte.dtype
    exponent = byte.to(torch.int32)
    bits = torch.where(exponent == 0, SMALLEST_E8M0_BITS, exponent << FP32_MANTISSA_BITS)
    return torch.where(exponent == E8M0_NAN, CANONICAL_NAN_BITS, bits).view(torch.float32)


def decoded(elements):
    return e8m0_decode(elements) if elements.dtype == torch.uint8 else elements.float()


def dequantize_mxfp8(values, scales):
    rows, cols = values.shape
    grid = decoded(scale_grid(scales, rows, cols // MXFP8_BLOCK))
    return values.float() * grid.repeat_interleave(MXFP8_BLOCK, dim=1)


def dequantize_block128_activation(values, scales):
    rows, cols = values.shape
    grid = decoded(scale_grid(scales, rows, cols // GROUP))
    return values.float() * grid.repeat_interleave(GROUP, dim=1)


def dequantize_block128_weight(values, scales):
    n, k = values.shape
    tiles = decoded(scales[:, : k // GROUP])
    return values.float() * tiles.repeat_interleave(GROUP, dim=0).repeat_interleave(GROUP, dim=1)


def dequantize_rowwise(values, scales):
    return values.float() * scales.float()


def dequantize(values, scales, recipe):
    """Returns the FP32 matrix of a pack. A block128 pack with 2-D scales is a weight pack."""
    assert recipe in RECIPES, recipe
    if recipe == "mxfp8":
        return dequantize_mxfp8(values, scales)
    if recipe == "fp8_rowwise":
        return dequantize_rowwise(values, scales)
    if scales.dim() == 2:
        return dequantize_block128_weight(values, scales)
    return dequantize_block128_activation(values, scales)
