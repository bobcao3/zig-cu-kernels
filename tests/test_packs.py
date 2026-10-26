import hashlib

import pytest
import torch

import pack_references as ref
from zig_cu_kernels import packs

SHAPES = [(1, 128), (7, 640), (130, 640), (257, 5376), (300, 128)]
DTYPES = [torch.bfloat16, torch.float32]


def lattice(rows, cols, dtype):
    """Returns rows of 24-bit integers, each row times a power of two from 2^-63 to 2^-3, with an all-zero first
    row, an all-zero 128-column block in the second row and E4M3 ties and saturating values in the third and fourth."""
    index = torch.arange(rows * cols, dtype=torch.int64).view(rows, cols)
    integers = (index * 2654435761 + 12345) % (1 << 24) - (1 << 23)
    exponents = torch.arange(rows, dtype=torch.int64)[:, None] * 13 % 61 - 63
    x = integers.double() * torch.exp2(exponents.double())
    x[0] = 0
    if rows > 1 and cols >= 256:
        x[1, 128:256] = 0
    if rows > 3:
        edge = torch.tensor([448.0, -448.0, 464.0, 447.75, 25.0, 17.0, 480.0, 1e-3], dtype=torch.float64)
        x[2, : edge.numel()] = edge
        x[3, :8] = torch.arange(17.0, 32, 2.0, dtype=torch.float64)
        x[3, 8] = 448.0
    return x.to(dtype)


def digest(pack):
    hasher = hashlib.sha256()
    for tensor in pack:
        hasher.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        hasher.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()[:16]


def quantizers():
    return {
        "mxfp8": ref.mxfp8_quant,
        "fp8_b128_e8m0": lambda x: ref.fp8_b128_quant(x, e8m0=True),
        "fp8_b128_fp32": lambda x: ref.fp8_b128_quant(x, e8m0=False),
        "fp8_rowwise": ref.fp8_rowwise_quant,
    }


def dequantization_bound(x, pack, recipe):
    """Returns whether each dequantized value lies within half an E4M3 step of its source value, with a margin
    of 2^-4 times the step for the rounding of FP32 scales."""
    values, scales = pack
    rows, cols = x.shape
    dequantized = packs.dequantize(values, scales, recipe)[:rows, :cols]
    scale = (dequantized / values.float()[:rows, :cols]).nan_to_num(nan=0.0)
    step = torch.exp2(torch.floor(torch.log2(values.float()[:rows, :cols].abs().clamp(min=2**-6))) - 3)
    error = (dequantized.double() - x.double()).abs()
    within = error <= (0.5 + 2**-4) * step.double() * scale.abs().double()
    saturated = values.float()[:rows, :cols].abs() == packs.E4M3_MAX
    zero = values.float()[:rows, :cols] == 0
    return within | saturated | zero


def test_scale_atom_index_finds_the_grid_elements_in_the_atoms():
    for rows, blocks in [(1, 4), (130, 20), (257, 168), (128, 8)]:
        grid = torch.arange(rows * blocks, dtype=torch.int32).view(rows, blocks) + 1
        atoms = packs.scale_atoms(grid, 0)
        row = torch.arange(rows)[:, None]
        block = torch.arange(blocks)[None, :]
        assert torch.equal(atoms.flatten()[packs.scale_atom_index(row, block, atoms.shape[1] * 4)], grid)
        assert (atoms != 0).sum() == rows * blocks
        assert torch.equal(packs.scale_grid(atoms, rows, blocks), grid)


def test_scale_atom_index_of_single_elements():
    assert packs.scale_atom_index(0, 0, 4) == 0
    assert packs.scale_atom_index(33, 2, 4) == 16 + 4 + 2
    assert packs.scale_atom_index(127, 3, 8) == 31 * 16 + 12 + 3
    assert packs.scale_atom_index(128, 5, 8) == 3 * 512 + 1


@pytest.mark.parametrize(("rows", "cols"), SHAPES + [(5, 200)])
def test_pack_shapes(rows, cols):
    x = lattice(rows, cols, torch.bfloat16)
    values, scales = ref.mxfp8_quant(x)
    assert values.shape == packs.mxfp8_values_shape(rows, cols)
    assert scales.shape == packs.mxfp8_scale_shape(rows, cols) and scales.dtype == torch.uint8
    values, scales = ref.fp8_rowwise_quant(x)
    assert values.shape == x.shape and scales.shape == packs.fp8_rowwise_scale_shape(x.shape)
    if cols % packs.GROUP == 0:
        for e8m0, dtype in [(True, torch.uint8), (False, torch.float32)]:
            values, scales = ref.fp8_b128_quant(x, e8m0=e8m0)
            assert scales.shape == packs.fp8_b128_activation_scale_shape(rows, cols) and scales.dtype == dtype
            if rows % packs.GROUP == 0:
                values, scales = ref.fp8_b128_quant_weight(x, e8m0=e8m0)
                assert scales.shape == packs.fp8_b128_weight_scale_shape(rows, cols) and scales.dtype == dtype


def test_mxfp8_of_three_dimensions_pads_value_rows_to_whole_atoms():
    x = lattice(390, 640, torch.bfloat16).view(3, 130, 640)
    values, scales = ref.mxfp8_quant(x)
    assert values.shape == (3, 256, 640) and scales.shape == (3, 2, 5, 32, 16)
    for matrix in range(3):
        assert torch.equal(values[matrix, :130].view(torch.uint8), ref.mxfp8_quant(x[matrix])[0].view(torch.uint8))
        assert (values[matrix, 130:].view(torch.uint8) == 0).all()


def test_e8m0_ceil_rounds_up_to_powers_of_two():
    exponents = torch.arange(-126, 127, dtype=torch.float32)
    powers = torch.exp2(exponents)
    assert torch.equal(packs.e8m0_ceil(powers), (exponents + 127).to(torch.uint8))
    above = torch.nextafter(powers, torch.tensor(float("inf")))
    assert torch.equal(packs.e8m0_ceil(above[:-1]), (exponents[:-1] + 128).to(torch.uint8))
    assert packs.e8m0_ceil(torch.tensor([0.0, 2.0**-149, 3e38])).tolist() == [0, 1, 254]


def test_the_fp32_reciprocal_of_448_is_the_rounded_literal_of_the_mxfp8_kernels():
    assert ref.INV_448 == torch.tensor(0.002232142857, dtype=torch.float32)


def test_e8m0_decode_is_exact():
    byte = torch.arange(255, dtype=torch.int32)
    assert torch.equal(packs.e8m0_decode(byte.to(torch.uint8)).double(), torch.exp2(byte.double() - 127))
    assert packs.e8m0_decode(torch.tensor([255], dtype=torch.uint8)).isnan().all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("rows", "cols"), SHAPES)
def test_mxfp8_scales_and_padding(rows, cols, dtype):
    x = lattice(rows, cols, dtype)
    values, scales = ref.mxfp8_quant(x)
    grid = packs.scale_grid(scales, scales.shape[0] * packs.ATOM_ROWS, cols // packs.MXFP8_BLOCK)
    amax = x.float().view(rows, -1, packs.MXFP8_BLOCK).abs().amax(dim=-1)
    assert (grid[:rows][amax == 0] == packs.E8M0_BIAS).all()
    assert (grid[rows:] == packs.E8M0_BIAS).all()
    nonzero = amax > 0
    assert (amax[nonzero] <= packs.e8m0_decode(grid[:rows][nonzero]) * packs.E4M3_MAX).all()
    assert (amax[nonzero] > packs.e8m0_decode(grid[:rows][nonzero]) * (packs.E4M3_MAX / 2) * (1 - 2**-20)).all()
    assert dequantization_bound(x, (values, scales), "mxfp8").all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("rows", "cols"), SHAPES)
@pytest.mark.parametrize("e8m0", [True, False])
def test_fp8_b128_scales_and_padding(rows, cols, dtype, e8m0):
    x = lattice(rows, cols, dtype)
    values, scales = ref.fp8_b128_quant(x, e8m0=e8m0)
    slots = scales.shape[1] * packs.ATOM_BLOCKS
    padded = packs.scale_grid(scales, scales.shape[0] * packs.ATOM_ROWS, slots)
    assert (padded[rows:] == 0).all() and (padded[:, cols // packs.GROUP :] == 0).all()
    amax = x.float().view(rows, -1, packs.GROUP).abs().amax(dim=-1)
    grid = padded[:rows, : cols // packs.GROUP]
    if e8m0:
        assert (grid[amax == 0] == packs.E8M0_BIAS).all()
    else:
        assert torch.equal(grid, torch.maximum(amax * ref.INV_448, ref.MIN_GROUP_SCALE))
    assert dequantization_bound(x, (values, scales), "fp8_b128").all()
    if rows % packs.GROUP == 0:
        values, scales = ref.fp8_b128_quant_weight(x, e8m0=e8m0)
        assert (scales[:, cols // packs.GROUP :] == 0).all()
        assert dequantization_bound(x, (values, scales), "fp8_b128").all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_fp8_rowwise_rounds_ties_to_even_and_saturates(dtype):
    ties = torch.arange(17.0, 32, 2.0)
    row = torch.cat([ties, -ties, torch.tensor([448.0, -448.0])])
    x = (row * torch.exp2(torch.arange(-8, 8))[:, None]).to(dtype)
    values, scales = ref.fp8_rowwise_quant(x)
    assert torch.equal(scales, torch.exp2(torch.arange(-8.0, 8))[:, None])
    expected = torch.cat([ties + torch.where(ties % 4 == 1, -1.0, 1.0), -ties - torch.where(ties % 4 == 1, -1, 1)])
    assert torch.equal(values.float()[:, :16], expected.expand(16, 16))
    assert (values.float().abs().amax(dim=1) == packs.E4M3_MAX).all()


def test_fp8_rowwise_scales_of_zero_rows_and_bf16_rounding():
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    x[1, 0] = 3.0
    _, scales = ref.fp8_rowwise_quant(x)
    assert scales[0, 0] == torch.tensor(torch.tensor(1e-12, dtype=torch.bfloat16).item() / 448).bfloat16().float()
    assert scales[1, 0] == torch.tensor(3 / 448).bfloat16().float()
    _, scales = ref.fp8_rowwise_quant(x.float())
    assert scales[1, 0] == torch.tensor(3.0) * torch.tensor(1 / 448)


def test_glu_h_packs_follow_their_recipes():
    h = lattice(130, 1280, torch.bfloat16)
    assert digest(ref.glu_h_quant(h, "mxfp8")) == digest(ref.mxfp8_quant(h))
    assert digest(ref.glu_h_quant(h, "fp8_b128", e8m0=True)) == digest(ref.fp8_b128_quant(h, e8m0=True))
    assert digest(ref.glu_h_quant(h, "fp8_rowwise")) == digest(ref.fp8_rowwise_quant(h))


DIGESTS = {
    "mxfp8 1x128 bfloat16": "45af7bad6dcab034",
    "fp8_b128_e8m0 1x128 bfloat16": "05ddc163fb9a204c",
    "fp8_b128_fp32 1x128 bfloat16": "00f43cde65f7f723",
    "fp8_rowwise 1x128 bfloat16": "34733fae31c27e09",
    "mxfp8 7x640 bfloat16": "729240c942b7aad0",
    "fp8_b128_e8m0 7x640 bfloat16": "c0af674cc7e36c72",
    "fp8_b128_fp32 7x640 bfloat16": "310a736a95acc395",
    "fp8_rowwise 7x640 bfloat16": "083dcc19047d9acb",
    "mxfp8 130x640 bfloat16": "4ce70cc5a7d51581",
    "fp8_b128_e8m0 130x640 bfloat16": "9ea7e3cf0becf8bc",
    "fp8_b128_fp32 130x640 bfloat16": "d7d21d431fb26f78",
    "fp8_rowwise 130x640 bfloat16": "2bb88fa476d4c412",
    "mxfp8 257x5376 bfloat16": "9339b35b48f9ae1b",
    "fp8_b128_e8m0 257x5376 bfloat16": "12126ddc1ae58a6c",
    "fp8_b128_fp32 257x5376 bfloat16": "94710aea83bd728f",
    "fp8_rowwise 257x5376 bfloat16": "77a8a12ea235c512",
    "mxfp8 300x128 bfloat16": "d70efbc97f438bb6",
    "fp8_b128_e8m0 300x128 bfloat16": "7ecc3e95bc2fa2e2",
    "fp8_b128_fp32 300x128 bfloat16": "18e4132634d192f5",
    "fp8_rowwise 300x128 bfloat16": "a39389806a9e255f",
    "mxfp8 1x128 float32": "45af7bad6dcab034",
    "fp8_b128_e8m0 1x128 float32": "05ddc163fb9a204c",
    "fp8_b128_fp32 1x128 float32": "00f43cde65f7f723",
    "fp8_rowwise 1x128 float32": "6a606807b210faa9",
    "mxfp8 7x640 float32": "22253fbc818ab0c8",
    "fp8_b128_e8m0 7x640 float32": "8e64951eeaabe294",
    "fp8_b128_fp32 7x640 float32": "8644dbbe0b24d3f2",
    "fp8_rowwise 7x640 float32": "3121dc2971eb0ed0",
    "mxfp8 130x640 float32": "a1f1704c9acd5c49",
    "fp8_b128_e8m0 130x640 float32": "d2affa84cc905993",
    "fp8_b128_fp32 130x640 float32": "39dd5ba3b44a81dc",
    "fp8_rowwise 130x640 float32": "573bab4f7932c1a4",
    "mxfp8 257x5376 float32": "bebc6cad5a5e3e9a",
    "fp8_b128_e8m0 257x5376 float32": "8db25562f6282a13",
    "fp8_b128_fp32 257x5376 float32": "9628b33f3d4b060f",
    "fp8_rowwise 257x5376 float32": "72a3285fae0d2f05",
    "mxfp8 300x128 float32": "99666775c62fc96f",
    "fp8_b128_e8m0 300x128 float32": "18450822849c6337",
    "fp8_b128_fp32 300x128 float32": "39a7c065d214fe1d",
    "fp8_rowwise 300x128 float32": "769e1ea6a392a45a",
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("rows", "cols"), SHAPES)
@pytest.mark.parametrize("quantizer", ["mxfp8", "fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise"])
def test_packs_keep_the_bits_checked_against_the_upstream_references(rows, cols, dtype, quantizer):
    key = f"{quantizer} {rows}x{cols} {str(dtype).removeprefix('torch.')}"
    assert digest(quantizers()[quantizer](lattice(rows, cols, dtype))) == DIGESTS[key]
