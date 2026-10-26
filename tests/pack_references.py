"""Reference quantizers in plain PyTorch for the FP8 packs of `zig_cu_kernels.packs`."""

import torch

from zig_cu_kernels import packs
from zig_cu_kernels.packs import E4M3_MAX, E8M0_BIAS, GROUP, MXFP8_BLOCK

INV_448 = torch.tensor(1 / 448, dtype=torch.float32)
MIN_GROUP_SCALE = torch.tensor(1e-30, dtype=torch.float32)
AMAX_MIN = {
    torch.bfloat16: torch.tensor(1e-12, dtype=torch.bfloat16).item(),
    torch.float32: torch.tensor(1e-12, dtype=torch.float32).item(),
}


def e4m3(quotients):
    return quotients.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def e8m0_scale(amax):
    byte = packs.e8m0_ceil(amax * INV_448.to(amax.device))
    byte = torch.where(amax == 0, torch.full_like(byte, E8M0_BIAS), byte)
    return byte, torch.exp2(E8M0_BIAS - byte.float())


def fp32_group_scale(amax):
    scale = torch.maximum(amax * INV_448.to(amax.device), MIN_GROUP_SCALE.to(amax.device))
    return scale, 1 / scale


def check_source(x, dims):
    assert x.dim() in dims, x.shape
    assert x.dtype in (torch.bfloat16, torch.float32), x.dtype


def mxfp8_quant_matrix(x, row_pad):
    rows, cols = x.shape
    rows_out, cols_out = packs.round_up(rows, row_pad), packs.round_up(cols, GROUP)
    padded = torch.nn.functional.pad(x.float(), (0, cols_out - cols, 0, rows_out - rows))
    blocks = padded.view(rows_out, cols_out // MXFP8_BLOCK, MXFP8_BLOCK)
    byte, reciprocal = e8m0_scale(blocks.abs().amax(dim=-1))
    values = e4m3(blocks * reciprocal[..., None]).view(rows_out, cols_out)
    return values, packs.scale_atoms(byte, E8M0_BIAS)


def mxfp8_quant(x):
    """Returns the MXFP8 pack of `[rows, K]` or of each matrix of `[L, rows, K]`, rows padded to whole atoms."""
    check_source(x, (2, 3))
    if x.dim() == 2:
        values, scales = mxfp8_quant_matrix(x, 1)
        return values, scales
    pairs = [mxfp8_quant_matrix(matrix, packs.ATOM_ROWS) for matrix in x]
    return torch.stack([v for v, _ in pairs]), torch.stack([s for _, s in pairs])


def mxfp8_quant_columnwise(x):
    """Returns the columnwise MXFP8 pack of `x` `[rows, K]`: values `[rows, round128(K)]` in the layout of `x`, where
    32 rows of a column share a scale, and scales `[ceil(rows/128), round128(K)/128, 32, 16]`. The pack is the rowwise
    pack of the transpose of `x` with the values transposed back and the two atom dimensions swapped."""
    check_source(x, (2,))
    values, scales = mxfp8_quant_matrix(x.t(), GROUP)
    return values[:, : x.shape[0]].t().contiguous(), scales.transpose(0, 1).contiguous()


def mxfp8_quant_dual(x):
    """Returns the rowwise and the columnwise MXFP8 packs of `x` `[rows, K]`."""
    return (*mxfp8_quant(x), *mxfp8_quant_columnwise(x))


def group_scales(amax, e8m0):
    if e8m0:
        return e8m0_scale(amax)
    return fp32_group_scale(amax)


def fp8_b128_quant(x, *, e8m0):
    check_source(x, (2,))
    rows, cols = x.shape
    assert cols % GROUP == 0, cols
    groups = x.float().view(rows, cols // GROUP, GROUP)
    element, reciprocal = group_scales(groups.abs().amax(dim=-1), e8m0)
    values = e4m3(groups * reciprocal[..., None]).view(rows, cols)
    return values, packs.scale_atoms(element, 0)


def fp8_b128_quant_weight(w, *, e8m0):
    check_source(w, (2,))
    n, k = w.shape
    assert n % GROUP == 0 and k % GROUP == 0, w.shape
    tiles = w.float().view(n // GROUP, GROUP, k // GROUP, GROUP)
    element, reciprocal = group_scales(tiles.abs().amax(dim=(1, 3)), e8m0)
    values = e4m3(tiles * reciprocal[:, None, :, None]).view(n, k)
    scales = element.new_zeros(packs.fp8_b128_weight_scale_shape(n, k))
    scales[:, : k // GROUP] = element
    return values, scales


def fp8_rowwise_scales(x):
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=AMAX_MIN[x.dtype])
    scales = amax * INV_448.to(x.device)
    return scales.bfloat16().float() if x.dtype == torch.bfloat16 else scales


def fp8_rowwise_quant(x):
    """Returns the rowwise pack of torchao `PerRow`. PyTorch on CUDA divides the amax by 448 as a product with the
    FP32 reciprocal of 448, and divides `x` by the scale with correct rounding."""
    check_source(x, range(2, 9))
    if x.shape[-1] == 0:
        return x.to(torch.float8_e4m3fn), x.new_zeros(packs.fp8_rowwise_scale_shape(x.shape), dtype=torch.float32)
    scales = fp8_rowwise_scales(x)
    return e4m3(x.float() / scales), scales


def glu_h_quant(h, recipe, *, e8m0=False):
    """Returns the pack of the BF16 GLU output `h` `[rows, hidden]` that the GLU GEMM of `recipe` writes."""
    assert h.dtype == torch.bfloat16, h.dtype
    assert recipe in packs.RECIPES, recipe
    if recipe == "mxfp8":
        return mxfp8_quant(h)
    if recipe == "fp8_b128":
        return fp8_b128_quant(h, e8m0=e8m0)
    return fp8_rowwise_quant(h)
