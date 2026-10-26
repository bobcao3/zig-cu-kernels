"""References in plain PyTorch for the operators of `zig_cu_kernels.torch.fp8_mm`. The GLU activations use approximate
instructions, so `glu` returns the least and the greatest H that their error bounds allow."""

import math

import torch

import pack_references
from zig_cu_kernels import packs

FP32 = torch.float32
EX2_ULPS = 2
RCP_ULPS = 1
RSQRT_RELATIVE_ERROR = 2.0**-22.9
NEGATIVE_LOG2E = torch.tensor(0xBFB8AA3B - (1 << 32), dtype=torch.int32).view(FP32)
GELU_BETA = math.sqrt(2.0 / math.pi)
GELU_KAPPA = 0.044715
GELU_EXPONENT_LINEAR = torch.tensor(2.0 * math.log2(math.e) * GELU_BETA, dtype=FP32)
GELU_EXPONENT_CUBIC = torch.tensor(2.0 * math.log2(math.e) * GELU_BETA * GELU_KAPPA, dtype=FP32)


def dequantized(values, scales, recipe):
    grid = packs.dequantize(torch.ones(values.shape, dtype=FP32).to(values.dtype), scales.cpu(), recipe)
    return values.cpu().double() * grid.double()


def dequantized_columnwise(values, scales):
    k, m = values.shape
    blocks = packs.ceil_div(k, packs.MXFP8_BLOCK)
    grid = packs.scale_grid(scales.cpu().permute(1, 0, 2, 3).contiguous(), m, blocks)
    expanded = packs.decoded(grid).double().repeat_interleave(packs.MXFP8_BLOCK, dim=1)[:, :k]
    return values.cpu().double() * expanded.t()


def product(a, b):
    return a @ b.t(), a.abs() @ b.abs().t()


def bf16_store(*terms):
    total = terms[0].float()
    for term in terms[1:]:
        if term is not None:
            total = total + term.float()
    return total.bfloat16()


def fma(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def ulp_shifted(x, ulps):
    toward = torch.full_like(x, math.inf if ulps > 0 else -math.inf)
    for _ in range(abs(ulps)):
        x = torch.nextafter(x, toward)
    return x


def approximations(exact, ulps):
    return ulp_shifted(exact, -ulps), ulp_shifted(exact, ulps)


def reciprocals(x):
    return approximations(1 / x, RCP_ULPS)


def silus(gate):
    """Returns the FP32 values of `gate · rcp.approx(1 + ex2.approx(gate · -log2 e))` at the ends of the error bounds of
    `ex2` (PTX ISA 9.7.3.21) and `rcp` (PTX ISA 9.7.3.13)."""
    exponent = gate * NEGATIVE_LOG2E
    results = []
    for power in approximations(torch.exp2(exponent.double()).float(), EX2_ULPS):
        results += [gate * reciprocal for reciprocal in reciprocals(power + 1)]
    return results


def gelus(gate):
    square = gate * gate
    exponent = gate.abs() * fma(GELU_EXPONENT_CUBIC, square, GELU_EXPONENT_LINEAR)
    sign = torch.where(gate < 0, -0.5, 0.5).float()
    results = []
    for power in approximations(torch.exp2(exponent.double()).float(), EX2_ULPS):
        for reciprocal in reciprocals(power + 1):
            magnitude = fma(reciprocal, torch.tensor(-2.0), torch.tensor(1.0))
            results.append(gate * fma(sign, magnitude, torch.tensor(0.5)))
    return results


def halves(z, gate_first):
    first, second = z.float().chunk(2, dim=1)
    return (second, first) if gate_first else (first, second)


def glu(z, activation, gate_first, clamp_limit=None):
    """SwiGLU computes `bf16(up · bf16(gate · sigmoid(gate)))` after the optional clamp of both halves, and GEGLU
    computes `bf16(bf16(gelu_tanh(gate)) · up)`."""
    assert z.dtype == torch.bfloat16 and z.dim() == 2 and z.shape[1] % 2 == 0, (z.dtype, z.shape)
    up, gate = halves(z.cpu(), gate_first)
    if activation == "geglu_tanh":
        assert gate_first and clamp_limit is None
        activations = gelus(gate)
    else:
        assert activation == "swiglu", activation
        if clamp_limit is not None:
            limit = torch.tensor(clamp_limit, dtype=FP32).bfloat16().float()
            up, gate = up.clamp(-limit, limit), gate.clamp(max=limit)
        activations = silus(gate)
    products = torch.stack([(up * value.bfloat16().float()).bfloat16().float() for value in activations])
    return products.amin(0).bfloat16(), products.amax(0).bfloat16()


def glu_pack(z, recipe, activation, gate_first, clamp_limit=None, hidden=None, e8m0=True):
    h_low, h_high = glu(z, activation, gate_first, clamp_limit)
    if hidden is not None:
        h_low[:, hidden:], h_high[:, hidden:] = 0, 0
    block = {"mxfp8": packs.MXFP8_BLOCK, "fp8_b128": packs.GROUP, "fp8_rowwise": h_low.shape[1]}[recipe]
    agree = (h_low.view(torch.int16) == h_high.view(torch.int16)).view(h_low.shape[0], -1, block).all(-1)
    low = pack_references.glu_h_quant(h_low, recipe, e8m0=e8m0)
    high = pack_references.glu_h_quant(h_high, recipe, e8m0=e8m0)
    return low, high, agree


def glu_z(z, hidden):
    padded = z.shape[1] // 2
    return torch.cat([z[:, :hidden], z[:, padded : padded + hidden]], dim=1)


def rms_norm_rows(x, scale, eps, rsqrt_shift):
    values = x.float()
    total = (values * values).bfloat16().float().sum(-1, keepdim=True)
    root = torch.rsqrt(total / x.shape[-1] + eps).double() * (1 + rsqrt_shift)
    coefficient = (scale.float()[:, None] * root.float()).bfloat16().float()
    return (values * coefficient).bfloat16()


def rope_pairs(x, rope):
    rotated = rope.shape[1]
    pairs = x[..., :rotated].float().unflatten(-1, (-1, 2))
    cos, sin = rope.view(rope.shape[0], 1, -1, 2).unbind(-1)
    first, second = pairs.unbind(-1)
    turned = torch.stack([first * cos - second * sin, first * sin + second * cos], dim=-1).flatten(-2)
    out = x.clone()
    out[..., :rotated] = turned.bfloat16()
    return out


def qkv_prep(y, qk_scales, rope, eps, n_heads, head_dim, save_qkv, rsqrt_shift=0.0):
    """Returns Q, KV and the saved QKV of the BF16 product `y`. Q and the k plane of KV pass through the per-head
    RMSNorm and RoPE, and the v plane of KV and the saved QKV are copies of `y`."""
    y = y.cpu()
    m, attn_dim = y.shape[0], n_heads * head_dim
    rope = rope.cpu().reshape(m, -1)
    planes = y.view(m, 3, n_heads, head_dim).unbind(1)
    scales = qk_scales.cpu()
    q, k = (rope_pairs(rms_norm_rows(plane, scales, eps, rsqrt_shift), rope).reshape(m, attn_dim)
            for plane in planes[:2])
    kv = torch.cat([k, planes[2].reshape(m, attn_dim)], dim=1)
    saved = y.clone() if save_qkv else y.new_empty((0, 3 * attn_dim))
    return q, kv, saved


GELU_HALF_SLOPE_CONSTANT = torch.tensor(0.5 * GELU_BETA, dtype=FP32)
GELU_HALF_SLOPE_QUADRATIC = torch.tensor(1.5 * GELU_BETA * GELU_KAPPA, dtype=FP32)


FP32_MIN_NORMAL = 2.0**-126


def flushed(x):
    """Replaces the subnormal values of `x` by zeros of the same sign, as an `.ftz` instruction does."""
    return torch.where(x.abs() < FP32_MIN_NORMAL, x * 0, x)


def mul_ftz(a, b):
    return flushed(flushed(a) * flushed(b))


def swiglu_gradients(dh, up, gate, power, reciprocal_shift):
    """The SwiGLU steps flush subnormal operands and results, so a gradient below 2^-126 becomes a signed zero."""
    sigmoid = flushed(ulp_shifted(1 / (power + 1), reciprocal_shift))
    silu = mul_ftz(gate, sigmoid)
    slope = flushed(fma(silu, 1 - sigmoid, sigmoid))
    return mul_ftz(dh, silu).bfloat16(), mul_ftz(mul_ftz(dh, up), slope).bfloat16(), mul_ftz(up, silu).bfloat16()


def geglu_gradients(dh, up, gate, power, reciprocal_shift):
    square = gate * gate
    magnitude = fma(ulp_shifted(1 / (power + 1), reciprocal_shift), torch.tensor(-2.0), torch.tensor(1.0))
    half = fma(torch.where(gate < 0, -0.5, 0.5).float(), magnitude, torch.tensor(0.5))
    half_slope = fma(GELU_HALF_SLOPE_QUADRATIC, square, GELU_HALF_SLOPE_CONSTANT)
    derivative = fma(gate * fma(-magnitude, magnitude, torch.tensor(1.0)), half_slope, half)
    gelu = (gate * half).bfloat16().float()
    up_gradient = (dh * up).bfloat16().float()
    return (dh * gelu).bfloat16(), (derivative * up_gradient).bfloat16(), (up * gelu).bfloat16()


def glu_backward(dh, z, activation, gate_first, clamp_limit=None):
    """Returns the least and the greatest `(dz, h)` that the error bounds of `ex2.approx` and `rcp.approx` allow, with
    `dz` `[M, 2F]` in the half order of `z`. SwiGLU computes `dz_up = bf16(dh · silu)`,
    `dz_gate = bf16(dh · up · (σ + silu · (1 - σ)))` and `h = bf16(up · silu)` from the clamped operands, and zeros the
    gradient of a clamped operand. GEGLU computes `dz_up = bf16(dh · bf16(gelu))`,
    `dz_gate = bf16(gelu' · bf16(dh · up))` and `h = bf16(up · bf16(gelu))`."""
    up, gate = halves(z.cpu(), gate_first)
    dh = dh.cpu().float()
    up_kept = gate_kept = None
    if activation == "geglu_tanh":
        assert gate_first and clamp_limit is None
        gradients = geglu_gradients
        exponent = gate.abs() * fma(GELU_EXPONENT_CUBIC, gate * gate, GELU_EXPONENT_LINEAR)
    else:
        assert activation == "swiglu", activation
        if clamp_limit is not None:
            limit = torch.tensor(clamp_limit, dtype=FP32).bfloat16().float()
            clamped_up, clamped_gate = up.clamp(-limit, limit), gate.clamp(max=limit)
            up_kept, gate_kept = clamped_up == up, clamped_gate == gate
            up, gate = clamped_up, clamped_gate
        gradients = swiglu_gradients
        exponent = mul_ftz(gate, NEGATIVE_LOG2E)
    corners = [gradients(dh, up, gate, power, shift)
               for power in approximations(torch.exp2(exponent.double()).float(), EX2_ULPS)
               for shift in (-RCP_ULPS, RCP_ULPS)]
    bounds = []
    for values in zip(*corners):
        stacked = torch.stack([value.float() for value in values])
        bounds.append((stacked.amin(0), stacked.amax(0)))
    (up_low, up_high), (gate_low, gate_high), (h_low, h_high) = bounds
    if up_kept is not None:
        up_low, up_high = (torch.where(up_kept, x, x * 0) for x in (up_low, up_high))
        gate_low, gate_high = (torch.where(gate_kept, x, x * 0) for x in (gate_low, gate_high))
    order = (lambda u, g: (g, u)) if gate_first else (lambda u, g: (u, g))
    low = torch.cat(order(up_low, gate_low), dim=1).bfloat16(), h_low.bfloat16()
    high = torch.cat(order(up_high, gate_high), dim=1).bfloat16(), h_high.bfloat16()
    return low, high


def glu_backward_packs(dz, h):
    """Returns the packs `(dz_rw, s_dz_rw, dz_cw, s_dz_cw, h_cw, s_h_cw)` of the BF16 `dz` and `h`."""
    return (*pack_references.mxfp8_quant(dz), *pack_references.mxfp8_quant_columnwise(dz),
            *pack_references.mxfp8_quant_columnwise(h))
