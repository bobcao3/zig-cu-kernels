"""Custom ops for the products of FP8 packs. The schemas match the `tcgen05_kernels` ops of the same names, so a
backend module can bind either set by name. The checks read only shapes, dtypes and devices, never device values."""

import torch
from torch._library.autograd import make_autograd_impl

from .. import fp8_mm as launcher
from .. import packs
from ..packs import GROUP
from .common import current_stream_handle
from .ops import NAMESPACE, WITHOUT_FORMULA, library_of

FP8, BYTE, BF16, FP32 = launcher.FP8, launcher.E8M0_BYTE, launcher.BF16, torch.float32

SCHEMAS = {
    "mxfp8_mm_abt": "(Tensor a, Tensor s_a, Tensor b, Tensor s_b, Tensor? bias=None, int? n=None) -> Tensor",
    "mxfp8_mm_abt_add": "(Tensor a, Tensor s_a, Tensor b, Tensor s_b, Tensor residual) -> Tensor",
    "mxfp8_mm_abt_low_rank": "(Tensor a, Tensor s_a, Tensor b, Tensor s_b, Tensor low, Tensor up, Tensor? bias=None, "
                             "int? n=None) -> Tensor",
    "mxfp8_mm_glu": "(Tensor a, Tensor s_a, Tensor w1, Tensor s_w1, bool gate_first=False, bool out_z=False, "
                    "float? clamp_limit=None, str activation=\"swiglu\", int? hidden=None) -> "
                    "(Tensor?, Tensor, Tensor)",
    "mxfp8_mm_qkv_prep": "(Tensor a, Tensor s_a, Tensor b, Tensor s_b, Tensor qk_scales, Tensor rope, int start_elem, "
                         "float eps, int n_heads, int head_dim, bool save_qkv) -> (Tensor, Tensor, Tensor)",
    "mxfp8_mm_atb": "(Tensor a_cw, Tensor a_cw_scales, Tensor b_cw, Tensor b_cw_scales, int? n=None) -> Tensor",
    "fp8_b128_mm_abt": "(Tensor a, Tensor s_a, Tensor b, Tensor s_b) -> Tensor",
    "fp8_b128_mm_glu": "(Tensor a, Tensor s_a, Tensor w1, Tensor s_w1, str activation, bool out_z) -> "
                       "(Tensor?, Tensor, Tensor)",
    "fp8_rowwise_mm_abt": "(Tensor a, Tensor s_a, Tensor b, Tensor s_b, Tensor? bias) -> Tensor",
    "fp8_rowwise_mm_glu": "(Tensor a, Tensor s_a, Tensor w1, Tensor s_w1, str activation, bool out_z) -> "
                      "(Tensor?, Tensor, Tensor)",
}


def check_tensor(op, name, tensor, dtypes, dim):
    assert isinstance(tensor, torch.Tensor), f"{op}: {name} must be a tensor, got {type(tensor).__name__}"
    assert tensor.dtype in dtypes, f"{op}: {name} must have a dtype of {dtypes}, got {tensor.dtype}"
    assert tensor.dim() == dim, f"{op}: {name} must have {dim} dimensions, got the shape {tuple(tensor.shape)}"
    assert tensor.is_contiguous(), f"{op}: {name} must be contiguous"


def check_devices(op, **tensors):
    tensors = {name: tensor for name, tensor in tensors.items() if tensor is not None}
    first_name, first = next(iter(tensors.items()))
    assert first.device.type in ("cuda", "meta"), f"{op}: {first_name} must be a CUDA tensor, got a tensor on " \
        f"{first.device}"
    for name, tensor in tensors.items():
        assert tensor.device == first.device, f"{op}: {name} must be on {first.device}, got {tensor.device}"


def check_shape(op, name, tensor, shape):
    assert tuple(tensor.shape) == tuple(shape), f"{op}: {name} must have the shape {tuple(shape)}, got " \
        f"{tuple(tensor.shape)}"


def check_width(op, name, width, padded):
    assert 0 < width <= padded and width % launcher.WIDTH_MULTIPLE == 0, \
        f"{op}: {name} must be a positive multiple of {launcher.WIDTH_MULTIPLE} up to {padded}, got {width}"


def check_mxfp8_operands(op, a, s_a, b, s_b, sizes):
    """Returns M, N and K of the MXFP8 product `a bᵀ` after checking the packs."""
    check_tensor(op, "a", a, (FP8,), 2)
    check_tensor(op, "b", b, (FP8,), 2)
    check_tensor(op, "s_a", s_a, (BYTE,), 4)
    check_tensor(op, "s_b", s_b, (BYTE,), 4)
    m, k = a.shape
    n = b.shape[0]
    if sizes:
        assert b.shape[1] == k, f"{op}: b must be [N, {k}], got {tuple(b.shape)}"
        assert m > 0, f"{op}: a needs at least one row"
        assert k % GROUP == 0, f"{op}: K must be a multiple of {GROUP}, got {k}"
        assert n % GROUP == 0, f"{op}: N must be a multiple of {GROUP}, got {n}"
        check_shape(op, "s_a", s_a, packs.mxfp8_scale_shape(m, k))
        check_shape(op, "s_b", s_b, packs.mxfp8_scale_shape(n, k))
    return m, n, k


def check_bias(op, bias, width, sizes):
    if bias is not None:
        check_tensor(op, "bias", bias, (BF16,), 1)
        if sizes:
            check_shape(op, "bias", bias, (width,))


def check_activation(op, activation):
    assert activation in launcher.ACTIVATIONS, f"{op}: activation must be one of {launcher.ACTIVATIONS}, got " \
        f"{activation!r}"


def check_mm_abt(a, s_a, b, s_b, bias, n, sizes):
    op = "mxfp8_mm_abt"
    check_devices(op, a=a, s_a=s_a, b=b, s_b=s_b, bias=bias)
    _, padded_n, _ = check_mxfp8_operands(op, a, s_a, b, s_b, sizes)
    width = padded_n if n is None else n
    if sizes:
        check_width(op, "n", width, padded_n)
    check_bias(op, bias, width, sizes)
    return width


def check_mm_abt_add(a, s_a, b, s_b, residual, sizes):
    op = "mxfp8_mm_abt_add"
    check_devices(op, a=a, s_a=s_a, b=b, s_b=s_b, residual=residual)
    m, padded_n, _ = check_mxfp8_operands(op, a, s_a, b, s_b, sizes)
    check_tensor(op, "residual", residual, (BF16,), 2)
    if sizes:
        assert 0 < residual.shape[0] <= m, f"{op}: residual must have 1 to {m} rows, got {residual.shape[0]}"
        assert residual.shape[1] == padded_n, f"{op}: residual must have {padded_n} columns, got {residual.shape[1]}"


def check_mm_abt_low_rank(a, s_a, b, s_b, low, up, bias, n, sizes):
    op = "mxfp8_mm_abt_low_rank"
    check_devices(op, a=a, s_a=s_a, b=b, s_b=s_b, low=low, up=up, bias=bias)
    m, padded_n, _ = check_mxfp8_operands(op, a, s_a, b, s_b, sizes)
    width = padded_n if n is None else n
    check_tensor(op, "low", low, (BF16,), 2)
    check_tensor(op, "up", up, (BF16,), 2)
    if sizes:
        check_width(op, "n", width, padded_n)
        rank = low.shape[1]
        assert rank in launcher.LOW_RANKS, f"{op}: the rank must be one of {launcher.LOW_RANKS}, got {rank}"
        check_shape(op, "low", low, (m, rank))
        check_shape(op, "up", up, (width, rank))
    check_bias(op, bias, width, sizes)
    return width


def check_mm_glu(a, s_a, w1, s_w1, gate_first, clamp_limit, activation, hidden, sizes):
    op = "mxfp8_mm_glu"
    check_devices(op, a=a, s_a=s_a, w1=w1, s_w1=s_w1)
    _, rows, _ = check_mxfp8_operands(op, a, s_a, w1, s_w1, sizes)
    check_activation(op, activation)
    assert activation == "swiglu" or (gate_first and clamp_limit is None), \
        f"{op}: geglu_tanh takes the half order [gate, up] (gate_first=True) and no clamp_limit"
    padded_hidden = rows // 2
    hidden = padded_hidden if hidden is None else hidden
    if sizes:
        assert rows % (2 * GROUP) == 0, f"{op}: w1 must have 2·Fp rows with Fp a multiple of {GROUP}, got {rows}"
        check_width(op, "hidden", hidden, padded_hidden)
        assert packs.round_up(hidden, GROUP) == padded_hidden, \
            f"{op}: hidden must round up to Fp = {padded_hidden}, got {hidden}"
    return padded_hidden, hidden


def check_mm_qkv_prep(a, s_a, b, s_b, qk_scales, rope, start_elem, n_heads, head_dim, sizes):
    op = "mxfp8_mm_qkv_prep"
    check_devices(op, a=a, s_a=s_a, b=b, s_b=s_b, qk_scales=qk_scales, rope=rope)
    m, padded_n, _ = check_mxfp8_operands(op, a, s_a, b, s_b, sizes)
    check_tensor(op, "qk_scales", qk_scales, (FP32, BF16), 1)
    check_tensor(op, "rope", rope, (FP32,), 5)
    assert head_dim in launcher.HEAD_DIMS, f"{op}: head_dim must be one of {launcher.HEAD_DIMS}, got {head_dim}"
    assert 0 < n_heads <= launcher.MAX_HEADS, f"{op}: n_heads must be in [1, {launcher.MAX_HEADS}], got {n_heads}"
    assert start_elem == 0, f"{op}: start_elem must be 0, got {start_elem}"
    attn_dim = n_heads * head_dim
    assert attn_dim % launcher.QKV_PLANE_COLUMNS == 0, \
        f"{op}: n_heads·head_dim must be a multiple of {launcher.QKV_PLANE_COLUMNS}, got {attn_dim}"
    if sizes:
        assert padded_n == 3 * attn_dim, f"{op}: b must have 3·n_heads·head_dim = {3 * attn_dim} rows, got {padded_n}"
        check_shape(op, "qk_scales", qk_scales, (n_heads,))
        assert rope.shape[4] == 2 and rope.shape[0] * rope.shape[1] == m, \
            f"{op}: rope must be [B, S, ndims, Hrope, 2] with B·S = {m}, got {tuple(rope.shape)}"
        rotated = rope.shape[2] * rope.shape[3] * 2
        assert 0 < rotated <= head_dim, f"{op}: ndims·Hrope·2 must be in [1, {head_dim}], got {rotated}"


def check_mm_atb(a_cw, a_cw_scales, b_cw, b_cw_scales, n, sizes):
    op = "mxfp8_mm_atb"
    check_devices(op, a_cw=a_cw, a_cw_scales=a_cw_scales, b_cw=b_cw, b_cw_scales=b_cw_scales)
    for name, tensor in (("a_cw", a_cw), ("b_cw", b_cw)):
        check_tensor(op, name, tensor, (FP8,), 2)
    for name, tensor in (("a_cw_scales", a_cw_scales), ("b_cw_scales", b_cw_scales)):
        check_tensor(op, name, tensor, (BYTE,), 4)
    k, m = a_cw.shape
    padded_n = b_cw.shape[1]
    width = padded_n if n is None else n
    if sizes:
        assert k > 0 and b_cw.shape[0] == k, f"{op}: b_cw must be [{k}, N] with K > 0, got {tuple(b_cw.shape)}"
        assert m % GROUP == 0 and padded_n % GROUP == 0, f"{op}: M and N must be multiples of {GROUP}, got " \
            f"{m} and {padded_n}"
        k_atoms = packs.ceil_div(k, packs.ATOM_ROWS)
        check_shape(op, "a_cw_scales", a_cw_scales, (k_atoms, m // GROUP, 32, 16))
        check_shape(op, "b_cw_scales", b_cw_scales, (k_atoms, padded_n // GROUP, 32, 16))
        check_width(op, "n", width, padded_n)
    return m, width


def check_block128_operands(op, a, s_a, b, s_b, sizes):
    check_tensor(op, "a", a, (FP8,), 2)
    check_tensor(op, "b", b, (FP8,), 2)
    check_tensor(op, "s_a", s_a, (BYTE, FP32), 4)
    check_tensor(op, "s_b", s_b, (BYTE, FP32), 2)
    assert s_a.dtype == s_b.dtype, f"{op}: s_a and s_b must both be E8M0 bytes or both FP32, got {s_a.dtype} and " \
        f"{s_b.dtype}"
    m, k = a.shape
    n = b.shape[0]
    if sizes:
        assert m > 0 and b.shape[1] == k, f"{op}: a must be [M, K] with M > 0 and b [N, {k}], got {tuple(b.shape)}"
        assert k % GROUP == 0 and n % GROUP == 0, f"{op}: K and N must be multiples of {GROUP}, got {k} and {n}"
        check_shape(op, "s_a", s_a, packs.fp8_b128_activation_scale_shape(m, k))
        check_shape(op, "s_b", s_b, packs.fp8_b128_weight_scale_shape(n, k))
    return m, n, k


def check_block128_mm_abt(a, s_a, b, s_b, sizes):
    op = "fp8_b128_mm_abt"
    check_devices(op, a=a, s_a=s_a, b=b, s_b=s_b)
    check_block128_operands(op, a, s_a, b, s_b, sizes)


def check_block128_mm_glu(a, s_a, w1, s_w1, activation, sizes):
    op = "fp8_b128_mm_glu"
    check_devices(op, a=a, s_a=s_a, w1=w1, s_w1=s_w1)
    _, rows, _ = check_block128_operands(op, a, s_a, w1, s_w1, sizes)
    check_activation(op, activation)
    if sizes:
        assert rows % (2 * GROUP) == 0, f"{op}: w1 must have 2·F rows with F a multiple of {GROUP}, got {rows}"


def check_rowwise_operands(op, a, s_a, b, s_b, sizes):
    check_tensor(op, "a", a, (FP8,), 2)
    check_tensor(op, "b", b, (FP8,), 2)
    check_tensor(op, "s_a", s_a, (FP32,), 2)
    check_tensor(op, "s_b", s_b, (FP32,), 2)
    m, k = a.shape
    n = b.shape[0]
    if sizes:
        assert m > 0 and b.shape[1] == k, f"{op}: a must be [M, K] with M > 0 and b [N, {k}], got {tuple(b.shape)}"
        assert k % launcher.ROWWISE_K_MULTIPLE == 0, \
            f"{op}: K must be a multiple of {launcher.ROWWISE_K_MULTIPLE}, got {k}"
        assert n % launcher.WIDTH_MULTIPLE == 0, f"{op}: N must be a multiple of {launcher.WIDTH_MULTIPLE}, got {n}"
        check_shape(op, "s_a", s_a, (m, 1))
        check_shape(op, "s_b", s_b, (n, 1))
    return m, n, k


def check_rowwise_mm_abt(a, s_a, b, s_b, bias, sizes):
    op = "fp8_rowwise_mm_abt"
    check_devices(op, a=a, s_a=s_a, b=b, s_b=s_b, bias=bias)
    _, n, _ = check_rowwise_operands(op, a, s_a, b, s_b, sizes)
    check_bias(op, bias, n, sizes)


def check_rowwise_mm_glu(a, s_a, w1, s_w1, activation, sizes):
    op = "fp8_rowwise_mm_glu"
    check_devices(op, a=a, s_a=s_a, w1=w1, s_w1=s_w1)
    _, rows, _ = check_rowwise_operands(op, a, s_a, w1, s_w1, sizes)
    check_activation(op, activation)
    if sizes:
        assert rows % (2 * GROUP) == 0, f"{op}: w1 must have 2·F rows with F a multiple of {GROUP}, got {rows}"


def empty(like, shape_and_dtype):
    if shape_and_dtype is None:
        return None
    shape, dtype = shape_and_dtype
    return like.new_empty(shape, dtype=dtype)


def glu_outputs(a, recipe, padded_hidden, hidden, out_z, h_scale_dtype=BYTE):
    return tuple(empty(a, spec) for spec in launcher.glu_shapes(recipe, a.shape[0], padded_hidden, hidden, out_z,
                                                                h_scale_dtype))


def solve(operation, recipe, a, b, m, n, padded_n, k, epilogue, outputs):
    """Describes the product for the kernel of the device of `a`, launches it on the current stream and returns the
    outputs."""
    device_index = a[0].get_device()
    problem = launcher.Problem(operation, recipe, launcher.Pack(*a), launcher.Pack(*b), m, n, padded_n, k, epilogue,
                               outputs, device_index, current_stream_handle(device_index))
    launcher.launch(problem)
    return outputs


def epilogue_of(bias=None, residual=None, low=None, up=None):
    tensors = {name: tensor for name, tensor in (("bias", bias), ("residual", residual), ("low", low), ("up", up))
               if tensor is not None}
    kind = "low_rank" if low is not None else "residual" if residual is not None else \
        "bias" if bias is not None else "store"
    return launcher.Epilogue(kind, tensors)


def mxfp8_mm_abt_fake(a, s_a, b, s_b, bias=None, n=None):
    width = check_mm_abt(a, s_a, b, s_b, bias, n, False)
    return a.new_empty((a.shape[0], width), dtype=BF16)


def mxfp8_mm_abt_cuda(a, s_a, b, s_b, bias=None, n=None):
    width = check_mm_abt(a, s_a, b, s_b, bias, n, True)
    y = a.new_empty((a.shape[0], width), dtype=BF16)
    solve("mm_abt", "mxfp8", (a, s_a), (b, s_b), a.shape[0], width, b.shape[0], a.shape[1], epilogue_of(bias),
          {"y": y})
    return y


def mxfp8_mm_abt_add_fake(a, s_a, b, s_b, residual):
    check_mm_abt_add(a, s_a, b, s_b, residual, False)
    return residual.new_empty(residual.shape, dtype=BF16)


def mxfp8_mm_abt_add_cuda(a, s_a, b, s_b, residual):
    check_mm_abt_add(a, s_a, b, s_b, residual, True)
    y = residual.new_empty(residual.shape, dtype=BF16)
    solve("mm_abt", "mxfp8", (a, s_a), (b, s_b), residual.shape[0], b.shape[0], b.shape[0], a.shape[1],
          epilogue_of(residual=residual), {"y": y})
    return y


def mxfp8_mm_abt_low_rank_fake(a, s_a, b, s_b, low, up, bias=None, n=None):
    width = check_mm_abt_low_rank(a, s_a, b, s_b, low, up, bias, n, False)
    return a.new_empty((a.shape[0], width), dtype=BF16)


def mxfp8_mm_abt_low_rank_cuda(a, s_a, b, s_b, low, up, bias=None, n=None):
    width = check_mm_abt_low_rank(a, s_a, b, s_b, low, up, bias, n, True)
    y = a.new_empty((a.shape[0], width), dtype=BF16)
    solve("mm_abt", "mxfp8", (a, s_a), (b, s_b), a.shape[0], width, b.shape[0], a.shape[1],
          epilogue_of(bias, low=low, up=up), {"y": y})
    return y


def mxfp8_mm_glu_fake(a, s_a, w1, s_w1, gate_first=False, out_z=False, clamp_limit=None, activation="swiglu",
                      hidden=None):
    padded_hidden, hidden = check_mm_glu(a, s_a, w1, s_w1, gate_first, clamp_limit, activation, hidden, False)
    return glu_outputs(a, "mxfp8", padded_hidden, hidden, out_z)


def mxfp8_mm_glu_cuda(a, s_a, w1, s_w1, gate_first=False, out_z=False, clamp_limit=None, activation="swiglu",
                      hidden=None):
    padded_hidden, hidden = check_mm_glu(a, s_a, w1, s_w1, gate_first, clamp_limit, activation, hidden, True)
    z, h, s_h = glu_outputs(a, "mxfp8", padded_hidden, hidden, out_z)
    parameters = {"gate_first": gate_first, "clamp_limit": clamp_limit, "activation": activation, "hidden": hidden}
    solve("mm_glu", "mxfp8", (a, s_a), (w1, s_w1), a.shape[0], 2 * padded_hidden, w1.shape[0], a.shape[1],
          launcher.Epilogue("glu", parameters=parameters), {"z": z, "h": h, "s_h": s_h})
    return z, h, s_h


def mxfp8_mm_qkv_prep_fake(a, s_a, b, s_b, qk_scales, rope, start_elem, eps, n_heads, head_dim, save_qkv):
    check_mm_qkv_prep(a, s_a, b, s_b, qk_scales, rope, start_elem, n_heads, head_dim, False)
    return tuple(a.new_empty(shape, dtype=BF16) for shape in launcher.qkv_shapes(a.shape[0], n_heads, head_dim,
                                                                                 save_qkv))


def mxfp8_mm_qkv_prep_cuda(a, s_a, b, s_b, qk_scales, rope, start_elem, eps, n_heads, head_dim, save_qkv):
    check_mm_qkv_prep(a, s_a, b, s_b, qk_scales, rope, start_elem, n_heads, head_dim, True)
    q, kv, saved_qkv = (a.new_empty(shape, dtype=BF16) for shape in launcher.qkv_shapes(a.shape[0], n_heads,
                                                                                          head_dim, save_qkv))
    parameters = {"start_elem": start_elem, "eps": eps, "n_heads": n_heads, "head_dim": head_dim,
                  "save_qkv": save_qkv}
    epilogue = launcher.Epilogue("qkv_prep", {"qk_scales": qk_scales, "rope": rope}, parameters)
    solve("mm_qkv_prep", "mxfp8", (a, s_a), (b, s_b), a.shape[0], b.shape[0], b.shape[0], a.shape[1], epilogue,
          {"q": q, "kv": kv, "saved_qkv": saved_qkv})
    return q, kv, saved_qkv


def mxfp8_mm_atb_fake(a_cw, a_cw_scales, b_cw, b_cw_scales, n=None):
    m, width = check_mm_atb(a_cw, a_cw_scales, b_cw, b_cw_scales, n, False)
    return a_cw.new_empty((m, width), dtype=BF16)


def mxfp8_mm_atb_cuda(a_cw, a_cw_scales, b_cw, b_cw_scales, n=None):
    m, width = check_mm_atb(a_cw, a_cw_scales, b_cw, b_cw_scales, n, True)
    d = a_cw.new_empty((m, width), dtype=BF16)
    solve("mm_atb", "mxfp8", (a_cw, a_cw_scales), (b_cw, b_cw_scales), m, width, b_cw.shape[1], a_cw.shape[0],
          launcher.Epilogue("store"), {"y": d})
    return d


def fp8_b128_mm_abt_fake(a, s_a, b, s_b):
    check_block128_mm_abt(a, s_a, b, s_b, False)
    return a.new_empty((a.shape[0], b.shape[0]), dtype=BF16)


def fp8_b128_mm_abt_cuda(a, s_a, b, s_b):
    check_block128_mm_abt(a, s_a, b, s_b, True)
    y = a.new_empty((a.shape[0], b.shape[0]), dtype=BF16)
    solve("mm_abt", launcher.fp8_b128_recipe(s_a.dtype), (a, s_a), (b, s_b), a.shape[0], b.shape[0], b.shape[0],
          a.shape[1], launcher.Epilogue("store"), {"y": y})
    return y


def fp8_b128_mm_glu_fake(a, s_a, w1, s_w1, activation, out_z):
    check_block128_mm_glu(a, s_a, w1, s_w1, activation, False)
    hidden = w1.shape[0] // 2
    return glu_outputs(a, "fp8_b128", hidden, hidden, out_z, s_a.dtype)


def fp8_b128_mm_glu_cuda(a, s_a, w1, s_w1, activation, out_z):
    check_block128_mm_glu(a, s_a, w1, s_w1, activation, True)
    hidden = w1.shape[0] // 2
    z, h, s_h = glu_outputs(a, "fp8_b128", hidden, hidden, out_z, s_a.dtype)
    parameters = {"gate_first": activation == "geglu_tanh", "clamp_limit": None, "activation": activation,
                  "hidden": hidden}
    solve("mm_glu", launcher.fp8_b128_recipe(s_a.dtype), (a, s_a), (w1, s_w1), a.shape[0], w1.shape[0], w1.shape[0],
          a.shape[1], launcher.Epilogue("glu", parameters=parameters), {"z": z, "h": h, "s_h": s_h})
    return z, h, s_h


def fp8_rowwise_mm_abt_fake(a, s_a, b, s_b, bias):
    check_rowwise_mm_abt(a, s_a, b, s_b, bias, False)
    return a.new_empty((a.shape[0], b.shape[0]), dtype=BF16)


def fp8_rowwise_mm_abt_cuda(a, s_a, b, s_b, bias):
    check_rowwise_mm_abt(a, s_a, b, s_b, bias, True)
    y = a.new_empty((a.shape[0], b.shape[0]), dtype=BF16)
    solve("mm_abt", "fp8_rowwise", (a, s_a), (b, s_b), a.shape[0], b.shape[0], b.shape[0], a.shape[1], epilogue_of(bias),
          {"y": y})
    return y


def fp8_rowwise_mm_glu_fake(a, s_a, w1, s_w1, activation, out_z):
    check_rowwise_mm_glu(a, s_a, w1, s_w1, activation, False)
    hidden = w1.shape[0] // 2
    return glu_outputs(a, "fp8_rowwise", hidden, hidden, out_z)


def fp8_rowwise_mm_glu_cuda(a, s_a, w1, s_w1, activation, out_z):
    check_rowwise_mm_glu(a, s_a, w1, s_w1, activation, True)
    hidden = w1.shape[0] // 2
    z, h, s_h = glu_outputs(a, "fp8_rowwise", hidden, hidden, out_z)
    parameters = {"gate_first": activation == "geglu_tanh", "clamp_limit": None, "activation": activation,
                  "hidden": hidden}
    solve("mm_glu", "fp8_rowwise", (a, s_a), (w1, s_w1), a.shape[0], w1.shape[0], w1.shape[0], a.shape[1],
          launcher.Epilogue("glu", parameters=parameters), {"z": z, "h": h, "s_h": s_h})
    return z, h, s_h


def register():
    library = library_of(NAMESPACE)
    operators = getattr(torch.ops, NAMESPACE)
    for name, schema in SCHEMAS.items():
        library.define(name + schema, tags=(torch.Tag.pt2_compliant_tag,))
        library.impl(name, globals()[f"{name}_cuda"], "CUDA")
        torch.library.register_fake(f"{NAMESPACE}::{name}", globals()[f"{name}_fake"], lib=library)
        operator = getattr(operators, name).default
        library.impl(name, make_autograd_impl(operator, WITHOUT_FORMULA), "Autograd", with_keyset=True)
        globals()[name] = globals()[f"{name}_op"] = operator


register()
__all__ = [*SCHEMAS, *(f"{name}_op" for name in SCHEMAS)]
