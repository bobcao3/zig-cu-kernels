import pytest
import torch

import fp8_mm_references as references
import pack_references
from opcheck import assert_symbolic, call_symbolic, check_op
from zig_cu_kernels import fp8_mm as launcher
from zig_cu_kernels import packs, roofline

CUDA = torch.cuda.is_available()
if CUDA:
    from zig_cu_kernels.torch import fp8_mm as ops

needs_cuda = pytest.mark.skipif(not CUDA, reason="no CUDA device")
FP8 = torch.float8_e4m3fn
GAMMA = 2.0**-9
ROWWISE_FROBENIUS = 2.0**-8
BF16_UNIT = 2.0**-8
QKV_NOISE_RATE = 1e-3
QKV_ROW_RELATIVE = 2.0**-6


def generator(seed):
    return torch.Generator().manual_seed(seed)


def integers(shape, seed, bound=4):
    return torch.randint(-bound, bound + 1, shape, generator=generator(seed)).float()


def normal(shape, seed):
    return torch.randn(shape, generator=generator(seed))


def mxfp8_pack(x, device="cpu"):
    values, scales = pack_references.mxfp8_quant(x)
    return values.to(device), scales.to(device)


def fp8_b128_packs(a, b, e8m0, device="cpu"):
    a_pack = pack_references.fp8_b128_quant(a, e8m0=e8m0)
    b_pack = pack_references.fp8_b128_quant_weight(b, e8m0=e8m0)
    return [t.to(device) for t in (*a_pack, *b_pack)]


def fp8_rowwise_packs(a, b, device="cpu"):
    return [t.to(device) for t in (*pack_references.fp8_rowwise_quant(a), *pack_references.fp8_rowwise_quant(b))]


def require_kernel(recipe, operation):
    if not CUDA:
        pytest.skip("no CUDA device")
    if not launcher.available(recipe, operation):
        pytest.skip(f"no {recipe} {operation} kernel for this device")


def assert_within_bound(y, a, b, *terms):
    exact, magnitude = references.product(a, b)
    for term in terms:
        exact = exact + term.double()
    bound = GAMMA * magnitude + BF16_UNIT * exact.abs() + 1e-30
    assert ((y.cpu().double() - exact).abs() <= bound).all()


def test_dequantized_is_exact_and_matches_the_packs():
    values, scales = mxfp8_pack(normal((130, 640), 1))
    assert torch.equal(references.dequantized(values, scales, "mxfp8").float(),
                       packs.dequantize(values, scales, "mxfp8"))
    values, scales = pack_references.fp8_b128_quant(normal((7, 256), 2), e8m0=False)
    expected = values.double() * packs.scale_grid(scales, 7, 2).double().repeat_interleave(128, dim=1)
    assert torch.equal(references.dequantized(values, scales, "fp8_b128"), expected)


def test_columnwise_dequantization_transposes_the_rowwise_pack():
    x = normal((256, 300), 3)
    values, scales = pack_references.mxfp8_quant(x.t().contiguous())
    columnwise = references.dequantized_columnwise(values.t()[:300].contiguous(), scales.permute(1, 0, 2, 3))
    assert torch.equal(columnwise, references.dequantized(values, scales, "mxfp8")[:, :300].t())


def test_bf16_store_adds_from_left_to_right():
    acc = torch.tensor([1.0, 2.0**20])
    c = torch.tensor([2.0**-9, 1.0]).bfloat16()
    bias = torch.tensor([2.0**-9, -(2.0**20)]).bfloat16()
    stored = references.bf16_store(acc, c, bias)
    assert stored.tolist() == [((acc + c.float()) + bias.float()).bfloat16()[i].item() for i in range(2)]
    assert torch.equal(references.bf16_store(acc, None, None), acc.bfloat16())


@pytest.mark.parametrize("activation, gate_first", [("swiglu", False), ("swiglu", True), ("geglu_tanh", True)])
def test_glu_bounds_hold_the_float64_activation(activation, gate_first):
    z = (normal((64, 512), 4) * 4).bfloat16()
    low, high = references.glu(z, activation, gate_first)
    up, gate = references.halves(z, gate_first)
    gate64 = gate.double()
    exact = gate64 * torch.sigmoid(gate64) if activation == "swiglu" else \
        torch.nn.functional.gelu(gate64, approximate="tanh")
    nominal = (up.double() * exact.float().bfloat16().double()).bfloat16().double()
    spacing = BF16_UNIT * nominal.abs()
    tame = gate > -2
    assert ((low.double() <= nominal + spacing) & (nominal - spacing <= high.double()))[tame].all()
    assert (low == high).float().mean() > 0.99


def test_swiglu_clamps_both_halves():
    z = torch.tensor([[32.0, -32.0, 32.0, 32.0]]).bfloat16()
    low, high = references.glu(z, "swiglu", False, clamp_limit=7.0)
    assert torch.equal(low, high)
    silu = (torch.tensor(7.0) * torch.sigmoid(torch.tensor(7.0))).bfloat16().float()
    assert low.float().tolist()[0] == [(7 * silu).bfloat16().item(), (-7 * silu).bfloat16().item()]


def test_glu_pack_zeroes_the_padding_and_marks_agreement():
    z = (normal((5, 512), 5) * 2).bfloat16()
    (values, scales), _, agree = references.glu_pack(z, "mxfp8", "swiglu", False, hidden=200)
    assert values.shape == (5, 256) and scales.shape == packs.mxfp8_scale_shape(5, 256)
    assert (values[:, 200:].float() == 0).all()
    assert agree.shape == (5, 8) and agree[:, 7].all()
    assert torch.equal(references.glu_z(z, 200), torch.cat([z[:, :200], z[:, 256:456]], dim=1))


def test_qkv_prep_reference():
    m, heads, dim = 6, 2, 128
    y = normal((m, 3 * heads * dim), 6).bfloat16()
    scales = torch.tensor([1.0, 0.5])
    turn = torch.zeros(1, m, 1, 32, 2)
    turn[..., 0] = 1
    q, kv, saved = references.qkv_prep(y, scales, turn, 1e-6, heads, dim, True)
    assert torch.equal(saved, y) and torch.equal(kv[:, heads * dim:], y[:, 2 * heads * dim:])
    x = y[:, : heads * dim].view(m, heads, dim).double()
    rms = (x * x).mean(-1, keepdim=True).add(1e-6).rsqrt() * scales.double()[:, None]
    torch.testing.assert_close(q.view(m, heads, dim).double(), x * rms, rtol=2**-7, atol=0)
    swapped = torch.zeros_like(turn)
    swapped[..., 1] = 1
    turned, _, empty = references.qkv_prep(y, scales, swapped, 1e-6, heads, dim, False)
    pairs = q.view(m, heads, dim)[..., :64].unflatten(-1, (-1, 2))
    rotated = turned.view(m, heads, dim)[..., :64].unflatten(-1, (-1, 2))
    assert torch.equal(rotated[..., 0], -pairs[..., 1]) and torch.equal(rotated[..., 1], pairs[..., 0])
    assert empty.shape == (0, 3 * heads * dim)


def test_dispatch_names_the_missing_kernel():
    with pytest.raises(NotImplementedError, match="fp8_rowwise_mm_atb_sm100 does not exist for compute capability 10.3"):
        launcher.lookup(103, "fp8_rowwise", "mm_atb")
    assert launcher.arch_of(103) == 100 and launcher.arch_of(121) == 120 and launcher.arch_of(80) is None
    launcher.register(90, "fp8_rowwise", "mm_abt", print)
    try:
        assert launcher.lookup(90, "fp8_rowwise", "mm_abt") is print
        with pytest.raises(NotImplementedError, match="fp8_rowwise_mm_abt_sm91 does not exist"):
            launcher.lookup(91, "fp8_rowwise", "mm_abt")
    finally:
        del launcher.kernels[(90, "fp8_rowwise")]


def mxfp8_arguments(m=300, n=512, k=256, device="cuda", seed=7):
    a, s_a = mxfp8_pack(normal((m, k), seed), device)
    b, s_b = mxfp8_pack(normal((n, k), seed + 1), device)
    return a, s_a, b, s_b


def qkv_arguments(m=192, heads=2, dim=128, device="cuda"):
    a, s_a, b, s_b = mxfp8_arguments(m, 3 * heads * dim, 256, device)
    rope = normal((1, m, 1, 32, 2), 9).to(device)
    return a, s_a, b, s_b, torch.rand(heads, generator=generator(10)).add(0.5).to(device), rope, 0, 1e-6, heads, dim


def op_cases(device="cuda"):
    a, s_a, b, s_b = mxfp8_arguments(device=device)
    bias = normal((504,), 11).bfloat16().to(device)
    low, up = normal((300, 32), 12).bfloat16().to(device), normal((504, 32), 13).bfloat16().to(device)
    residual = normal((290, 512), 14).bfloat16().to(device)
    x, w = normal((300, 256), 15), normal((512, 256), 16)
    block = fp8_b128_packs(x, w, True, device)
    block_fp32 = fp8_b128_packs(x, w, False, device)
    rowwise = fp8_rowwise_packs(x, w, device)
    a_cw, s_cw = mxfp8_pack(normal((256, 200), 17), device)
    b_cw, t_cw = mxfp8_pack(normal((384, 200), 18), device)
    columnwise = (a_cw.t()[:200].contiguous(), s_cw.permute(1, 0, 2, 3).contiguous(),
                  b_cw.t()[:200].contiguous(), t_cw.permute(1, 0, 2, 3).contiguous())
    return {
        "mxfp8_mm_abt": (("mxfp8", "mm_abt"), (a, s_a, b, s_b, bias, 504), ((300, 504),)),
        "mxfp8_mm_abt_add": (("mxfp8", "mm_abt"), (a, s_a, b, s_b, residual), ((290, 512),)),
        "mxfp8_mm_abt_low_rank": (("mxfp8", "mm_abt"), (a, s_a, b, s_b, low, up, None, 504), ((300, 504),)),
        "mxfp8_mm_glu": (("mxfp8", "mm_glu"), (a, s_a, b, s_b, False, True, 7.0, "swiglu", 200),
                         ((300, 400), (300, 256), (3, 2, 32, 16))),
        "mxfp8_mm_qkv_prep": (("mxfp8", "mm_qkv_prep"), (*qkv_arguments(device=device), True),
                              ((192, 256), (192, 512), (192, 768))),
        "mxfp8_mm_atb": (("mxfp8", "mm_atb"), (*columnwise, 376), ((256, 376),)),
        "fp8_b128_mm_abt": (("fp8_b128_e8m0", "mm_abt"), tuple(block), ((300, 512),)),
        "fp8_b128_mm_abt_fp32": (("fp8_b128_fp32", "mm_abt"), tuple(block_fp32), ((300, 512),)),
        "fp8_b128_mm_glu": (("fp8_b128_e8m0", "mm_glu"), (*block, "swiglu", False),
                            (None, (300, 256), (3, 1, 32, 16))),
        "fp8_rowwise_mm_abt": (("fp8_rowwise", "mm_abt"), (*rowwise, None), ((300, 512),)),
        "fp8_rowwise_mm_glu": (("fp8_rowwise", "mm_glu"), (*rowwise, "geglu_tanh", True), ((300, 512), (300, 256), (300, 1))),
    }


def op_of(name):
    return getattr(ops, name.removesuffix("_fp32"))


def shapes_of(result):
    results = result if isinstance(result, tuple) else (result,)
    return tuple(None if tensor is None else tuple(tensor.shape) for tensor in results)


@needs_cuda
@pytest.mark.parametrize("name", sorted(op_cases("meta")))
def test_fake_results_have_the_contract_shapes(name):
    _, arguments, shapes = op_cases("meta")[name]
    result = op_of(name)(*arguments)
    assert shapes_of(result) == shapes
    dtypes = [tensor.dtype for tensor in (result if isinstance(result, tuple) else (result,)) if tensor is not None]
    assert dtypes[0] in (torch.bfloat16, FP8)


@needs_cuda
@pytest.mark.parametrize("name", ["mxfp8_mm_abt", "mxfp8_mm_glu", "mxfp8_mm_qkv_prep", "fp8_rowwise_mm_glu",
                                  "fp8_b128_mm_abt"])
def test_fake_results_keep_symbolic_rows(name):
    _, arguments, _ = op_cases("cpu")[name]
    arguments = tuple(argument.cuda() if isinstance(argument, torch.Tensor) else argument for argument in arguments)
    static = {index: range(argument.dim()) for index, argument in enumerate(arguments)
              if isinstance(argument, torch.Tensor) and index not in (0, 1)}
    static[1] = range(arguments[1].dim())
    result = call_symbolic(op_of(name), *arguments, static_dims=static)
    first = result[1] if isinstance(result, tuple) else result
    assert_symbolic(first, (0,))


@needs_cuda
@pytest.mark.parametrize("name", sorted(op_cases("meta")))
def test_ops_without_a_kernel_raise_not_implemented(name):
    (recipe, operation), arguments, _ = op_cases()[name]
    if launcher.available(recipe, operation):
        pytest.skip("the kernel exists")
    with pytest.raises(NotImplementedError, match=f"{recipe}_{operation}_sm"):
        op_of(name)(*arguments)


@needs_cuda
@pytest.mark.parametrize("change, message", [
    (lambda a: (a[0].float(), *a[1:]), "a must have a dtype"),
    (lambda a: (a[0], a[1][:1], *a[2:]), "s_a must have the shape"),
    (lambda a: (*a[:4], a[4], 500), "n must be a positive multiple of 8 up to 512"),
    (lambda a: (*a[:4], a[4][:8], 504), "bias must have the shape"),
])
def test_mxfp8_mm_abt_checks_its_contract(change, message):
    _, arguments, _ = op_cases()["mxfp8_mm_abt"]
    with pytest.raises(AssertionError, match=message):
        ops.mxfp8_mm_abt(*change(arguments))


@needs_cuda
def test_glu_and_qkv_prep_check_their_contracts():
    a, s_a, b, s_b = mxfp8_arguments()
    with pytest.raises(AssertionError, match="geglu_tanh takes the half order"):
        ops.mxfp8_mm_glu(a, s_a, b, s_b, False, False, None, "geglu_tanh")
    with pytest.raises(AssertionError, match="hidden must round up"):
        ops.mxfp8_mm_glu(a, s_a, b, s_b, False, False, None, "swiglu", 64)
    arguments = [*qkv_arguments(), True]
    arguments[9] = 96
    with pytest.raises(AssertionError, match="head_dim must be one of"):
        ops.mxfp8_mm_qkv_prep(*arguments)
    block = fp8_b128_packs(normal((300, 256), 1), normal((512, 256), 2), True, "cuda")
    block[3] = block[3].float()
    with pytest.raises(AssertionError, match="both be E8M0 bytes or both FP32"):
        ops.fp8_b128_mm_abt(*block)


@needs_cuda
@pytest.mark.parametrize("name", sorted(op_cases("meta")))
def test_ops_pass_opcheck(name):
    (recipe, operation), arguments, _ = op_cases()[name]
    require_kernel(recipe, operation)
    check_op(op_of(name), *arguments)


@pytest.mark.parametrize("recipe", ["mxfp8", "fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise"])
@pytest.mark.parametrize("m, n, k", [(1, 128, 128), (300, 512, 640), (1024, 1024, 2048)])
def test_mm_abt_within_the_error_bound(recipe, m, n, k):
    require_kernel(recipe, "mm_abt")
    x, w = normal((m, k), m), normal((n, k), n) * 0.05
    if recipe == "mxfp8":
        a, s_a, b, s_b = (*mxfp8_pack(x, "cuda"), *mxfp8_pack(w, "cuda"))
        y = ops.mxfp8_mm_abt(a, s_a, b, s_b)
    elif recipe == "fp8_rowwise":
        a, s_a, b, s_b = fp8_rowwise_packs(x, w, "cuda")
        y = ops.fp8_rowwise_mm_abt(a, s_a, b, s_b, None)
    else:
        a, s_a, b, s_b = fp8_b128_packs(x, w, recipe.endswith("e8m0"), "cuda")
        y = ops.fp8_b128_mm_abt(a, s_a, b, s_b)
    kind = "fp8_rowwise" if recipe == "fp8_rowwise" else "fp8_b128" if recipe != "mxfp8" else "mxfp8"
    a64, b64 = references.dequantized(a, s_a, kind), references.dequantized(b, s_b, kind)
    if recipe == "fp8_rowwise" and torch.cuda.get_device_capability()[0] == 9:
        exact = references.product(a64, b64)[0]
        assert torch.linalg.norm(y.cpu().double() - exact) <= ROWWISE_FROBENIUS * torch.linalg.norm(exact)
        return
    assert_within_bound(y, a64, b64)


@pytest.mark.parametrize("epilogue", ["bias", "residual", "low_rank"])
def test_mxfp8_epilogues_are_bitwise_with_integer_operands(epilogue):
    require_kernel("mxfp8", "mm_abt")
    m, n, k = 260, 384, 512
    a, s_a = mxfp8_pack(integers((m, k), 1), "cuda")
    b, s_b = mxfp8_pack(integers((n, k), 2), "cuda")
    acc = references.product(references.dequantized(a, s_a, "mxfp8"), references.dequantized(b, s_b, "mxfp8"))[0]
    bias = integers((n,), 3).bfloat16()
    if epilogue == "bias":
        y = ops.mxfp8_mm_abt(a, s_a, b, s_b, bias.cuda())
        expected = references.bf16_store(acc.float(), None, bias)
    elif epilogue == "residual":
        residual = (normal((200, n), 4) * 64).bfloat16()
        y = ops.mxfp8_mm_abt_add(a, s_a, b, s_b, residual.cuda())
        expected = references.bf16_store(acc[:200].float(), residual)
    else:
        low, up = integers((m, 64), 5).bfloat16(), integers((n, 64), 6).bfloat16()
        y = ops.mxfp8_mm_abt_low_rank(a, s_a, b, s_b, low.cuda(), up.cuda(), bias.cuda())
        expected = references.bf16_store((acc + low.double() @ up.double().t()).float(), None, bias)
    assert torch.equal(y.cpu(), expected)


def test_mxfp8_mm_atb_within_the_error_bound():
    require_kernel("mxfp8", "mm_atb")
    tokens, m, n = 300, 256, 384
    a_rows, s_rows = mxfp8_pack(normal((m, tokens), 1))
    b_rows, t_rows = mxfp8_pack(normal((n, tokens), 2))
    a_cw, s_cw = a_rows.t()[:tokens].contiguous(), s_rows.permute(1, 0, 2, 3).contiguous()
    b_cw, t_cw = b_rows.t()[:tokens].contiguous(), t_rows.permute(1, 0, 2, 3).contiguous()
    d = ops.mxfp8_mm_atb(a_cw.cuda(), s_cw.cuda(), b_cw.cuda(), t_cw.cuda(), 320)
    a64, b64 = references.dequantized_columnwise(a_cw, s_cw), references.dequantized_columnwise(b_cw, t_cw)
    assert_within_bound(d, a64.t(), b64.t()[:320])


@pytest.mark.parametrize("recipe", ["mxfp8", "fp8_b128_e8m0", "fp8_b128_fp32", "fp8_rowwise"])
@pytest.mark.parametrize("activation, gate_first, clamp_limit", [("swiglu", False, None), ("swiglu", True, 7.0),
                                                                 ("geglu_tanh", True, None)])
def test_glu_h_packs_match_the_unfused_chain(recipe, activation, gate_first, clamp_limit):
    require_kernel(recipe, "mm_glu")
    if recipe != "mxfp8" and (clamp_limit is not None or gate_first != (activation == "geglu_tanh")):
        pytest.skip("the block128 and rowwise GLU take the canonical half order and no clamp")
    m, hidden, k = 300, 256, 512
    x, w = normal((m, k), 1), normal((2 * hidden, k), 2) * 0.05
    if recipe == "mxfp8":
        a, s_a, w1, s_w1 = (*mxfp8_pack(x, "cuda"), *mxfp8_pack(w, "cuda"))
        z, h, s_h = ops.mxfp8_mm_glu(a, s_a, w1, s_w1, gate_first, True, clamp_limit, activation, hidden)
        z_full = ops.mxfp8_mm_abt(a, s_a, w1, s_w1)
    elif recipe == "fp8_rowwise":
        a, s_a, w1, s_w1 = fp8_rowwise_packs(x, w, "cuda")
        z, h, s_h = ops.fp8_rowwise_mm_glu(a, s_a, w1, s_w1, activation, True)
        z_full = ops.fp8_rowwise_mm_abt(a, s_a, w1, s_w1, None)
    else:
        a, s_a, w1, s_w1 = fp8_b128_packs(x, w, recipe.endswith("e8m0"), "cuda")
        z, h, s_h = ops.fp8_b128_mm_glu(a, s_a, w1, s_w1, activation, True)
        z_full = ops.fp8_b128_mm_abt(a, s_a, w1, s_w1)
    assert torch.equal(z, z_full)
    kind = "fp8_b128" if recipe.startswith("fp8_b128") else recipe
    (values, scales), _, agree = references.glu_pack(z_full, kind, activation, gate_first, clamp_limit,
                                                     e8m0=recipe != "fp8_b128_fp32")
    block = values.shape[1] // agree.shape[1]
    columns = agree.repeat_interleave(block, dim=1)
    assert torch.equal(h.cpu().view(torch.uint8)[columns], values.view(torch.uint8)[columns])
    if kind == "fp8_rowwise":
        assert torch.equal(s_h.cpu()[agree], scales[agree])


def test_qkv_prep_matches_the_unfused_chain():
    require_kernel("mxfp8", "mm_qkv_prep")
    arguments = qkv_arguments()
    a, s_a, b, s_b, scales, rope, _, eps, heads, dim = arguments
    q, kv, saved = ops.mxfp8_mm_qkv_prep(*arguments, True)
    y = ops.mxfp8_mm_abt(a, s_a, b, s_b)
    shifts = (-references.RSQRT_RELATIVE_ERROR, 0.0, references.RSQRT_RELATIVE_ERROR)
    q_refs, kv_refs, saved_refs = zip(*(references.qkv_prep(y, scales, rope, eps, heads, dim, True, shift)
                                        for shift in shifts))
    assert torch.equal(saved.cpu(), saved_refs[1])
    attn_dim = heads * dim
    assert torch.equal(kv.cpu()[:, attn_dim:], kv_refs[1][:, attn_dim:])
    for result, expected in ((q.cpu(), q_refs), (kv.cpu()[:, :attn_dim], [ref[:, :attn_dim] for ref in kv_refs])):
        unexplained = torch.stack([result != ref for ref in expected]).all(0)
        assert unexplained.float().mean() <= QKV_NOISE_RATE
        rows = expected[1].float().view(-1, heads, dim)
        difference = (result.float().view(-1, heads, dim) - rows).abs()
        assert (difference <= QKV_ROW_RELATIVE * rows.abs().amax(-1, keepdim=True)).all()
    q_without, kv_without, empty = ops.mxfp8_mm_qkv_prep(*arguments, False)
    assert torch.equal(q_without, q) and torch.equal(kv_without, kv) and empty.shape == (0, 3 * attn_dim)


@pytest.mark.parametrize("capability, bf16, e4m3", [(90, 4096, 8192), (100, 8192, 16384), (103, 8192, 16384),
                                                    (120, 1024, 2048)])
def test_roofline_knows_the_tensor_rates(capability, bf16, e4m3):
    assert roofline.table_rates(capability) == {"bf16": bf16, "e4m3": e4m3}
