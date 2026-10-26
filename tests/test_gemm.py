import math
import threading

import pytest
import torch
import torch.nn.functional as F
from cuda.bindings import driver

from checks import case, queue, queue_equal_check, queue_requirement
from conftest import cuda_seed, declare_gpu_work
from opcheck import (
    assert_symbolic,
    call_symbolic,
    check_compiled,
    check_graph_replay,
    check_op,
    fresh_compiler,  # noqa: F401
)
from zig_cu_kernels import gemm as launcher
from zig_cu_kernels import runtime
from zig_cu_kernels.torch import gated_gemm, gated_gemm_op, gated_gemm_out_op, gemm, gemm_op, gemm_out_op

UNAVAILABLE = runtime.unavailable_reason("gemm_store_bf16_m8")
pytestmark = pytest.mark.skipif(UNAVAILABLE is not None, reason=UNAVAILABLE or "")

ACTIVATIONS = {
    "none": lambda x: x,
    "silu": F.silu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu": F.gelu,
}
NAMES = tuple(ACTIVATIONS)
KINDS = ("store_bf16", "store_f32", "gated")
RASTERS = ((1, 0), (3, 0), (2, 1), (1000, 1))
EXACT_SHAPE = (200, 264, 1000)
FEW_ROWS = {8: (1, 7, 8), 32: (9, 20, 32)}
ALL_BF16 = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
EPILOGUE_ROWS = 4
RANDOM_SHAPES = ((1, 256, 512), (7, 264, 200), (31, 136, 1000), (77, 200, 136), (300, 384, 1000), (64, 128, 16384),
                 (130, 264, 8192), (1020, 1024, 256))
SLOPE = 1.2
SPLIT_SHAPE = (64, 256, 8192)
COMPILED_SIZES = ((40, 24, 16), (72, 40, 32), (24, 56, 48))
LAYOUT_ROWS = (6, 30, 150)
POISON = 0x55555555


def stream():
    return torch.cuda.current_stream().cuda_stream


def device_id():
    return runtime.current_device().device_id


def randn(*shape, scale=1.0):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale


def exact_values(*shape):
    """Returns multiples of 1/16 in [-1/2, 1/2], so that FP32 holds every sum of fewer than 2^18 of their products."""
    return (torch.randint(-8, 9, shape, device="cuda") / 16).bfloat16()


def product_work(m, columns, k, kind="store_bf16", bias=True, residual=True):
    """Returns the FLOPs of one launch and its DRAM bytes when every operand moves through DRAM once."""
    rows_of_b = 2 * columns if kind == "gated" else columns
    out_bytes = 4 if kind == "store_f32" else 2
    dram = 2 * (m + rows_of_b) * k + 2 * columns * bias + 2 * m * columns * residual + out_bytes * m * columns
    return 2 * m * rows_of_b * k, dram


def declare_products(launches, m, columns, k, kind="store_bf16"):
    flops, dram_bytes = product_work(m, columns, k, kind)
    declare_gpu_work(launches, flops=flops, dram_bytes=dram_bytes)


def sums_of(a, w):
    """Returns the exact sums `s*` in FP64, where every product of two BF16 values is exact."""
    return a.double().reshape(math.prod(a.shape[:-1]), a.shape[-1]) @ w.double().T


def biased(sums, bias):
    """Returns `t = fp32(s*) + bias` in FP32. A missing bias adds zero."""
    return sums.float() + (0.0 if bias is None else bias.float())


def store_reference(sums, bias, residual, activation, out_dtype=torch.bfloat16):
    t = biased(sums, bias)
    if out_dtype == torch.float32:
        return t, t, t
    activated = ACTIVATIONS[activation](t.bfloat16())
    out = activated if residual is None else activated + residual.reshape(activated.shape)
    return out, t, activated


def gated_reference(gate_sums, up_sums, bias_gate, bias_up, residual, activation):
    g, v = biased(gate_sums, bias_gate).bfloat16(), biased(up_sums, bias_up).bfloat16()
    activated = ACTIVATIONS[activation](g)
    out = activated * v
    return (out if residual is None else out + residual.reshape(out.shape)), g, v, activated


def reference(kind, a, b, w_up, bias, bias_up, residual, activation):
    if kind == "gated":
        return gated_reference(sums_of(a, b), sums_of(a, w_up), bias, bias_up, residual, activation)[0]
    out_dtype = torch.float32 if kind == "store_f32" else torch.bfloat16
    return store_reference(sums_of(a, b), bias, residual, activation, out_dtype)[0]


def launch(kind, a, b, w_up, out, bias, bias_up, residual, activation, **plan):
    if kind == "gated":
        launcher.gated_gemm(a, b, w_up, out, bias_gate=bias, bias_up=bias_up, residual=residual,
                            activation=activation, stream=stream(), **plan)
    else:
        launcher.gemm(a, b, out, bias=bias, residual=residual, activation=activation, stream=stream(), **plan)


def output_dtype(kind):
    return torch.float32 if kind == "store_f32" else torch.bfloat16


def tiled(kind):
    return [info for info in launcher.tilings(kind) if not launcher.is_few_rows(info)]


def few_rows(kind):
    return [info for info in launcher.tilings(kind) if launcher.is_few_rows(info)]


def epilogue_cases(kind, index):
    if kind == "store_f32":
        return "none", False
    return NAMES[index % len(NAMES)], index % 2 == 1


def test_exact_sums_are_bitwise_on_every_plan():
    """Every tiled kernel with 1 to 8 parts of k and several rasters, and every kernel for few rows, gives the
    reference bits when FP32 holds every partial sum. The shape has tails in m, n and k for every tiling."""
    cuda_seed(1)
    m, n, k = EXACT_SHAPE
    a, b, w_up = exact_values(m, k), exact_values(n, k), exact_values(n, k)
    bias, bias_up, residual = randn(n), randn(n), randn(m, n, scale=4.0)
    declare_products(400, m, n, k, "gated")
    references = {}

    def expected(kind, activation, with_residual, rows):
        key = kind, activation, with_residual
        if key not in references:
            references[key] = reference(kind, a, b, w_up, bias, bias_up, residual if with_residual else None,
                                        activation)
        return references[key][:rows]

    for kind in KINDS:
        plans = [(info, splits, None) for info in tiled(kind) for splits in range(1, info.max_splits + 1)]
        plans += [(info, splits, raster) for info in tiled(kind) for splits in (1, 2) for raster in RASTERS]
        for index, (info, splits, raster) in enumerate(plans):
            activation, with_residual = epilogue_cases(kind, index)
            with case(kernel=info.kernel, splits=splits, raster=raster, activation=activation,
                      residual=with_residual):
                out = torch.empty(m, n, device="cuda", dtype=output_dtype(kind))
                launch(kind, a, b, w_up, out, bias, bias_up, residual if with_residual else None, activation,
                       tiling=launcher.tiling_name(info), splits=splits, raster=raster)
                queue_equal_check(out, expected(kind, activation, with_residual, m), "out")
        for info in few_rows(kind):
            for index, rows in enumerate(FEW_ROWS[info.tile_m] * len(NAMES)):
                activation, with_residual = epilogue_cases(kind, index)
                with case(kernel=info.kernel, rows=rows, activation=activation, residual=with_residual):
                    out = torch.empty(rows, n, device="cuda", dtype=output_dtype(kind))
                    launch(kind, a[:rows], b, w_up, out, bias, bias_up,
                           residual[:rows] if with_residual else None, activation, tiling=launcher.tiling_name(info))
                    queue_equal_check(out, expected(kind, activation, with_residual, rows), "out")


def padded(tensor, before=8, after=16):
    holder = torch.full((*tensor.shape[:-1], before + tensor.shape[-1] + after), float("nan"), device="cuda",
                        dtype=tensor.dtype)
    view = holder[..., before : before + tensor.shape[-1]]
    view.copy_(tensor)
    return view


def shifted(tensor):
    """Returns a copy of `tensor` that starts one element past a multiple of 16 bytes."""
    holder = torch.empty(tensor.numel() + 1, device="cuda", dtype=tensor.dtype)
    view = holder[1:].view(tensor.shape)
    view.copy_(tensor)
    return view


@pytest.mark.parametrize("m", LAYOUT_ROWS)
def test_operand_layouts_are_bitwise(m):
    """Strided, misaligned, batched, transposed, repeated and overlapping operands with exact FP32 partial sums give
    the reference bits on the few-rows and tiled kernels, whether the ops read the operands in place or copy them."""
    cuda_seed(2)
    n, k = 200, 264
    for kind in KINDS:
        assert launcher.is_few_rows(launcher.plan(device_id(), kind, m, n, k).info) == (m <= max(FEW_ROWS))
    a, b, w_up = exact_values(m, k), exact_values(n, k), exact_values(n, k)
    bias, residual = randn(n), randn(m, n, scale=4.0)
    repeated = residual[:1].expand(m, n)
    declare_products(80, m, 2 * n, k, "gated")
    expected = reference("store_bf16", a, b, None, bias, None, residual, "gelu")
    gated_expected = reference("gated", a, b, w_up, None, bias, residual, "silu")

    def check_gemm(label, result, reference_bits=expected):
        queue_equal_check(result.reshape(reference_bits.shape), reference_bits, label)

    check_gemm("padded rows", gemm(padded(a), padded(b, 0, 40), bias=bias, activation="gelu",
                                   residual=padded(residual)))
    check_gemm("rows 8 bytes past a multiple of 16 bytes apart",
               gemm(padded(a, 8, 4), padded(b, 8, 4), bias=bias, activation="gelu", residual=padded(residual, 8, 4)))
    check_gemm("misaligned operands", gemm(shifted(a), b.T.contiguous().T, bias=shifted(bias), activation="gelu",
                                           residual=shifted(residual)))
    every_other = torch.zeros(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    every_other[:, ::2] = a
    check_gemm("every second column", gemm(every_other[:, ::2], b, bias=bias, activation="gelu", residual=residual))
    check_gemm("batch dimensions", gemm(a.view(3, m // 3, k), b, bias=bias, activation="gelu",
                                        residual=residual.view(3, m // 3, n)))
    check_gemm("rows of two strides", gemm(a.view(m // 3, 3, k).transpose(0, 1), b, bias=bias, activation="gelu",
                                           residual=residual.view(m // 3, 3, n).transpose(0, 1)),
               expected.view(m // 3, 3, n).transpose(0, 1))
    middle_rows = torch.stack([a, a, a], dim=1)[:, 1:2]
    check_gemm("dimension of size 1", gemm(middle_rows, b, bias=bias, activation="gelu",
                                           residual=residual.view(m, 1, n)))
    check_gemm("repeated residual row", gemm(a, b, bias=bias, activation="gelu", residual=repeated),
               reference("store_bf16", a, b, None, bias, None, repeated, "gelu"))
    holder = torch.full((m + 3, n + 16), float("nan"), device="cuda", dtype=torch.bfloat16)
    out = holder[1 : m + 1, 8 : 8 + n]
    assert gemm(a, b, bias=bias, activation="gelu", residual=residual, out=out) is out
    check_gemm("strided out", out)
    outside = torch.ones_like(holder, dtype=torch.bool)
    outside[1 : m + 1, 8 : 8 + n] = False
    queue_requirement("values around the strided out", holder[outside].isnan().all())
    out = shifted(torch.empty(m, n, device="cuda", dtype=torch.bfloat16))
    check_gemm("misaligned out", gemm(a, b, bias=bias, activation="gelu", residual=residual, out=out))
    out = residual.clone()
    check_gemm("residual is out", gemm(a, b, bias=bias, activation="gelu", residual=out, out=out))
    for shift in (1, -1):
        buffer = torch.empty(m + 1, n, device="cuda", dtype=torch.bfloat16)
        overlapping, out = (buffer[1:], buffer[:m]) if shift == 1 else (buffer[:m], buffer[1:])
        overlapping.copy_(residual)
        check_gemm(f"residual shifted by {shift} rows", gemm(a, b, bias=bias, activation="gelu", residual=overlapping,
                                                              out=out))
    shared = torch.empty(m, k + n, device="cuda", dtype=torch.bfloat16)
    shared[:, :k] = a
    check_gemm("out between the rows of a", gemm(shared[:, :k], b, bias=bias, activation="gelu", residual=residual,
                                                 out=shared[:, k:]))
    f32_holder = torch.full((m, n + 4), float("nan"), device="cuda")
    check_gemm("strided FP32 out", gemm(padded(a), b, bias=bias, out=f32_holder[:, :n], out_dtype=torch.float32),
               reference("store_f32", a, b, None, bias, None, None, "none"))
    queue_requirement("values beside the FP32 out", f32_holder[:, n:].isnan().all())
    fused = padded(torch.cat([w_up, b]))
    check_gemm("gate after up", gated_gemm(a, fused[n:], fused[:n], bias_up=bias, residual=padded(residual)),
               gated_expected)
    check_gemm("weights of two strides", gated_gemm(a, padded(b), w_up, bias_up=bias, residual=residual),
               gated_expected)
    out = torch.full((m, n + 12), float("nan"), device="cuda", dtype=torch.bfloat16)[:, 8 : 8 + n]
    check_gemm("gated rows 8 bytes past a multiple of 16 bytes apart",
               gated_gemm(padded(a, 8, 4), padded(b, 8, 4), padded(w_up, 8, 4), bias_up=bias,
                          residual=padded(residual, 8, 4), out=out), gated_expected)
    out = padded(residual)
    check_gemm("gated residual is out", gated_gemm(a, b, w_up, bias_up=bias, residual=out, out=out), gated_expected)
    out = torch.full((m, n + 40), float("nan"), device="cuda", dtype=torch.bfloat16)[:, 16 : 16 + n]
    check_gemm("gated strided out", gated_gemm(padded(a), b, w_up, bias_up=bias, residual=repeated, out=out),
               reference("gated", a, b, w_up, None, bias, repeated, "silu"))
    for kind in KINDS:
        activation, used_residual = ("none", None) if kind == "store_f32" else ("gelu_tanh", padded(residual))
        for info in few_rows(kind):
            if m <= info.tile_m:
                out = torch.full((m, n + 40), float("nan"), device="cuda", dtype=output_dtype(kind))[:, 16 : 16 + n]
                launch(kind, padded(a), padded(b, 0, 40), padded(w_up, 0, 40), out, bias, bias, used_residual,
                       activation, tiling=launcher.tiling_name(info))
                check_gemm(f"{info.kernel} on padded operands", out,
                           reference(kind, a, b, w_up, bias, bias, used_residual, activation))
    empty_a, empty_b = (torch.empty(rows, 0, device="cuda", dtype=torch.bfloat16) for rows in (m, n))
    check_gemm("k = 0", gemm(empty_a, empty_b, bias=bias, activation="silu", residual=residual),
               reference("store_bf16", empty_a, empty_b, None, bias, None, residual, "silu"))
    empty_rows = torch.empty(m // 3, 3, 0, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    check_gemm("k = 0 in rows of two strides", gated_gemm(empty_rows, empty_b, empty_b, bias_gate=bias, bias_up=bias),
               reference("gated", empty_a, empty_b, empty_b, bias, bias, None, "silu"))
    assert gemm(a[:0], b).shape == (0, n) and gated_gemm(a.view(m, 1, k)[:, :0], b, w_up).shape == (m, 0, n)
    assert gemm(a, b[:0]).shape == (m, 0)


def test_row_strides_above_the_launcher_bound_give_the_reference_bits():
    """An `a` whose row stride is 2^31 or 2^32 elements, above the bound 2^31 - 1 of the BF16 GEMM launcher, gives the
    reference bits."""
    cuda_seed(6)
    m, n, k = 2, 64, 64
    holder = torch.empty((1 << 32) + k, device="cuda", dtype=torch.bfloat16)
    b, w_up = exact_values(n, k), exact_values(n, k)
    declare_products(8, m, 2 * n, k, "gated")
    for stride in (1 << 31, 1 << 32):
        a = holder.as_strided((m, k), (stride, 1))
        a.copy_(exact_values(m, k))
        with case(stride=stride):
            queue_equal_check(gemm(a, b), reference("store_bf16", a, b, None, None, None, None, "none"), "gemm")
            queue_equal_check(gated_gemm(a, b, w_up), reference("gated", a, b, w_up, None, None, None, "silu"),
                              "gated_gemm")
    del holder, a
    torch.cuda.empty_cache()


def test_epilogue_inputs_are_bitwise():
    """With zero sums, each of the 65536 BF16 bit patterns as the bias, the gate bias and the residual gives the
    result of the PyTorch BF16 CUDA ops on every kernel."""
    cuda_seed(3)
    n, k = ALL_BF16.numel(), 16
    patterns = ALL_BF16.cuda()
    zeros_a = torch.zeros(EPILOGUE_ROWS, k, device="cuda", dtype=torch.bfloat16)
    zeros_b = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    residual = torch.stack([patterns[torch.randperm(n, device="cuda")] for _ in range(EPILOGUE_ROWS)])
    bias_up = patterns[torch.randperm(n, device="cuda")]
    declare_products(120, EPILOGUE_ROWS, n, k, "gated")
    t = torch.zeros(EPILOGUE_ROWS, n, device="cuda") + patterns.float()
    v = (torch.zeros(EPILOGUE_ROWS, n, device="cuda") + bias_up.float()).bfloat16()
    for kind in KINDS:
        for info in launcher.tilings(kind):
            activations = ("none",) if kind == "store_f32" else NAMES
            for activation in activations:
                for used_residual in (None, residual) if kind != "store_f32" else (None,):
                    if kind == "gated":
                        expected = ACTIVATIONS[activation](t.bfloat16()) * v
                    else:
                        expected = t if kind == "store_f32" else ACTIVATIONS[activation](t.bfloat16())
                    if used_residual is not None:
                        expected = expected + used_residual
                    with case(kernel=info.kernel, activation=activation, residual=used_residual is not None):
                        out = torch.empty(EPILOGUE_ROWS, n, device="cuda", dtype=output_dtype(kind))
                        launch(kind, zeros_a, zeros_b, zeros_b, out, patterns, bias_up, used_residual, activation,
                               tiling=launcher.tiling_name(info))
                        queue_equal_check(out, expected, "out")


def ulp(values):
    """Returns the gap between neighboring BF16 values at `|values|` in FP64."""
    magnitude = values.double().abs()
    exponent = torch.frexp(magnitude).exponent
    return torch.where(magnitude >= 2.0**-126, torch.ldexp(torch.ones_like(magnitude), exponent - 8), 2.0**-133)


def roundings(kind, product_plan, k):
    """Returns `R`, the FP32 roundings in sequence that can reach one sum of the plan. A part of k covers whole k
    tiles, and an `mma` step whose 16 products all lie past k adds zeros without a rounding."""
    info, splits = product_plan.info, product_plan.splits
    if launcher.is_few_rows(info):
        return 12 + math.ceil(k / 512)
    tiles = math.ceil(k / info.tile_k)
    parts = [range(part, tiles, splits) for part in range(splits)]
    if kind == "store_f32":
        run_tiles = 256 // info.tile_k
        return 16 + max(math.ceil(len(part) / run_tiles) for part in parts) + splits
    steps = max(sum(math.ceil(min(info.tile_k, k - tile * info.tile_k) / 16) for tile in part) for part in parts)
    return steps + splits


def deltas(a, w, r):
    return 2.0**-22 * r * sums_of(a.abs(), w.abs())


def store_bound(a, b, bias, residual, activation, out_dtype, r):
    sums = sums_of(a, b)
    ref, t, activated = store_reference(sums, bias, residual, activation, out_dtype)
    delta = deltas(a, b, r)
    if out_dtype == torch.float32:
        return ref, delta + 2.0**-23 * ref.double().abs()
    bound = delta + 2 * ulp(t)
    if activation != "none":
        bound = SLOPE * bound + 2 * ulp(activated)
    return ref, bound + (0 if residual is None else 2 * ulp(ref))


def gated_bound(a, w_gate, w_up, bias_gate, bias_up, residual, activation, r):
    ref, g, v, activated = gated_reference(sums_of(a, w_gate), sums_of(a, w_up), bias_gate, bias_up, residual,
                                           activation)
    dg = deltas(a, w_gate, r) + 2 * ulp(g)
    if activation != "none":
        dg = SLOPE * dg + 2 * ulp(activated)
    dv = deltas(a, w_up, r) + 2 * ulp(v)
    bound = v.double().abs() * dg + activated.double().abs() * dv + dg * dv + 2 * ulp(ref)
    return ref, bound + (0 if residual is None else 2 * ulp(ref))


def queue_within(name, value, ref, bound):
    error = (value.double().reshape(ref.shape) - ref.double()).abs()
    queue("bound", f"{name} error/bound", torch.stack([(error / bound).max(), torch.ones_like(bound[0, 0])]))


def random_operands(m, n, k):
    """Returns `a ~ N(0, 1)`, two weights `~ N(0, 1/k)`, two biases `~ N(0, 1)` and a residual `~ N(0, 16)`."""
    weights = randn(n, k, scale=k**-0.5), randn(n, k, scale=k**-0.5)
    return randn(m, k), *weights, randn(n), randn(n), randn(m, n, scale=4.0)


def check_random(kind, plan_arguments, m, n, k, operands, activation, with_bias, with_residual):
    a, b, w_up, bias, bias_up, residual = operands
    used_bias, used_residual = bias if with_bias else None, residual if with_residual else None
    product_plan = launcher.plan(device_id(), kind, m, n, k, *plan_arguments)
    r = roundings(kind, product_plan, k)
    if kind == "gated":
        ref, bound = gated_bound(a, b, w_up, used_bias, bias_up, used_residual, activation, r)
        if plan_arguments[0] is None:
            out = gated_gemm(a, b, w_up, bias_gate=used_bias, bias_up=bias_up, activation=activation,
                             residual=used_residual)
        else:
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            launch(kind, a, b, w_up, out, used_bias, bias_up, used_residual, activation, tiling=plan_arguments[0],
                   splits=plan_arguments[2])
    else:
        out_dtype = output_dtype(kind)
        ref, bound = store_bound(a, b, used_bias, used_residual, activation, out_dtype, r)
        if plan_arguments[0] is None:
            out = gemm(a, b, bias=used_bias, activation=activation, residual=used_residual, out_dtype=out_dtype)
        else:
            out = torch.empty(m, n, device="cuda", dtype=out_dtype)
            launch(kind, a, b, None, out, used_bias, None, used_residual, activation, tiling=plan_arguments[0],
                   splits=plan_arguments[2])
    with case(plan=launcher.plan_name(product_plan), R=r):
        queue_within("out", out, ref, bound)


@pytest.mark.parametrize("kind", KINDS)
def test_random_operands_meet_the_per_element_bounds(kind):
    """Random operands meet the per-element error bounds on the plans of the torch ops and on the
    tiled kernels with all of k and with 8 parts of k."""
    cuda_seed(4)
    declare_products(len(RANDOM_SHAPES) * 4 + 30, 1020, 1024, 1024, kind)
    activations = ("none",) if kind == "store_f32" else NAMES
    for index, (m, n, k) in enumerate(RANDOM_SHAPES):
        operands = random_operands(m, n, k)
        for activation in activations:
            with case(shape=(m, n, k), activation=activation):
                check_random(kind, (None, None, None), m, n, k, operands, activation, True, kind != "store_f32")
        with case(shape=(m, n, k)):
            check_random(kind, (None, None, None), m, n, k, operands, "none", index % 2 == 0, False)
    m, n, k = 200, 264, 4104
    operands = random_operands(m, n, k)
    for info in tiled(kind):
        for splits in (1, info.max_splits):
            activation = "none" if kind == "store_f32" else NAMES[splits % len(NAMES)]
            with case(kernel=info.kernel, splits=splits, activation=activation):
                check_random(kind, (launcher.tiling_name(info), None, splits), m, n, k, operands, activation, True,
                             kind != "store_f32")


def special_operands(m, n, k):
    """Returns `a` and `b` with NaN in row 0 of `a` and in row `n - 1` of `b`. An infinity in row 1 of `a` meets zeros
    in rows 0 to 7 of `b`, and the sums of row 2 of `a` with rows 8 to 23 of `b` exceed the FP32 range."""
    a, b = randn(m, k), randn(n, k, scale=k**-0.5)
    b[:, k // 2] = torch.where(b[:, k // 2] == 0, 1.0, b[:, k // 2])
    b[:8, k // 2] = 0
    b[:, :16] = 0
    b[8:16, :16], b[16:24, :16] = 1.0, -1.0
    a[0, 3], a[1, k // 2] = float("nan"), float("inf")
    a[2] = 0
    a[2, :16] = 2.0**127
    b[n - 1, 5] = float("nan")
    return a, b


def test_nan_and_infinity_follow_ieee_rules():
    """NaN, infinities, infinity times zero and sums beyond the FP32 range follow IEEE arithmetic on every kernel,
    and the epilogue returns what the PyTorch BF16 ops return for these sums."""
    cuda_seed(5)
    m, n, k = 32, 136, 264
    a, b = special_operands(m, n, k)
    sums = sums_of(a, b)
    special = ~sums.float().isfinite()
    assert special[0].all() and special[:, n - 1].all() and special[1].all() and special[2, 8:24].all()
    bias, residual = randn(n), randn(m, n)
    declare_products(60, m, n, k, "gated")
    for kind in KINDS:
        activation = "none" if kind == "store_f32" else "silu"
        for info in launcher.tilings(kind):
            rows = min(m, info.tile_m) if launcher.is_few_rows(info) else m
            used_residual = None if kind == "store_f32" else residual[:rows]
            out = torch.empty(rows, n, device="cuda", dtype=output_dtype(kind))
            launch(kind, a[:rows], b, b, out, bias, bias, used_residual, activation, tiling=launcher.tiling_name(info))
            if kind == "gated":
                expected = gated_reference(sums[:rows], sums[:rows], bias, bias, used_residual, activation)[0]
            else:
                expected = store_reference(sums[:rows], bias, used_residual, activation, output_dtype(kind))[0]
            with case(kernel=info.kernel):
                queue_equal_check(out[special[:rows]], expected[special[:rows]], "special sums")
                queue_requirement("finite sums", out[~special[:rows]].isfinite().all())


def test_one_plan_gives_the_same_bits():
    """A plan fixes the order of every FP32 addition, so rasters, streams, graph replays and the tiled kernels with
    all of k give the same bits on random operands, and so do repeated calls of the torch ops."""
    cuda_seed(6)
    m, n, k = 450, 488, 328
    a, b, w_up, bias, _, residual = random_operands(m, n, k)
    declare_products(120, m, n, k, "gated")
    side = torch.cuda.Stream()
    for kind in KINDS:
        activation, used_residual = ("none", None) if kind == "store_f32" else ("gelu_tanh", residual)
        for splits in (1, 3):
            infos = tiled(kind) if splits == 1 else tiled(kind)[:1]
            first = torch.empty(m, n, device="cuda", dtype=output_dtype(kind))
            launch(kind, a, b, w_up, first, bias, None, used_residual, activation,
                   tiling=launcher.tiling_name(infos[0]), splits=splits)
            for info in infos:
                for raster in RASTERS:
                    with case(kind=kind, kernel=info.kernel, splits=splits, raster=raster):
                        out = torch.empty_like(first)
                        launch(kind, a, b, w_up, out, bias, None, used_residual, activation,
                               tiling=launcher.tiling_name(info), splits=splits, raster=raster)
                        queue_equal_check(out, first, "out")
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                on_side = torch.empty_like(first)
                launch(kind, a, b, w_up, on_side, bias, None, used_residual, activation,
                       tiling=launcher.tiling_name(infos[0]), splits=splits)
            torch.cuda.current_stream().wait_stream(side)
            queue_equal_check(on_side, first, f"{kind} split {splits} on a side stream")
    for call in (lambda: gemm(a, b, bias=bias, residual=residual), lambda: gated_gemm(a, b, w_up, bias_up=bias)):
        queue_equal_check(call(), call(), "repeated torch op")


def test_ops_pass_opcheck():
    cuda_seed(7)
    for m, n, k in ((5, 24, 64), SPLIT_SHAPE):
        a, b, w_up, bias, bias_up, residual = random_operands(m, n, k)
        a3, residual3 = a.view(1, m, k), residual.view(1, m, n)
        declare_products(40, m, n, k, "gated")
        check_op(gemm_op, a3, b, bias, residual3, "silu", torch.bfloat16)
        check_op(gemm_op, a, b, None, None, "none", torch.float32)
        check_op(gemm_out_op, a3, b, bias, residual3, "gelu", torch.empty_like(residual3))
        check_op(gemm_out_op, a, b, bias, None, "none", torch.empty(m, n, device="cuda"))
        check_op(gated_gemm_op, a3, b, w_up, bias, bias_up, residual3, "gelu_tanh")
        check_op(gated_gemm_out_op, a, b, w_up, None, bias_up, None, "none", torch.empty_like(residual))


def test_fakes_keep_every_size_symbolic():
    a = torch.empty(3, 5, 64, device="cuda", dtype=torch.bfloat16)
    b, residual = torch.empty(24, 64, device="cuda", dtype=torch.bfloat16), torch.empty(3, 5, 24, device="cuda",
                                                                                         dtype=torch.bfloat16)
    bias = torch.empty(24, device="cuda", dtype=torch.bfloat16)

    def products(a, b, bias, residual, out):
        return (gemm(a, b, bias=bias, activation="silu", residual=residual),
                gemm(a, b, bias=bias, out_dtype=torch.float32),
                gated_gemm(a, b, b, bias_gate=bias, residual=residual),
                gemm(a, b, residual=out, out=out),
                gated_gemm(a, b, b, bias_up=bias, out=out))

    outputs = call_symbolic(products, a, b, bias, residual, residual.clone())
    for output in outputs:
        assert_symbolic(output)
    assert [output.dtype for output in outputs] == [torch.bfloat16, torch.float32, torch.bfloat16, torch.bfloat16,
                                                    torch.bfloat16]


def compiled_case(m, n, k):
    def make():
        generator = torch.Generator(device="cuda").manual_seed(m * n * k)

        def values(*shape):
            return (torch.randint(-8, 9, shape, device="cuda", generator=generator) / 16).bfloat16()

        a, b, w_up, bias = values(2, m, k), values(n, k), values(n, k), values(n)
        residual = torch.randn(2, m, n, device="cuda", generator=generator).bfloat16()
        return a, b, w_up, bias, residual, residual.clone(), residual.clone()

    return make


def compiled_products(a, b, w_up, bias, residual, alias, gated_alias):
    return (gemm(a, b, bias=bias, activation="gelu", residual=residual),
            gemm(a, b, bias=bias, out_dtype=torch.float32),
            gated_gemm(a, b, w_up, bias_gate=bias, residual=residual),
            gemm(a, b, bias=bias, activation="silu", residual=alias, out=alias),
            gated_gemm(a, b, w_up, residual=gated_alias, out=gated_alias),
            alias, gated_alias)


@pytest.mark.usefixtures("fresh_compiler")
def test_compiled_calls_match_eager_calls():
    """`torch.compile(fullgraph=True, dynamic=True)` compiles once for three sizes of every dimension, and the
    compiled `residual is out` aliases match the eager ones."""
    declare_products(10 * len(COMPILED_SIZES), 2 * 72, 56, 48, "gated")
    check_compiled(compiled_products, [compiled_case(*size) for size in COMPILED_SIZES])


def test_calls_never_synchronize_and_replay_in_graphs():
    """The ops and the launchers run under the sync debug mode and inside a CUDA graph that replays on new inputs,
    split products included."""
    cuda_seed(8)
    m, n, k = SPLIT_SHAPE
    assert launcher.plan(device_id(), "store_bf16", m, n, k).splits > 1
    assert launcher.plan(device_id(), "gated", m, n // 2, k).splits > 1
    a, b, _, bias, _, residual = random_operands(m, n, k)
    given = shifted(torch.empty(m, n, device="cuda", dtype=torch.bfloat16))
    given_gated = shifted(torch.empty(m, n // 2, device="cuda", dtype=torch.bfloat16))
    launched = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    declare_products(48, m, n, k)

    def run():
        launcher.gemm(a, b, launched, bias=bias, splits=4, stream=stream())
        return (gemm(a, b, bias=bias, residual=residual), gated_gemm(a, b[: n // 2], b[n // 2 :]),
                gemm(a, b, out_dtype=torch.float32), gemm(a[:5], b, activation="silu"),
                gemm(a, b, residual=residual, out=given), gated_gemm(a, b[: n // 2], b[n // 2 :], out=given_gated))

    def refresh():
        for tensor in (a, b, bias, residual):
            tensor.copy_(torch.randn_like(tensor) * (k**-0.5 if tensor is b else 1.0))

    _, outputs = check_graph_replay(run, refresh)
    gated = gated_gemm(a, b[: n // 2], b[n // 2 :])
    expected = (gemm(a, b, bias=bias, residual=residual), gated, gemm(a, b, out_dtype=torch.float32),
                gemm(a[:5], b, activation="silu"), gemm(a, b, residual=residual), gated)
    for index, (result, reference_bits) in enumerate(zip(outputs, expected)):
        queue_equal_check(result, reference_bits, f"graph output {index}")
    expected = torch.empty_like(launched)
    launcher.gemm(a, b, expected, bias=bias, splits=4, stream=stream())
    queue_equal_check(launched, expected, "launcher in the graph")


def test_launcher_zeroes_its_counters_in_graphs():
    """A split product of the launcher in a CUDA graph zeroes its counters in every replay, also when its graph
    allocation reuses memory that an earlier node of the graph filled with other values."""
    cuda_seed(12)
    m, n, k = SPLIT_SHAPE
    a, b = exact_values(m, k), exact_values(n, k)
    plan = {"tiling": launcher.tiling_name(tiled("store_bf16")[0]), "splits": 4}
    declare_products(4, m, n, k)
    expected = reference("store_bf16", a, b, None, None, None, None, "none")
    out = torch.empty_like(expected)
    launch, _ = launcher.store_launch(a, b, out, stream=stream(), **plan)
    size = launch.partial_bytes + launcher.COUNTER_BYTES * launch.counter_count
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        filled = launcher.checked(driver.cuMemAllocAsync(size, stream()))
        launcher.checked(driver.cuMemsetD32Async(filled, POISON, size // 4, stream()))
        launcher.checked(driver.cuMemFreeAsync(filled, stream()))
        launcher.gemm(a, b, out, stream=stream(), **plan)
    for replay in range(3):
        out.fill_(float("nan"))
        graph.replay()
        queue_equal_check(out, expected, f"replay {replay}")


def test_split_products_in_concurrent_graphs():
    """Graphs captured on one stream replay at once on two streams beside eager split products on that stream."""
    cuda_seed(9)
    m, n, k = SPLIT_SHAPE
    inputs = [randn(m, k) for _ in range(2)]
    b, bias = randn(n, k, scale=k**-0.5), randn(n)
    declare_products(80, m, n, k)

    def products(a):
        return gemm(a, b, bias=bias), gated_gemm(a, b[: n // 2], b[n // 2 :])

    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        products(inputs[0])
    torch.cuda.current_stream().wait_stream(capture)
    expected = [products(a) for a in inputs]
    graphs, outputs = [], []
    for a in inputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture):
            outputs.append([products(a) for _ in range(4)])
        graphs.append(graph)
    replays = [torch.cuda.Stream() for _ in graphs]
    for replay in replays:
        replay.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        for graph, replay in zip(graphs, replays):
            with torch.cuda.stream(replay):
                graph.replay()
    with torch.cuda.stream(capture):
        eager = [products(inputs[1]) for _ in range(4)]
    for other in (*replays, capture):
        torch.cuda.current_stream().wait_stream(other)
    for index, (graph_outputs, reference_bits) in enumerate(zip(outputs, expected)):
        for out, gated in graph_outputs:
            queue_equal_check(out, reference_bits[0], f"graph {index}")
            queue_equal_check(gated, reference_bits[1], f"gated graph {index}")
    for out, gated in eager:
        queue_equal_check(out, expected[1][0], "eager")
        queue_equal_check(gated, expected[1][1], "gated eager")


def test_split_products_on_two_streams():
    """Each stream has its own launcher workspace, so split products on two streams run at once."""
    cuda_seed(10)
    m, n, k = 256, 512, 2048
    a, b, _, bias, _, _ = random_operands(m, n, k)
    declare_products(8, m, n, k)
    plan = {"tiling": launcher.tiling_name(tiled("store_bf16")[0]), "splits": 4}
    expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    launcher.gemm(a, b, expected, bias=bias, stream=stream(), **plan)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    outs = [torch.empty_like(expected) for _ in range(6)]
    for index, out in enumerate(outs):
        target = side if index % 2 else torch.cuda.current_stream()
        launcher.gemm(a, b, out, bias=bias, stream=target.cuda_stream, **plan)
    torch.cuda.current_stream().wait_stream(side)
    for out in outs:
        queue_equal_check(out, expected, "out")


def test_launcher_keeps_the_workspaces_of_recent_streams():
    """A device keeps the launcher workspaces of its most recent streams and frees an older workspace after its
    launches, also when the stream of those launches no longer exists."""
    cuda_seed(13)
    m, n, k = SPLIT_SHAPE
    a, b = exact_values(m, k), exact_values(n, k)
    plan = {"tiling": launcher.tiling_name(tiled("store_bf16")[0]), "splits": 4}
    expected = reference("store_bf16", a, b, None, None, None, None, "none")
    device = runtime.current_device()
    streams = [device.create_stream() for _ in range(launcher.WORKSPACES_PER_DEVICE + 8)]
    declare_products(len(streams), m, n, k)
    outs = [torch.empty_like(expected) for _ in streams]
    torch.cuda.synchronize()
    launcher.gemm(a, b, outs[0], stream=streams[0], **plan)
    oldest = launcher.workspaces[device.device_id, int(streams[0].handle)]
    streams[0].close()
    for out, side in zip(outs[1:], streams[1:]):
        launcher.gemm(a, b, out, stream=side, **plan)
    torch.cuda.synchronize()
    kept = [key for key in launcher.workspaces if key[0] == device.device_id]
    assert len(kept) <= launcher.WORKSPACES_PER_DEVICE
    assert oldest.parts is None and oldest.counters is None, "the oldest workspace is freed"
    for out in outs:
        queue_equal_check(out, expected, "split product on its own stream")
    for side in streams[1:]:
        side.close()


def test_threads_share_one_stream():
    """Threads that launch split products on one stream with growing workspaces get the reference bits."""
    cuda_seed(11)
    k = 2048
    shapes = [(64 * (1 + index % 3), 128 * (1 + index % 4)) for index in range(12)]
    a, b = exact_values(192, k), exact_values(512, k)
    references = {shape: reference("store_bf16", a[: shape[0]], b[: shape[1]], None, None, None, None, "none")
                  for shape in set(shapes)}
    declare_products(4 * len(shapes) * 2, 192, 512, k)
    splits = tiled("store_bf16")[0].max_splits
    results, failures = [], []

    def work(index):
        try:
            for m, n in shapes[index:] + shapes[:index]:
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                launcher.gemm(a[:m], b[:n], out, splits=splits, stream=stream())
                results.append(((m, n), out, gemm(a[:m], b[:n])))
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=work, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures, failures
    for shape, launched, op in results:
        queue_equal_check(launched, references[shape], f"launcher {shape}")
        queue_equal_check(op, references[shape], f"op {shape}")
