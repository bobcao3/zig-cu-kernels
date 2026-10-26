import itertools
import math
import re
from collections import namedtuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import checks
from checks import case, queue_equal_check, queue_requirement
from conftest import cuda_seed, declare_gpu_work
from opcheck import (
    assert_same,
    assert_symbolic,
    call_symbolic,
    check_compiled,
    check_graph_replay,
    check_op,
    fresh_compiler,  # noqa: F401
)
from zig_cu_kernels import fp8_gemm as launcher
from zig_cu_kernels import runtime
from zig_cu_kernels.torch import (
    common,
    fp8_gated_gemm,
    fp8_gated_gemm_op,
    fp8_gated_gemm_out_op,
    fp8_gemm,
    fp8_gemm_op,
    fp8_gemm_out_op,
    quantize_fp8_rows,
    quantize_fp8_rows_op,
    quantize_fp8_rows_out_op,
    quantize_fp8_static_op,
    quantize_fp8_static_out_op,
    quantize_fp8_tensorwise,
    quantize_fp8_tensorwise_op,
    quantize_fp8_tensorwise_out_op,
    quantize_fp8_tiles,
    quantize_fp8_tiles_op,
    quantize_fp8_tiles_out_op,
)

UNAVAILABLE = runtime.unavailable_reason("fp8_gemm_rowwise_store_128x256") or \
    runtime.unavailable_reason("fp8_quantize_rows_bf16")
pytestmark = pytest.mark.skipif(UNAVAILABLE is not None, reason=UNAVAILABLE or "")

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
SCALE_GROUP = 128
U = 2.0**-8
PRACTICAL_GAMMA = 2.0**-9
FINITE_GAMMA = 2.0**-6
SECOND_ORDER = 1.01
SUBNORMAL_E4M3 = 2.0**-6
NAN_CODE = 0x7F
INPUT_DTYPES = (torch.bfloat16, torch.float32)
SCALE_DTYPES = (torch.float32, E8M0)
MODES = ("tensorwise", "rowwise", "block128", "block128_e8m0")
FUNCTIONS = {
    "none": lambda x: x,
    "silu": F.silu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu": F.gelu,
}
ACTIVATIONS = tuple(FUNCTIONS)
SLOPES = {"none": 1.0, "silu": 1.10, "gelu_tanh": 1.13, "gelu": 1.13}
RASTERS = ((1, "along_n"), (2, "along_n"), (3, "along_n"), (1, "along_m"), (2, "along_m"), (4, "along_m"))
# The positions of the codes address and of the scales address among the arguments of each quantizer launch function.
QUANTIZER_OUTPUTS = {"launch_rows": (3, 4), "launch_tiles": (3, 4), "launch_tensorwise": (3, 6)}
QuantizerLaunch = namedtuple("QuantizerLaunch", "codes scales")
ALL_BF16 = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
EPILOGUE_ROWS = 4


@pytest.fixture(autouse=True)
def fresh_plans():
    launcher.product_launch.cache_clear()
    launcher.choose.cache_clear()
    yield
    launcher.product_launch.cache_clear()
    launcher.choose.cache_clear()


def replace(monkeypatch, target, name, value):
    """Replaces a function that the plans depend on and drops the plans made before."""
    monkeypatch.setattr(target, name, value)
    launcher.product_launch.cache_clear()


def declare_products(launches, m, columns, k, gated=False):
    """Declares launches that read FP8 operands, a BF16 bias and a BF16 residual and write a BF16 output in DRAM."""
    rows_of_b = 2 * columns if gated else columns
    dram = m * k + rows_of_b * k + 2 * columns * (1 + gated) + 2 * 2 * m * columns
    declare_gpu_work(launches, flops=2 * m * rows_of_b * k, dram_bytes=dram, element="e4m3")


def generator_of(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def allocation_count():
    return torch.cuda.memory_stats()["allocation.all.allocated"]


def scale_dtype_of(mode):
    return E8M0 if mode.endswith("e8m0") else torch.float32


def scaling_of(mode):
    return mode.removesuffix("_e8m0")


def e8m0_of(exponents):
    return (exponents + 127).to(torch.uint8).view(E8M0)


def block_scale_a(values):
    """Returns `values` (`(m, k / 128)`) in the BlockWise1x128 layout, with strides `(1, m)`."""
    scales = torch.empty(values.shape[::-1], dtype=values.dtype, device=values.device).t()
    scales.copy_(values)
    return scales


def scale_factor(scale, mode, rows, k, tiles=False):
    """Returns the FP64 factor of each element of an operand with `rows` rows of `k` values."""
    values = scale.float().double()
    scaling = scaling_of(mode)
    if scaling == "tensorwise":
        return values.reshape(())
    if scaling == "rowwise":
        return values.reshape(rows, 1)
    scale_groups = k // SCALE_GROUP
    if tiles:
        return (values.reshape(-1, scale_groups).repeat_interleave(SCALE_GROUP, 0)[:rows]
                .repeat_interleave(SCALE_GROUP, 1))
    return values.reshape(rows, scale_groups).repeat_interleave(SCALE_GROUP, 1)


def dequantized(codes, scale, mode, tiles=False):
    rows, k = math.prod(codes.shape[:-1]), codes.shape[-1]
    return codes.reshape(rows, k).double() * scale_factor(scale, mode, rows, k, tiles)


def is_subnormal(codes):
    magnitudes = codes.float().abs()
    return (magnitudes < SUBNORMAL_E4M3) & (magnitudes > 0)


def magnitudes(a, b, a_codes, b_codes):
    """Returns `P` and the part of `P` from products with a subnormal code."""
    total = a.abs() @ b.abs().T
    a_normal = a.abs() * ~is_subnormal(a_codes.reshape(a.shape))
    b_normal = b.abs() * ~is_subnormal(b_codes.reshape(b.shape))
    return total, total - a_normal @ b_normal.T


def queue_bound(name, out, reference, allowed):
    """Queues the per-element check `|out - reference| <= 1.01 * allowed`, reporting the largest ratio."""
    out = out.double().reshape(reference.shape)
    error = torch.where(out == reference, 0.0, (out - reference).abs())
    limit = SECOND_ORDER * allowed
    ratio = torch.where(error <= limit, 0.0, error / limit)
    checks.queue("bound", f"{name}: largest error / bound", torch.stack([ratio.amax(), torch.ones_like(ratio[0, 0])]))


def store_bound(s, error, bias, activation, residual):
    """Returns the FP64 reference and the allowed error of `fp8_gemm` for the sums `s` and their error term."""
    y = s + (0.0 if bias is None else bias.double())
    allowed = SLOPES[activation] * (error + U * y.abs())
    reference = FUNCTIONS[activation](y)
    if activation != "none":
        allowed = allowed + U * reference.abs()
    if residual is not None:
        reference = reference + residual.double().reshape(reference.shape)
        allowed = allowed + U * reference.abs()
    return reference, allowed


def gated_bound(s_gate, error_gate, s_up, error_up, bias_gate, bias_up, activation, residual):
    g = s_gate + (0.0 if bias_gate is None else bias_gate.double())
    v = s_up + (0.0 if bias_up is None else bias_up.double())
    activated = FUNCTIONS[activation](g)
    error_g, error_v = error_gate + U * g.abs(), error_up + U * v.abs()
    reference = activated * v
    allowed = v.abs() * (SLOPES[activation] * error_g + U * activated.abs()) + activated.abs() * error_v
    allowed = allowed + U * reference.abs()
    if residual is not None:
        reference = reference + residual.double().reshape(reference.shape)
        allowed = allowed + U * reference.abs()
    return reference, allowed


def sums_and_errors(a, scale_a, b, scale_b, mode, gamma):
    a_hp, b_hp = dequantized(a, scale_a, mode), dequantized(b, scale_b, mode, tiles=True)
    total, subnormal = magnitudes(a_hp, b_hp, a, b)
    return a_hp @ b_hp.T, gamma * total + subnormal


def reference_scale(amax, scale_dtype):
    """Returns the scale of the quantizer rules for FP32 `amax` values."""
    t = (amax.double() / 448).float()
    if scale_dtype == torch.float32:
        return torch.where(t == 0, torch.ones_like(t), t)
    mantissa, exponent = torch.frexp(t)
    byte = (exponent - (mantissa == 0.5).to(exponent.dtype) + 127).clamp(0, 254)
    byte = torch.where(t == 0, 127, byte)
    byte = torch.where(t.isnan() | t.isinf(), 255, byte)
    return byte.to(torch.uint8).view(E8M0)


def reference_codes(x, scale):
    return (x.float() / scale.float()).clamp(-448, 448).to(FP8)


def edge_groups(generator):
    """Returns FP32 scale groups whose largest magnitudes hit the edges of the scale rules. The groups hold zeros of
    both signs, quotients that underflow to 0 or lie near 2^-127, powers of two times 448 and their neighbors, the
    largest finite value, NaN and infinities, and normal values over a wide range."""

    def spread(amax):
        values = (torch.rand(SCALE_GROUP, device="cuda", generator=generator) * 2 - 1) * amax
        values[int(torch.randint(SCALE_GROUP, (1,), device="cuda", generator=generator))] = amax
        return values

    scale_groups = [torch.zeros(SCALE_GROUP, device="cuda"), -torch.zeros(SCALE_GROUP, device="cuda")]
    for bits in (1, 223, 224, 225, 448):
        scale_groups.append(spread(float(np.array([bits], dtype=np.uint32).view(np.float32)[0])))
    for power in (-140, -127, -126, -60, -1, 0, 1, 60, 119):
        amax = np.ldexp(np.float32(448), power)
        scale_groups += [spread(float(value)) for value in (amax, np.nextafter(amax, np.float32(np.inf)),
                                                      np.nextafter(amax, np.float32(0)))]
    scale_groups += [spread(3.4028234663852886e38), torch.full((SCALE_GROUP,), 448.0, device="cuda")]
    for special in (math.nan, math.inf, -math.inf):
        values = spread(1.0)
        values[5] = special
        scale_groups.append(values)
    scale_groups += [torch.randn(SCALE_GROUP, device="cuda", generator=generator) * sigma
                     for sigma in (1e-30, 1.0, 1e30)]
    return torch.stack(scale_groups)


def scale_group_matrix(rows, scale_groups, dtype, generator):
    cases = edge_groups(generator)
    order = torch.randperm(rows * scale_groups, device="cuda", generator=generator) % cases.shape[0]
    return cases[order].reshape(rows, scale_groups * SCALE_GROUP).to(dtype)


@pytest.mark.parametrize("dtype", INPUT_DTYPES)
@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
def test_quantizers_are_bitwise_equal_to_the_rules(dtype, scale_dtype):
    """Rows and tiles follow the quantizer rules for every edge scale group, strided inputs and partial last tiles."""
    generator = generator_of(1)
    declare_gpu_work(20)
    wide = scale_group_matrix(40, 7, dtype, generator)
    for name, x in (("rows", wide[:37, : 5 * SCALE_GROUP]), ("strided rows", wide[:, SCALE_GROUP:]),
                    ("batched rows", wide[:36, : 2 * SCALE_GROUP].reshape(4, 9, 2 * SCALE_GROUP))):
        q, scales = quantize_fp8_rows(x, scale_dtype=scale_dtype)
        m, k = math.prod(x.shape[:-1]), x.shape[-1]
        scale_groups = x.float().reshape(m, k // SCALE_GROUP, SCALE_GROUP)
        expected = reference_scale(scale_groups.abs().amax(-1), scale_dtype)
        assert scales.shape == (m, k // SCALE_GROUP) and scales.stride() == (1, m) and scales.dtype == scale_dtype
        queue_equal_check(scales, expected, f"{name} scales")
        queue_equal_check(q, reference_codes(scale_groups, expected[..., None]).reshape(x.shape), f"{name} codes")
    for n in (136, 200, 256):
        w = scale_group_matrix(n, 3, dtype, generator)
        w[100:110, SCALE_GROUP : 2 * SCALE_GROUP] = 0
        tile_rows = -(-n // SCALE_GROUP)
        padded = torch.zeros(tile_rows * SCALE_GROUP, 3 * SCALE_GROUP, device="cuda").float()
        padded[:n] = w.float()
        tiles = padded.reshape(tile_rows, SCALE_GROUP, 3, SCALE_GROUP)
        q, scales = quantize_fp8_tiles(w, scale_dtype=scale_dtype)
        expected = reference_scale(tiles.abs().amax((1, 3)), scale_dtype)
        assert scales.shape == (tile_rows, 3) and scales.is_contiguous()
        with case(n=n):
            queue_equal_check(scales, expected, "tile scales")
            queue_equal_check(q, reference_codes(tiles, expected[:, None, :, None]).reshape(-1, 3 * SCALE_GROUP)[:n],
                              "tile codes")


@pytest.mark.parametrize("dtype", INPUT_DTYPES)
def test_tensorwise_quantizers_are_bitwise_equal_to_the_rules(dtype):
    """Every input, an empty one included, gets the scale and the codes of the quantizer rules."""
    generator = generator_of(2)
    declare_gpu_work(40)
    full = (torch.randn(37, 200, device="cuda", generator=generator) * 30).to(dtype)
    cases = {"matrix": full[:, :192], "vector": full.flatten()[:1000], "batched": full[:36].reshape(4, 9, 200),
             "zeros": torch.zeros(64, 128, device="cuda", dtype=dtype), "underflow": full[:4, :64] * 1e-44,
             "nan": torch.where(full == full.max(), math.nan, full), "infinity": torch.where(full == full.min(),
                                                                                             -math.inf, full),
             "empty rows": torch.empty(2, 0, 16, device="cuda", dtype=dtype),
             "empty matrix": torch.empty(3, 0, device="cuda", dtype=dtype)}
    given = torch.full((1, 1), 0.0371, device="cuda")
    for name, x in cases.items():
        with case(input=name):
            q, scale = quantize_fp8_tensorwise(x)
            expected = reference_scale(F.pad(x.float().abs().flatten(), (0, 1)).amax(), torch.float32)
            assert scale.shape == () and scale.dtype == torch.float32
            queue_equal_check(scale, expected, "scale")
            queue_equal_check(q, reference_codes(x, expected), "codes")
            q, returned = quantize_fp8_tensorwise(x, given)
            assert returned.shape == () and returned.data_ptr() == given.data_ptr()
            queue_equal_check(q, reference_codes(x, given.reshape(())), "given scale codes")


def record_quantizer_launches(monkeypatch):
    """Returns the list to which each later quantizer launch appends the addresses of its codes and its scales."""
    launches = []
    for name, (codes, scales) in QUANTIZER_OUTPUTS.items():

        def recording(*arguments, launch=getattr(launcher, name), codes=codes, scales=scales):
            launches.append(QuantizerLaunch(arguments[codes], arguments[scales]))
            return launch(*arguments)

        monkeypatch.setattr(launcher, name, recording)
    return launches


def bytes_view(tensor, offset, dtype, shape):
    size = math.prod(shape) * dtype.itemsize
    return tensor.view(-1).view(torch.uint8)[offset : offset + size].view(dtype).view(shape)


def check_outputs_over_input(quantize, x, outputs, launches):
    """Checks that each output of `outputs` over the bytes of `x` goes through a temporary and gets the bits of a call
    on a copy of `x`, and that a separate output of the same layout goes in place."""
    source = x.clone()
    expected = quantize(source)

    def written(keyword):
        return launches[-1].codes if keyword == "out" else launches[-1].scales

    for keyword, output in outputs.items():
        separate = torch.empty_like(output)
        quantize(source, **{keyword: separate})
        assert written(keyword) == separate.data_ptr(), f"a separate {keyword} goes in place"
        x.copy_(source)
        results = quantize(x, **{keyword: output})
        assert written(keyword) != output.data_ptr(), f"{keyword} over the input goes through a temporary"
        with case(output_over_input=keyword):
            queue_equal_check(results[0], expected[0], "codes")
            queue_equal_check(results[1].reshape(expected[1].shape), expected[1], "scales")


def strided_outputs(shape, scale_shape, scale_dtype, transposed):
    """Returns codes and scales outputs with gaps between their rows and the scales in the other orientation."""
    codes = torch.zeros(shape[0], shape[1] + 32, device="cuda", dtype=torch.uint8).view(FP8)[:, 16 : 16 + shape[1]]
    rows, columns = scale_shape
    if transposed:
        scales = torch.zeros(columns, rows + 3, device="cuda", dtype=scale_dtype)[:, :rows].t()
    else:
        scales = torch.zeros(rows, columns + 3, device="cuda", dtype=scale_dtype)[:, :columns]
    return codes, scales


@pytest.mark.parametrize("scale_dtype", SCALE_DTYPES)
def test_quantizer_outputs_at_any_strides(scale_dtype, monkeypatch):
    """Outputs that the caller passes at any strides, also over the input, receive the bits of new outputs, in place
    where the kernels can write them and through temporaries otherwise."""
    cuda_seed(3)
    declare_gpu_work(80)
    launches = record_quantizer_launches(monkeypatch)
    x = torch.randn(70, 384, device="cuda", dtype=torch.bfloat16)
    q, scales = quantize_fp8_rows(x, scale_dtype=scale_dtype)
    for transposed in (False, True):
        out, scales_out = strided_outputs((70, 384), (70, 3), scale_dtype, not transposed)
        result = quantize_fp8_rows(x, scale_dtype=scale_dtype, out=out, scales_out=scales_out)
        assert result[0] is out and result[1] is scales_out
        with case(rows_scales_transposed=transposed):
            queue_equal_check(out, q, "rows codes")
            queue_equal_check(scales_out, scales, "rows scales")
    w = torch.randn(200, 256, device="cuda", dtype=torch.float32)
    q, scales = quantize_fp8_tiles(w, scale_dtype=scale_dtype)
    for transposed in (False, True):
        out, scales_out = strided_outputs((200, 256), (2, 2), scale_dtype, transposed)
        result = quantize_fp8_tiles(w, scale_dtype=scale_dtype, out=out, scales_out=scales_out)
        assert result[0] is out and result[1] is scales_out
        with case(tile_scales_transposed=transposed):
            queue_equal_check(out, q, "tile codes")
            queue_equal_check(scales_out, scales, "tile scales")
    halves = torch.empty(70, 768, device="cuda", dtype=torch.uint8).view(FP8)
    left, right = halves[:, :384], halves[:, 384:]
    quantize_fp8_rows(x, scale_dtype=scale_dtype, out=left)
    queue_equal_check(left, quantize_fp8_rows(x, scale_dtype=scale_dtype)[0], "codes beside another tensor")
    q, scale = quantize_fp8_tensorwise(x[:, :256], out=right[:, :256], scale_out=torch.empty(1, device="cuda"))
    expected_q, expected_scale = quantize_fp8_tensorwise(x[:, :256])
    queue_equal_check(q, expected_q, "tensorwise codes")
    queue_equal_check(scale.reshape(()), expected_scale, "tensorwise scale")
    rows_input, tile_input, tensorwise_input = x.clone(), w.clone(), x.clone()
    half, tile_half = x.numel(), 2 * w.numel()
    check_outputs_over_input(lambda tensor, **outputs: quantize_fp8_rows(tensor, scale_dtype=scale_dtype, **outputs),
                             rows_input, {"out": bytes_view(rows_input, half, FP8, x.shape),
                                          "scales_out": bytes_view(rows_input, half, scale_dtype, (3, 70)).t()},
                             launches)
    check_outputs_over_input(lambda tensor, **outputs: quantize_fp8_tiles(tensor, scale_dtype=scale_dtype, **outputs),
                             tile_input, {"out": bytes_view(tile_input, tile_half, FP8, w.shape),
                                          "scales_out": bytes_view(tile_input, tile_half, scale_dtype, (2, 2))},
                             launches)
    check_outputs_over_input(quantize_fp8_tensorwise, tensorwise_input,
                             {"out": bytes_view(tensorwise_input, half, FP8, x.shape),
                              "scale_out": bytes_view(tensorwise_input, half, torch.float32, (1,))}, launches)


def quantized_operands(mode, x, w, dtype=torch.float32):
    """Returns the codes and scales that the package's quantizers produce for `mode`; rowwise codes follow the FP32
    quantizer rule per row."""
    scaling, scale_dtype = scaling_of(mode), scale_dtype_of(mode)
    if scaling == "tensorwise":
        return (*quantize_fp8_tensorwise(x), *quantize_fp8_tensorwise(w))
    if scaling == "block128":
        return (*quantize_fp8_rows(x, scale_dtype=scale_dtype), *quantize_fp8_tiles(w, scale_dtype=scale_dtype))
    lead = x.shape[:-1]
    rows = x.float().reshape(-1, x.shape[-1])
    scale_a = reference_scale(rows.abs().amax(-1, keepdim=True), torch.float32)
    scale_w = reference_scale(w.float().abs().amax(-1, keepdim=True), torch.float32)
    return (reference_codes(rows, scale_a).reshape(x.shape), scale_a.reshape(*lead, 1),
            reference_codes(w, scale_w), scale_w)


PRACTICAL_SHAPES = ((1, 256, 512), (77, 136, 640), (2 * 37, 264, 256), (300, 576, 384), (64, 128, 4096))
SIXTEEN_SHAPES = ((33, 200, 144), (5, 8, 16), (40, 64, 1040))


def practical_inputs(m, n, k, generator, heavy):
    if heavy:
        t = torch.distributions.StudentT(torch.tensor(3.0, device="cuda"))
        x, w = t.sample((m, k)), t.sample((n, k)) * k**-0.5
    else:
        x = torch.randn(m, k, device="cuda", generator=generator)
        w = torch.randn(n, k, device="cuda", generator=generator) * k**-0.5
    return x.bfloat16(), w


@pytest.mark.parametrize("mode", MODES)
def test_products_meet_the_per_element_bounds(mode):
    """Codes that the quantizers produce from normal and Student-t data meet `gamma = 2^-9` for every epilogue."""
    generator = generator_of(4)
    shapes = PRACTICAL_SHAPES + (SIXTEEN_SHAPES if mode in ("tensorwise", "rowwise") else ())
    cuda_seed(4)
    for index, (m, n, k) in enumerate(shapes):
        declare_products(16, m, n, k, gated=True)
        x, w = practical_inputs(m, n, k, generator, heavy=index % 2 == 1)
        if m == 74:
            x = x.reshape(2, 37, k)
        w_up = torch.randn(n, k, device="cuda", generator=generator) * k**-0.5
        a, scale_a, b, scale_b = quantized_operands(mode, x, w)
        _, _, up, scale_up = quantized_operands(mode, x, w_up)
        s, error = sums_and_errors(a, scale_a, b, scale_b, mode, PRACTICAL_GAMMA)
        s_up, error_up = sums_and_errors(a, scale_a, up, scale_up, mode, PRACTICAL_GAMMA)
        bias, bias_up = (torch.randn(n, device="cuda", generator=generator).bfloat16() for _ in range(2))
        residual = (torch.randn(*x.shape[:-1], n, device="cuda", generator=generator) * 4).bfloat16()
        for activation, with_bias, with_residual in itertools.product(ACTIVATIONS, (False, True), (False, True)):
            if (ACTIVATIONS.index(activation) + with_bias + 2 * with_residual) % 3:
                continue
            options = {"bias": bias if with_bias else None, "residual": residual if with_residual else None}
            with case(m=m, n=n, k=k, activation=activation, bias=with_bias, residual=with_residual):
                out = fp8_gemm(a, b, scale_a, scale_b, activation=activation, **options)
                queue_bound("store", out, *store_bound(s, error, activation=activation, **options))
                gated_options = {"bias_gate": options["bias"], "bias_up": bias_up if with_bias else None,
                                 "residual": options["residual"]}
                out = fp8_gated_gemm(a, b, up, scale_a, scale_b, scale_up, activation=activation, **gated_options)
                queue_bound("gated", out, *gated_bound(s, error, s_up, error_up, activation=activation,
                                                       **gated_options))


def exact_operands(mode, m, n, k, generator):
    """Returns codes in {-1, 0, 1} and power-of-two scales whose product shrinks at most fourfold per scale group.
    Every chain sum, scaled chain and FP32 partial sum of these operands is exact for `k <= 1024`."""
    a = torch.randint(-1, 2, (m, k), device="cuda", generator=generator).float().to(FP8)
    b = torch.randint(-1, 2, (n, k), device="cuda", generator=generator).float().to(FP8)
    scaling, e8m0 = scaling_of(mode), scale_dtype_of(mode) == E8M0
    if scaling == "tensorwise":
        exponents = torch.randint(-4, 5, (2,), device="cuda", generator=generator).float()
        return a, b, torch.exp2(exponents[0]).reshape(()), torch.exp2(exponents[1]).reshape(1, 1)
    if scaling == "rowwise":
        row = torch.randint(-3, 4, (m, 1), device="cuda", generator=generator).float()
        column = torch.randint(-3, 4, (n, 1), device="cuda", generator=generator).float()
        return a, b, torch.exp2(row), torch.exp2(column)
    blocks = torch.arange(k // SCALE_GROUP, device="cuda")
    row = torch.randint(-3, 4, (m, 1), device="cuda", generator=generator)
    tile = torch.randint(-2, 3, (-(-n // SCALE_GROUP), 1), device="cuda", generator=generator)
    a_exponents, b_exponents = row - blocks[None, :], tile - blocks[None, :] % 2
    if e8m0:
        return a, b, block_scale_a(e8m0_of(a_exponents)), e8m0_of(b_exponents)
    return a, b, block_scale_a(torch.exp2(a_exponents.float())), torch.exp2(b_exponents.float())


def exact_sums(a, scale_a, b, scale_b, mode):
    return (dequantized(a, scale_a, mode) @ dequantized(b, scale_b, mode, tiles=True).T).float()


def store_expected(s, bias, activation, residual):
    y = (s + (0.0 if bias is None else bias.float())).bfloat16()
    y = FUNCTIONS[activation](y)
    return y if residual is None else y + residual.reshape(y.shape)


def gated_expected(s_gate, s_up, bias_gate, bias_up, activation, residual):
    g = (s_gate + (0.0 if bias_gate is None else bias_gate.float())).bfloat16()
    v = (s_up + (0.0 if bias_up is None else bias_up.float())).bfloat16()
    y = FUNCTIONS[activation](g) * v
    return y if residual is None else y + residual.reshape(y.shape)


def tile_count(family, kind, tiling, m, columns):
    info = runtime.info(f"fp8_gemm_{family}_{kind}_{tiling}")
    return -(-m // info.tile_m) * -(-columns // launcher.COLUMNS_PER_TILE[kind](info.tile_n))


def launch_choices(family, kind, m, columns, k):
    """Yields every tiling with every work group count of `launcher.choices`, and with two work groups that compute
    all tiles but the last whole and share the last one."""
    sms = runtime.num_sm(runtime.current_device().device_id)
    for tiling in launcher.TILINGS:
        info = runtime.info(f"fp8_gemm_{family}_{kind}_{tiling}")
        columns_per_tile = launcher.COLUMNS_PER_TILE[kind]
        for work_groups, data_parallel_tiles in launcher.choices(info, columns_per_tile, m, columns, k, sms, True):
            yield launcher.Choice(tiling, work_groups, data_parallel_tiles)
        yield launcher.Choice(tiling, 2, tile_count(family, kind, tiling, m, columns) - 1)


EXACT_FAMILIES = (("tensorwise", False), ("rowwise", False), ("rowwise", True), ("block128", False),
                  ("block128_e8m0", False))


@pytest.mark.parametrize(("mode", "fast"), EXACT_FAMILIES)
def test_exact_products_are_bitwise_on_every_plan(mode, fast, monkeypatch):
    """With exact sums `s`, every tiling, shared tile, raster order and repeated launch gives the bits of PyTorch's
    BF16 epilogue sequence on `fp32(s)` and leaves the workspace counters at zero. The sums of the fast kernels stay
    at or below 2^10, so their accumulators hold them without rounding."""
    generator = generator_of(5)
    m, n, k = 70, 384, 640 if scaling_of(mode) == "block128" else 656
    a, b, scale_a, scale_b = exact_operands(mode, m, n, k, generator)
    _, up, _, scale_up = exact_operands(mode, m, n, k, generator)
    bias, bias_up = (torch.randn(n, device="cuda", generator=generator).bfloat16() for _ in range(2))
    residual = torch.randn(m, n, device="cuda", generator=generator).bfloat16()
    s, s_up = exact_sums(a, scale_a, b, scale_b, mode), exact_sums(a, scale_a, up, scale_up, mode)
    family = launcher.family_of(scaling_of(mode), fast)
    plans = [(kind, choice, None) for kind in ("store", "gated") for choice in launch_choices(family, kind, m, n, k)]
    plans += [(kind, launcher.Choice("64x128", 2, tile_count(family, kind, "64x128", m, n)), raster)
              for kind in ("store", "gated") for raster in RASTERS]
    declare_products(2 * len(plans), m, n, k, gated=True)
    for index, (kind, choice, raster) in enumerate(plans):
        replace(monkeypatch, launcher, "choose", lambda *arguments, choice=choice: choice)
        if raster is not None:
            replace(monkeypatch, launcher, "raster", lambda *arguments, raster=raster: raster)
        activation = ACTIVATIONS[index % len(ACTIVATIONS)]
        with_residual = index % 2 == 1
        with case(kind=kind, tiling=choice.tiling, work_groups=choice.work_groups, whole=choice.data_parallel_tiles,
                  raster=raster):
            for _ in range(2):
                if kind == "store":
                    out = fp8_gemm(a, b, scale_a, scale_b, bias=bias, activation=activation, use_fast_accum=fast,
                                   residual=residual if with_residual else None)
                    expected = store_expected(s, bias, activation, residual if with_residual else None)
                else:
                    out = fp8_gated_gemm(a, b, up, scale_a, scale_b, scale_up, bias_up=bias_up, activation=activation,
                                         residual=residual if with_residual else None, use_fast_accum=fast)
                    expected = gated_expected(s, s_up, None, bias_up, activation, residual if with_residual else None)
                queue_equal_check(out, expected, "out")
    for workspace in common.eager_workspaces.values():
        queue_requirement("workspace counters", workspace.counters.eq(0).all())


def unit_scales(mode, m, columns, k):
    scaling = scaling_of(mode)
    if scaling == "tensorwise":
        return torch.ones((), device="cuda"), torch.ones((), device="cuda")
    if scaling == "rowwise":
        return torch.ones(m, 1, device="cuda"), torch.ones(columns, 1, device="cuda")
    exponents_a = torch.zeros(m, k // SCALE_GROUP, device="cuda", dtype=torch.int32)
    exponents_b = torch.zeros(-(-columns // SCALE_GROUP), k // SCALE_GROUP, device="cuda", dtype=torch.int32)
    if scale_dtype_of(mode) == E8M0:
        return block_scale_a(e8m0_of(exponents_a)), e8m0_of(exponents_b)
    return block_scale_a(torch.exp2(exponents_a.float())), torch.exp2(exponents_b.float())


@pytest.mark.parametrize(("mode", "fast"), EXACT_FAMILIES)
def test_epilogue_inputs_are_bitwise(mode, fast, monkeypatch):
    """With zero sums, each of the 65536 BF16 bit patterns as the bias, the gate bias and the residual gives the
    result of the PyTorch BF16 CUDA ops on every tiling."""
    generator = generator_of(13)
    m, n, k = EPILOGUE_ROWS, ALL_BF16.numel(), SCALE_GROUP
    patterns = ALL_BF16.cuda()
    zeros_a, zeros_b = torch.zeros(m, k, device="cuda").to(FP8), torch.zeros(n, k, device="cuda").to(FP8)
    scale_a, scale_b = unit_scales(mode, m, n, k)
    residual = torch.stack([patterns[torch.randperm(n, device="cuda", generator=generator)] for _ in range(m)])
    bias_up = patterns[torch.randperm(n, device="cuda", generator=generator)]
    s = torch.zeros(m, n, device="cuda")
    family = launcher.family_of(scaling_of(mode), fast)
    sms = runtime.num_sm(runtime.current_device().device_id)
    declare_products(4 * len(launcher.TILINGS) * len(ACTIVATIONS), m, n, k, gated=True)
    for tiling, kind in itertools.product(launcher.TILINGS, ("store", "gated")):
        tiles = tile_count(family, kind, tiling, m, n)
        replace(monkeypatch, launcher, "choose", lambda *arguments, choice=launcher.Choice(tiling, min(tiles, sms),
                                                                                          tiles): choice)
        for activation, with_residual in itertools.product(ACTIVATIONS, (False, True)):
            used_residual = residual if with_residual else None
            if kind == "store":
                out = fp8_gemm(zeros_a, zeros_b, scale_a, scale_b, bias=patterns, activation=activation,
                               residual=used_residual, use_fast_accum=fast)
                expected = store_expected(s, patterns, activation, used_residual)
            else:
                out = fp8_gated_gemm(zeros_a, zeros_b, zeros_b, scale_a, scale_b, scale_b, bias_gate=patterns,
                                     bias_up=bias_up, activation=activation, residual=used_residual,
                                     use_fast_accum=fast)
                expected = gated_expected(s, s, patterns, bias_up, activation, used_residual)
            with case(kind=kind, tiling=tiling, activation=activation, residual=with_residual):
                queue_equal_check(out, expected, "out")


SCALED_MM_SHAPES = ((128, 256, 512), (77, 384, 1024), (300, 512, 4096), (16, 1024, 8192))


@pytest.mark.parametrize("mode", ("tensorwise", "rowwise"))
@pytest.mark.parametrize("fast", (False, True))
def test_products_are_at_most_twice_as_far_from_exact_as_scaled_mm(mode, fast):
    """Each output errs at most twice as much as `torch._scaled_mm`, plus `U` times the largest exact value."""
    generator = generator_of(6)
    for m, n, k in SCALED_MM_SHAPES:
        declare_products(2, m, n, k)
        x, w = practical_inputs(m, n, k, generator, heavy=False)
        a, scale_a, b, scale_b = quantized_operands(mode, x, w)
        r = dequantized(a, scale_a, mode) @ dequantized(b, scale_b, mode, tiles=True).T
        out = fp8_gemm(a, b, scale_a, scale_b, use_fast_accum=fast)
        pt = torch._scaled_mm(a, b.t(), scale_a, scale_b.t(), out_dtype=torch.bfloat16, use_fast_accum=fast)
        error = (out.double() - r).abs().amax()
        allowed = 2 * (pt.double() - r).abs().amax() + U * r.abs().amax()
        with case(m=m, n=n, k=k):
            checks.queue("bound", "largest error", torch.stack([error, allowed]))


def test_fast_accumulation_of_one_chain_stops_growing(monkeypatch):
    """All-ones operands with `k = 32768` give the exact sum with chains of 64 values. One fast chain over all of k
    stops growing before k, because the FP8 tensor cores of sm_89 accumulate with fewer bits than FP32."""
    m, n, k = 16, 128, 32768
    a, b = torch.ones(m, k, device="cuda").to(FP8), torch.ones(n, k, device="cuda").to(FP8)
    one = torch.ones((), device="cuda")
    declare_products(2, m, n, k)
    replace(monkeypatch, launcher, "choose", lambda *arguments: launcher.Choice("128x256", 1, 1))
    queue_equal_check(fp8_gemm(a, b, one, one), torch.full((m, n), k, device="cuda", dtype=torch.bfloat16),
                      "chains of 64 values")
    fast = fp8_gemm(a, b, one, one, use_fast_accum=True)
    stop = fast.float().amax().item()
    print(f"one fast accumulation chain of {k} ones ends at {stop:g}")
    assert 2**14 <= stop < k and bool((fast == stop).all())


def test_any_finite_codes_meet_the_wide_bound():
    """Codes over the whole finite E4M3 range with scales over a wide range, and chains with one product 2^13 times
    larger than the others, meet `gamma = 2^-6` with default accumulation."""
    generator = generator_of(7)
    m, n, k = 64, 256, 1024
    declare_products(8, m, n, k)
    raw = torch.randint(0, 256, (m + n, k), device="cuda", generator=generator, dtype=torch.uint8)
    raw = torch.where(raw & 0x7F == NAN_CODE, raw ^ 1, raw).view(FP8)
    signs = torch.randint(0, 2, (m + n, k), device="cuda", generator=generator).float() * 2 - 1
    spiked = signs.clone()
    spiked[:m, ::64], spiked[m:, ::64] = 256, 32
    for name, codes in (("whole range", raw), ("spiked chains", spiked.to(FP8))):
        a, b = codes[:m], codes[m:]
        for mode in MODES:
            if scaling_of(mode) == "block128":
                exponents = torch.randint(-20, 21, (m, k // SCALE_GROUP), device="cuda", generator=generator)
                tiles = torch.randint(-20, 21, (n // SCALE_GROUP, k // SCALE_GROUP), device="cuda", generator=generator)
                if mode.endswith("e8m0"):
                    scale_a, scale_b = block_scale_a(e8m0_of(exponents)), e8m0_of(tiles)
                else:
                    scale_a = block_scale_a(torch.exp2(exponents.float()) * 1.1)
                    scale_b = torch.exp2(tiles.float()) * 0.9
            elif mode == "rowwise":
                scale_a = torch.exp2(torch.rand(m, 1, device="cuda", generator=generator) * 40 - 20)
                scale_b = torch.exp2(torch.rand(n, 1, device="cuda", generator=generator) * 40 - 20)
            else:
                scale_a, scale_b = torch.full((), 3e-3, device="cuda"), torch.full((), 7e2, device="cuda")
            s, error = sums_and_errors(a, scale_a, b, scale_b, mode, FINITE_GAMMA)
            with case(codes=name, mode=mode):
                queue_bound("out", fp8_gemm(a, b, scale_a, scale_b), *store_bound(s, error, None, "none", None))


def test_e8m0_scales_give_the_bits_of_fp32_scales():
    """Each combination of E8M0 and FP32 block scales, the bytes 0 (2^-127) and 255 (NaN) included, gives the bits
    of the call with `scale.float()`."""
    generator = generator_of(8)
    m, n, k = 100, 264, 512
    declare_products(16, m, n, k, gated=True)
    a = (torch.randn(m, k, device="cuda", generator=generator) * 30).to(FP8)
    b, up = ((torch.randn(n, k, device="cuda", generator=generator) * 30).to(FP8) for _ in range(2))
    tile_rows = -(-n // SCALE_GROUP)
    a_bytes = torch.randint(100, 140, (m, k // SCALE_GROUP), device="cuda", generator=generator).to(torch.uint8)
    a_bytes[3, 1], a_bytes[4, 2], a_bytes[5] = 0, 255, 0
    b_bytes = torch.randint(110, 130, (tile_rows, k // SCALE_GROUP), device="cuda", generator=generator).to(torch.uint8)
    b_bytes[1, 3] = 255
    up_bytes = torch.randint(110, 130, (tile_rows, k // SCALE_GROUP), device="cuda",
                             generator=generator).to(torch.uint8)
    one_byte = (block_scale_a(a_bytes.view(E8M0)), b_bytes.view(E8M0), up_bytes.view(E8M0))
    fp32 = tuple(scale.float() for scale in one_byte)
    expected = (fp8_gemm(a, b, fp32[0], fp32[1]), fp8_gated_gemm(a, b, up, *fp32))
    for choice in itertools.product((False, True), repeat=3):
        scales = [one if use else four for use, one, four in zip(choice, one_byte, fp32)]
        with case(e8m0=choice):
            queue_equal_check(fp8_gemm(a, b, scales[0], scales[1]), expected[0], "store")
            queue_equal_check(fp8_gated_gemm(a, b, up, *scales), expected[1], "gated")


def special_expected(a, scale_a, b, scale_b, mode):
    """Returns the output of exact chain sums under IEEE FP32 scale arithmetic. Rowwise and tensorwise give
    `bf16(fp32(s) * fp32(scale_a * scale_b))`, and block128 adds each chain of 64 values times its factor. A call
    without a bias adds +0, so a product of -0 gives +0, as in `torch._scaled_mm`."""
    m, k, n = a.shape[0], a.shape[1], b.shape[0]
    if scaling_of(mode) != "block128":
        factor = scale_factor(scale_a, mode, m, 1).float() * scale_factor(scale_b, mode, n, 1).float().reshape(1, -1)
        products = (a.double() @ b.double().T).float() * factor
    else:
        chains = torch.einsum("icj,ncj->inc", a.double().reshape(m, k // 64, 64), b.double().reshape(n, k // 64, 64))
        a_factor = scale_a.float().double().repeat_interleave(2, 1)
        b_factor = scale_b.float().double().repeat_interleave(SCALE_GROUP, 0)[:n].repeat_interleave(2, 1)
        factors = (a_factor[:, None, :].float() * b_factor[None, :, :].float()).double()
        products = (chains * factors).sum(-1).float()
    return (products + 0.0).bfloat16()


def test_special_values_follow_ieee_rules():
    """NaN codes, NaN, infinite, subnormal and extreme scales follow IEEE FP32 arithmetic without flush to zero."""
    generator = generator_of(9)
    m, n, k = 64, 256, 256
    declare_products(40, m, n, k)
    a = torch.randint(-1, 2, (m, k), device="cuda", generator=generator).float().to(FP8)
    b = torch.randint(-1, 2, (n, k), device="cuda", generator=generator).float().to(FP8)
    b[7] = 0
    nan_a, nan_b = a.clone(), b.clone()
    nan_a.view(torch.uint8)[3, 17] = NAN_CODE
    nan_b.view(torch.uint8)[200, 5] = NAN_CODE
    ones_m, ones_n = torch.ones(m, 1, device="cuda"), torch.ones(n, 1, device="cuda")
    cases = {"tensorwise NaN codes": ("tensorwise", nan_a, nan_b, torch.ones(()), torch.ones(())),
             "tensorwise overflow": ("tensorwise", a, b, torch.tensor(2.0**100), torch.tensor(2.0**100)),
             "tensorwise underflow": ("tensorwise", a, b, torch.tensor(2.0**-100), torch.tensor(2.0**-100)),
             "tensorwise subnormal": ("tensorwise", a, b, torch.tensor(2.0**-140), torch.tensor(2.0**4)),
             "tensorwise NaN": ("tensorwise", a, b, torch.tensor(math.nan), torch.ones(())),
             "tensorwise infinity": ("tensorwise", a, b, torch.tensor(math.inf), torch.ones(()))}
    row_scales = ones_m.clone()
    row_scales[2], row_scales[5], row_scales[9] = math.inf, math.nan, 0
    column_scales = ones_n.clone()
    column_scales[7], column_scales[11], column_scales[13] = math.inf, math.nan, 2.0**-140
    cases["rowwise NaN codes"] = ("rowwise", nan_a, nan_b, ones_m, ones_n)
    cases["rowwise special scales"] = ("rowwise", a, b, row_scales, column_scales)
    block_a = torch.ones(m, k // SCALE_GROUP, device="cuda")
    block_a[2, 1], block_a[5, 0], block_a[9, 1] = math.inf, math.nan, 0
    block_b = torch.ones(n // SCALE_GROUP, k // SCALE_GROUP, device="cuda")
    block_b[1, 0] = 2.0**-140
    cases["block128 NaN codes"] = ("block128", nan_a, nan_b, block_scale_a(torch.ones_like(block_a)),
                                   torch.ones_like(block_b))
    cases["block128 special scales"] = ("block128", a, b, block_scale_a(block_a), block_b)
    for name, (mode, a_codes, b_codes, scale_a, scale_b) in cases.items():
        scale_a, scale_b = scale_a.cuda(), scale_b.cuda()
        with case(scales=name):
            queue_equal_check(fp8_gemm(a_codes, b_codes, scale_a, scale_b),
                              special_expected(a_codes, scale_a, b_codes, scale_b, mode), "out")
    nan_rows = fp8_gemm(nan_a, nan_b, ones_m, ones_n)
    queue_requirement("NaN code rows", nan_rows[3].isnan().all() & nan_rows[:, 200].isnan().all())
    subnormal = torch.full((m, k), 2.0**-9, device="cuda").to(FP8)
    out = fp8_gemm(subnormal, torch.ones(n, k, device="cuda").to(FP8), ones_m, ones_n)
    exact = k * 2.0**-9
    counted = bool((out == exact).all())
    print(f"subnormal E4M3 operands count {'with their values' if counted else 'as zero'}")
    assert counted or bool((out == 0).all())


def plain_results(a, b, scale_a, scale_b, bias, residual, up, scale_up):
    return (fp8_gemm(a, b, scale_a, scale_b, bias=bias, activation="silu", residual=residual),
            fp8_gated_gemm(a, b, up, scale_a, scale_b, scale_up, bias_up=bias, residual=residual))


class ByteBuffer:
    def __init__(self, address, size):
        self.__cuda_array_interface__ = {"shape": (size,), "typestr": "|u1", "data": (address, False), "version": 3}


def shifted_storage(tensor, offset):
    """Returns a copy of `tensor` whose storage starts `offset` bytes after a multiple of 16 bytes."""
    base = torch.zeros(tensor.numel() * tensor.element_size() + 16, device="cuda", dtype=torch.uint8)
    view = torch.as_tensor(ByteBuffer(base.data_ptr() + offset, tensor.numel() * tensor.element_size()),
                           device="cuda")
    view.copy_(tensor.contiguous().view(-1).view(torch.uint8))
    return view.view(tensor.dtype).view(tensor.shape), base


def test_operand_layouts_give_the_bits_of_plain_layouts():
    """Operands and scales in every layout, read in place or through copies, give the bits of contiguous operands."""
    generator = generator_of(10)
    m, n, k = 74, 264, 512
    declare_products(70, m, n, k, gated=True)
    x, w = practical_inputs(m, n, k, generator, heavy=False)
    a, scale_a, b, scale_b = quantized_operands("block128", x, w)
    _, _, up, scale_up = quantized_operands("block128", x, w * 0.5)
    bias = torch.randn(n, device="cuda", generator=generator).bfloat16()
    residual = torch.randn(m, n, device="cuda", generator=generator).bfloat16()
    expected = plain_results(a, b, scale_a, scale_b, bias, residual, up, scale_up)
    deepgemm = torch.zeros(k // SCALE_GROUP, 76, device="cuda")[:, :m].t()
    deepgemm.copy_(scale_a)
    scrambled = torch.empty(k // SCALE_GROUP, 37, 2, device="cuda").permute(2, 1, 0)
    scrambled.copy_(scale_a.reshape(2, 37, k // SCALE_GROUP))
    padded_b = torch.zeros(3, 8, device="cuda")[:, : k // SCALE_GROUP]
    padded_b.copy_(scale_b)
    transposed_b = scale_b.t().contiguous().t()
    misaligned_a = torch.zeros(m * k + 8, device="cuda", dtype=torch.uint8).view(FP8)[8:].view(m, k)
    misaligned_a.copy_(a)
    wide_b = torch.zeros(n, k + 16, device="cuda", dtype=torch.uint8).view(FP8)[:, 16:]
    wide_b.copy_(b)
    shifted_bias = torch.zeros(n + 1, device="cuda", dtype=torch.bfloat16)[1:]
    shifted_bias.copy_(bias)
    tile_bytes = quantize_fp8_tiles(w, scale_dtype=E8M0)[1]
    odd_bytes, odd_storage = shifted_storage(tile_bytes, 1)
    batched = a.reshape(2, 37, k)
    layouts = {
        "DeepGEMM scale_a": (a, b, deepgemm, scale_b, bias),
        "K-major scale_a": (a, b, scale_a.contiguous(), scale_b, bias),
        "batched scale_a": (batched, b, scale_a.reshape(2, 37, -1), scale_b, bias),
        "copied scale_a": (batched, b, scrambled, scale_b, bias),
        "padded scale_b": (a, b, scale_a, padded_b, bias),
        "transposed scale_b": (a, b, scale_a, transposed_b, bias),
        "misaligned a": (misaligned_a, b, scale_a, scale_b, bias),
        "strided b": (a, wide_b, scale_a, scale_b, bias),
        "shifted bias": (a, b, scale_a, scale_b, shifted_bias),
    }
    for name, (a_in, b_in, scale_a_in, scale_b_in, bias_in) in layouts.items():
        store = fp8_gemm(a_in, b_in, scale_a_in, scale_b_in, bias=bias_in, activation="silu",
                         residual=residual.reshape(*a_in.shape[:-1], n))
        with case(layout=name):
            queue_equal_check(store.reshape(m, n), expected[0], "store")
    assert odd_bytes.untyped_storage().data_ptr() % 4 == 1 and odd_storage.numel() > 0
    queue_equal_check(fp8_gemm(a, b, scale_a, odd_bytes, bias=bias, activation="silu", residual=residual),
                      fp8_gemm(a, b, scale_a, tile_bytes, bias=bias, activation="silu", residual=residual),
                      "E8M0 storage at an odd address")
    in_place = torch.zeros(m, n + 24, device="cuda", dtype=torch.bfloat16)[:, 8 : 8 + n]
    in_place.copy_(residual)
    assert fp8_gemm(a, b, scale_a, scale_b, bias=bias, activation="silu", residual=in_place, out=in_place) is in_place
    queue_equal_check(in_place, expected[0], "residual is out")
    storage = torch.zeros(m + 1, n, device="cuda", dtype=torch.bfloat16)
    shifted_out, overlapping = storage[1:], storage[:-1]
    overlapping.copy_(residual)
    fp8_gemm(a, b, scale_a, scale_b, bias=bias, activation="silu", residual=overlapping, out=shifted_out)
    queue_equal_check(shifted_out, expected[0], "residual overlaps out")
    misaligned_out = torch.zeros(m * n + 1, device="cuda", dtype=torch.bfloat16)[1:].view(m, n)
    fp8_gemm(a, b, scale_a, scale_b, bias=bias, activation="silu", residual=residual, out=misaligned_out)
    queue_equal_check(misaligned_out, expected[0], "misaligned out")
    repeated = residual[:1].expand(m, n)
    queue_equal_check(fp8_gemm(a, b, scale_a, scale_b, residual=repeated),
                      fp8_gemm(a, b, scale_a, scale_b, residual=repeated.contiguous()), "row stride 0")
    fused = torch.cat([b, up])
    fused_scales = torch.cat([scale_b, scale_up])
    tile_rows = scale_b.shape[0]
    out = fp8_gated_gemm(a, fused[:n], fused[n:], scale_a, fused_scales[:tile_rows], fused_scales[tile_rows:],
                         bias_up=bias, residual=residual)
    queue_equal_check(out, expected[1], "fused gated halves")
    out = fp8_gated_gemm(a, wide_b, up, scale_a, scale_b, scale_up, bias_up=bias, residual=residual)
    queue_equal_check(out, expected[1], "gated weights of two row strides")
    misaligned_up = torch.zeros(n * k + 8, device="cuda", dtype=torch.uint8).view(FP8)[8:].view(n, k)
    misaligned_up.copy_(up)
    gated_out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    operands = (a, wide_b, misaligned_up, scale_a, scale_b, scale_up)
    fp8_gated_gemm(*operands, bias_up=bias, residual=residual, out=gated_out)
    allocations = allocation_count()
    fp8_gated_gemm(*operands, bias_up=bias, residual=residual, out=gated_out)
    assert allocation_count() - allocations == 2, "each weight that misses the rules costs one copy"
    queue_equal_check(gated_out, expected[1], "gated weights of two row strides, one of them misaligned")
    empty_rows = fp8_gemm(a[:0], b, scale_a[:0], scale_b)
    empty_columns = fp8_gated_gemm(a, b[:0], up[:0], scale_a, scale_b[:0], scale_up[:0])
    assert empty_rows.shape == (0, n) and empty_columns.shape == (m, 0)


def test_launchers_give_the_bits_of_the_ops():
    """The launchers give the bits of the ops for operands that they read in place, size-1 dimensions of any stride
    included, and for objects that export the CUDA Array Interface. They raise ValueError for an input that the ops
    copy and return at once for `n == 0` and `f == 0`."""
    generator = generator_of(11)
    m, n, k = 512, 1024, 256
    declare_products(20, m, n, k, gated=True)
    stream = torch.cuda.current_stream().cuda_stream
    x, w = practical_inputs(m, n, k, generator, heavy=False)
    a, scale_a, b, scale_b = quantized_operands("rowwise", x, w)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    launcher.gemm(a, b, scale_a, scale_b, out, activation="gelu", stream=stream)
    queue_equal_check(out, fp8_gemm(a, b, scale_a, scale_b, activation="gelu"), "gemm")
    gated = torch.empty(m, n // 2, device="cuda", dtype=torch.bfloat16)
    launcher.gated_gemm(a, b[: n // 2], b[n // 2 :], scale_a, scale_b[: n // 2], scale_b[n // 2 :], gated,
                        stream=stream)
    queue_equal_check(gated, fp8_gated_gemm(a, b[: n // 2], b[n // 2 :], scale_a, scale_b[: n // 2],
                                            scale_b[n // 2 :]), "gated_gemm")
    source = x.float()
    codes = torch.empty(m, k, device="cuda", dtype=torch.uint8).view(FP8)
    scales = torch.empty(k // SCALE_GROUP, m, device="cuda", dtype=E8M0).t()

    class Exported:
        __cuda_array_interface__ = {**source.__cuda_array_interface__, "version": 3}

    launcher.quantize_rows(Exported(), codes, scales, stream=stream)
    q, expected_scales = quantize_fp8_rows(source, scale_dtype=E8M0)
    queue_equal_check(codes, q, "exported input codes")
    queue_equal_check(scales, expected_scales, "exported input scales")
    row_codes = torch.empty(4, 4 * k, device="cuda", dtype=torch.uint8).view(FP8)[:1, :k]
    size_one_strides = {"input": (source.view(m // 2, 2, k)[:, 1:], codes[: m // 2], scales[: m // 2]),
                        "codes": (source[:1], row_codes, scales[:1])}
    for name, (rows, rows_codes, rows_scales) in size_one_strides.items():
        launcher.quantize_rows(rows, rows_codes, rows_scales, stream=stream)
        q, expected_scales = quantize_fp8_rows(rows, scale_dtype=E8M0)
        with case(size_one_dimension_of=name):
            queue_equal_check(rows_codes, q.reshape(rows_codes.shape), "codes")
            queue_equal_check(rows_scales, expected_scales, "scales")
    batches = source[:4].view(2, 2, k).transpose(0, 1)
    with pytest.raises(ValueError, match="flatten into rows of one stride"):
        launcher.quantize_rows(batches, codes[:4], scales[:4], stream=stream)
    q, batch_scales = quantize_fp8_rows(batches, scale_dtype=E8M0)
    expected_q, expected_scales = quantize_fp8_rows(batches.contiguous(), scale_dtype=E8M0)
    queue_equal_check(q, expected_q, "batches whose leading dimensions do not flatten")
    queue_equal_check(batch_scales, expected_scales, "scales of batches whose leading dimensions do not flatten")
    empty = torch.empty(m, 0, device="cuda", dtype=torch.bfloat16)
    launcher.gemm(a, b[:0], scale_a, scale_b[:0], empty, residual=empty, stream=stream)
    launcher.gated_gemm(a, b[:0], b[:0], scale_a, scale_b[:0], scale_b[:0], empty, stream=stream)


def opcheck_cases():
    generator = generator_of(12)
    m, n, k = 40, 256, 512
    x = torch.randn(2, m // 2, k, device="cuda", generator=generator).bfloat16()
    w = torch.randn(n, k, device="cuda", generator=generator) * k**-0.5
    a, scale_a, b, scale_b = quantized_operands("block128", x, w)
    e8m0 = quantized_operands("block128_e8m0", x, w)
    ta, tsa, tb, tsb = quantized_operands("tensorwise", x, w)
    bias = torch.randn(n, device="cuda", generator=generator).bfloat16()
    residual = torch.randn(2, m // 2, n, device="cuda", generator=generator).bfloat16()
    out, half = torch.empty_like(residual), torch.empty(2, m // 2, n // 2, device="cuda", dtype=torch.bfloat16)
    up, gate = b[: n // 2], b[n // 2 :]
    s_up, s_gate = scale_b[:1], scale_b[1:]
    codes = torch.empty(x.shape, device="cuda", dtype=FP8)
    return {
        "fp8_gemm block128": (fp8_gemm_op, a, b, scale_a, scale_b, bias, residual, "silu", False),
        "fp8_gemm e8m0": (fp8_gemm_op, e8m0[0], e8m0[2], e8m0[1], e8m0[3], None, None, "none", False),
        "fp8_gemm tensorwise fast": (fp8_gemm_op, ta, tb, tsa, tsb, bias, None, "gelu", True),
        "fp8_gemm_out": (fp8_gemm_out_op, a, b, scale_a, scale_b, None, residual, "gelu_tanh", False, out),
        "fp8_gated_gemm": (fp8_gated_gemm_op, a, gate, up, scale_a, s_gate, s_up, bias[: n // 2], None, None, "silu",
                           False),
        "fp8_gated_gemm_out": (fp8_gated_gemm_out_op, ta, tb[: n // 2], tb[n // 2 :], tsa, tsb, tsb, None, None,
                               residual[..., : n // 2], "none", True, half),
        "quantize_fp8_rows": (quantize_fp8_rows_op, x, E8M0),
        "quantize_fp8_rows_out": (quantize_fp8_rows_out_op, x.float(), codes, torch.empty(m, k // SCALE_GROUP,
                                                                                          device="cuda")),
        "quantize_fp8_tiles": (quantize_fp8_tiles_op, w[:200], torch.float32),
        "quantize_fp8_tiles_out": (quantize_fp8_tiles_out_op, w, torch.empty(n, k, device="cuda", dtype=FP8),
                                   torch.empty(2, 4, device="cuda", dtype=E8M0)),
        "quantize_fp8_tensorwise": (quantize_fp8_tensorwise_op, x),
        "quantize_fp8_tensorwise_out": (quantize_fp8_tensorwise_out_op, w, torch.empty(n, k, device="cuda",
                                                                                       dtype=FP8),
                                        torch.empty((), device="cuda")),
        "quantize_fp8_static": (quantize_fp8_static_op, x, tsa),
        "quantize_fp8_static_out": (quantize_fp8_static_out_op, x, tsa, codes),
    }


OPCHECK_NAMES = ("fp8_gemm block128", "fp8_gemm e8m0", "fp8_gemm tensorwise fast", "fp8_gemm_out", "fp8_gated_gemm",
                 "fp8_gated_gemm_out", "quantize_fp8_rows", "quantize_fp8_rows_out", "quantize_fp8_tiles",
                 "quantize_fp8_tiles_out", "quantize_fp8_tensorwise", "quantize_fp8_tensorwise_out",
                 "quantize_fp8_static", "quantize_fp8_static_out")


@pytest.mark.parametrize("name", OPCHECK_NAMES)
def test_ops_pass_opcheck(name):
    declare_products(20, 40, 256, 512)
    op, *arguments = opcheck_cases()[name]
    check_op(op, *arguments)


def test_fakes_keep_every_size_symbolic():
    x = torch.empty(3, 5, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(136, 256, device="cuda", dtype=torch.float32)
    a, b = torch.empty(3, 5, 256, device="cuda", dtype=FP8), torch.empty(136, 256, device="cuda", dtype=FP8)
    scale_a, scale_b = torch.empty(3, 5, 2, device="cuda"), torch.empty(2, 2, device="cuda", dtype=E8M0)
    residual = torch.empty(3, 5, 136, device="cuda", dtype=torch.bfloat16)
    scale = torch.empty((), device="cuda")

    def calls(x, w, a, b, scale_a, scale_b, residual, out, scale):
        return (*quantize_fp8_rows(x, scale_dtype=E8M0), *quantize_fp8_tiles(w), *quantize_fp8_tensorwise(x),
                quantize_fp8_tensorwise(x, scale)[0], fp8_gemm(a, b, scale_a, scale_b, residual=residual),
                fp8_gated_gemm(a, b, b, scale_a, scale_b, scale_b), fp8_gemm(a, b, scale_a, scale_b, residual=out,
                                                                             out=out),
                *quantize_fp8_rows(x, out=torch.empty_like(x, dtype=FP8)))

    outputs = call_symbolic(calls, x, w, a, b, scale_a, scale_b, residual, residual.clone(), scale)
    for index, output in enumerate(outputs):
        if index != 5:
            assert_symbolic(output)
    assert outputs[1].dtype == E8M0 and outputs[1].stride(0) == 1 and outputs[5].shape == ()


COMPILED_SIZES = ((40, 256, 384), (72, 384, 512), (136, 512, 640))


def compiled_case(m, n, k):
    """Returns a function that makes the inputs of `compiled_pipeline`. The given outputs start as zeros, so a call
    that skips the write of one of them gives a mismatch."""

    def make():
        generator = generator_of(m * n * k)
        x = torch.randn(2, m, k, device="cuda", generator=generator).bfloat16()
        w = torch.randn(2 * n, k, device="cuda", generator=generator) * k**-0.5
        residual = torch.randn(2, m, n, device="cuda", generator=generator).bfloat16()
        scale = torch.full((1,), 0.0371, device="cuda")
        codes = [torch.zeros(shape, device="cuda", dtype=FP8) for shape in (x.shape, w.shape, x.shape, x.shape)]
        row_scales = torch.zeros(2 * m, k // SCALE_GROUP, device="cuda", dtype=E8M0)
        tile_scales = torch.zeros(2 * n // SCALE_GROUP, k // SCALE_GROUP, device="cuda")
        gated = torch.zeros(2, m, n, device="cuda", dtype=torch.bfloat16)
        return (x, w, residual, residual.clone(), scale, *codes, row_scales, tile_scales,
                torch.zeros((), device="cuda"), gated)

    return make


def compiled_pipeline(x, w, residual, alias, scale, row_codes, tile_codes, tensorwise_codes, static_codes, row_scales,
                      tile_scales, scale_out, gated):
    n = residual.shape[-1]
    qx, sx = quantize_fp8_rows(x)
    qw, sw = quantize_fp8_tiles(w, scale_dtype=E8M0)
    tiles = sw.shape[0] // 2
    tx, tsx = quantize_fp8_tensorwise(x)
    tw, tsw = quantize_fp8_tensorwise(w)
    given = (*quantize_fp8_rows(x, scale_dtype=E8M0, out=row_codes, scales_out=row_scales),
             *quantize_fp8_tiles(w, out=tile_codes, scales_out=tile_scales),
             *quantize_fp8_tensorwise(x, out=tensorwise_codes, scale_out=scale_out),
             quantize_fp8_tensorwise(x, scale, out=static_codes)[0])
    return (fp8_gemm(qx, qw[:n], sx, sw[:tiles], activation="silu", residual=residual),
            fp8_gemm(qx, qw[:n], sx, sw[:tiles], residual=alias, out=alias),
            fp8_gated_gemm(tx, tw[:n], tw[n:], tsx, tsw, tsw, use_fast_accum=True),
            fp8_gated_gemm(qx, qw[:n], qw[n:], sx, sw[:tiles], sw[tiles:], residual=residual, out=gated),
            quantize_fp8_tensorwise(x, scale)[0], *given, alias, row_codes, tile_codes, tensorwise_codes, static_codes,
            row_scales, tile_scales, scale_out, gated)


def same_bits(result, expected):
    def bits(outputs):
        return [output.view(torch.uint8) if output.dtype in (FP8, E8M0) else output for output in outputs]

    assert_same(bits(result), bits(expected))


@pytest.mark.usefixtures("fresh_compiler")
def test_compiled_calls_match_eager_calls():
    """`torch.compile(fullgraph=True, dynamic=True)` compiles every op once for three sizes of every dimension, and
    the compiled calls write the given outputs and the `residual is out` alias as the eager calls do."""
    declare_products(48 * len(COMPILED_SIZES), 2 * 136, 1024, 640, gated=True)
    check_compiled(compiled_pipeline, [compiled_case(*size) for size in COMPILED_SIZES], compare=same_bits)


def test_calls_never_synchronize_and_replay_in_graphs():
    """Every op, split products included, runs under the sync debug mode and in a CUDA graph replayed on new inputs."""
    generator = generator_of(13)
    m, n, k = 32, 256, 4096
    plan = launcher.plan("block128", "store", m, n, k, runtime.current_device().device_id, True)
    assert plan.part_bytes > 0
    x = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
    w = torch.randn(n, k, device="cuda", generator=generator) * k**-0.5
    given = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    gated = torch.empty(m, n // 2, device="cuda", dtype=torch.bfloat16)
    one = torch.ones((), device="cuda")
    codes = [torch.empty(shape, device="cuda", dtype=FP8) for shape in ((m, k), (n, k), (m, k), (n, k))]
    scales = (torch.empty(m, k // SCALE_GROUP, device="cuda"),
              torch.empty(2, k // SCALE_GROUP, device="cuda", dtype=E8M0),
              torch.empty(1, device="cuda"))
    declare_products(40, m, n, k)

    def run():
        qx, sx = quantize_fp8_rows(x)
        qw, sw = quantize_fp8_tiles(w, scale_dtype=E8M0)
        tx, tsx = quantize_fp8_tensorwise(x)
        tw, _ = quantize_fp8_tensorwise(w, one)
        given_rows = quantize_fp8_rows(x, out=codes[0], scales_out=scales[0])
        given_tiles = quantize_fp8_tiles(w, scale_dtype=E8M0, out=codes[1], scales_out=scales[1])
        given_tensorwise = quantize_fp8_tensorwise(x, out=codes[2], scale_out=scales[2])
        given_static = quantize_fp8_tensorwise(w, one, out=codes[3])[0]
        return (fp8_gemm(qx, qw, sx, sw), fp8_gemm(tx, tw, tsx, one, out=given, use_fast_accum=True),
                fp8_gated_gemm(qx, qw[: n // 2], qw[n // 2 :], sx, sw[:1], sw[1:]), tsx,
                fp8_gated_gemm(tx, tw[: n // 2], tw[n // 2 :], tsx, one, one, out=gated), *given_rows, *given_tiles,
                *given_tensorwise, given_static)

    def refresh():
        x.copy_(torch.randn_like(x))
        w.copy_(torch.randn_like(w) * k**-0.5)

    _, outputs = check_graph_replay(run, refresh)
    replayed = [output.clone() for output in outputs]
    for index, (result, expected) in enumerate(zip(replayed, run())):
        if result.dtype in (FP8, E8M0):
            result, expected = result.view(torch.uint8), expected.view(torch.uint8)
        queue_equal_check(result, expected, f"graph output {index}")


def test_split_products_in_concurrent_graphs():
    """Graphs captured on one stream replay at once on two streams beside eager split products on that stream."""
    generator = generator_of(14)
    m, n, k = 40, 384, 2048
    inputs = [exact_operands("rowwise", m, n, k, generator) for _ in range(2)]
    assert launcher.plan("rowwise", "store", m, n, k, runtime.current_device().device_id, True).part_bytes > 0
    declare_products(60, m, n, k)

    def product(operands):
        return fp8_gemm(*operands)

    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        product(inputs[0])
    torch.cuda.current_stream().wait_stream(capture)
    expected = [product(operands) for operands in inputs]
    graphs, outputs = [], []
    for operands in inputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture):
            outputs.append([product(operands) for _ in range(4)])
        graphs.append(graph)
    replays = [torch.cuda.Stream() for _ in graphs]
    for replay in replays:
        replay.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        for graph, replay in zip(graphs, replays):
            with torch.cuda.stream(replay):
                graph.replay()
    with torch.cuda.stream(capture):
        eager = [product(inputs[1]) for _ in range(4)]
    for other in (*replays, capture):
        torch.cuda.current_stream().wait_stream(other)
    for index, (graph_outputs, reference_bits) in enumerate(zip(outputs, expected)):
        for out in graph_outputs:
            queue_equal_check(out, reference_bits, f"graph {index}")
    for out in eager:
        queue_equal_check(out, expected[1], "eager")


def module_ptx(module):
    directory, _ = runtime.manifest()
    path = directory / "ptx" / f"{module}.ptx"
    if not path.exists():
        pytest.skip(f"the build installs no {path}")
    return path.read_text()


def test_scale_and_sum_arithmetic_keeps_subnormal_values():
    """Only the activation helpers flush subnormal values to zero (`ex2.approx.ftz`, `rcp.approx.ftz`), and every
    FP32 `add`, `sub`, `mul`, `fma` and `div` names its rounding so that ptxas does not fuse them (PTX ISA 9.7.3)."""
    for module in ("fp8_gemm_sm89", "fp8_quantize_sm89"):
        text = module_ptx(module)
        flushing = set(re.findall(r"^\s*([a-z][\w.]*\.ftz[\w.]*)", text, re.MULTILINE))
        assert flushing <= {"ex2.approx.ftz.f32", "rcp.approx.ftz.f32"}, (module, flushing)
        arithmetic = set(re.findall(r"^\s*((?:add|sub|mul|fma|div)(?:\.\w+)*\.f32)\b", text, re.MULTILINE))
        assert arithmetic and all(re.search(r"\.r[nmpz]\b", operation) for operation in arithmetic), arithmetic
