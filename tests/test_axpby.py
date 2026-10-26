import itertools
import math

import numpy as np
import pytest
import torch

from checks import case, queue_equal_check
from conftest import cuda_seed, declare_gpu_work
from opcheck import (
    assert_symbolic,
    call_symbolic,
    check_compiled,
    check_graph_replay,
    check_op,
    fresh_compiler,  # noqa: F401
)
from zig_cu_kernels import axpby as launcher
from zig_cu_kernels import runtime
from zig_cu_kernels.torch import axpby, axpby_, axpby_op, axpby_out_op

DTYPES = (torch.bfloat16, torch.float32)
ELEMENTS = {torch.bfloat16: "bf16", torch.float32: "f32"}
BITS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
SIGNIFICAND_BITS = {torch.bfloat16: 7, torch.float32: 23}
FP32_MAX = float(np.finfo(np.float32).max)
FULL_RANGE_COUNT = 8195
EXPONENT_FIELDS = ((0, 255), (123, 131), (0, 4), (250, 254))
ZERO_PERIOD = 61
# The coefficients 1.5, 1 + 2^-17 and 5614251 / 2^22 times inputs with suitable significands give products halfway
# between two FP32 values, so the much smaller `beta * y` decides their rounding. The product of 5614251 / 2^22 and 1.5
# is 2 + 2^-7 + 2^-23, and its lower FP32 neighbor 2 + 2^-7 lies halfway between two BF16 values, so `beta * y` also
# decides the BF16 result.
FULL_RANGE_COEFFICIENTS = (1.0, -1.0, 0.5, 1 / 3, 1.1, -0.7, 0.0, -0.0, 1.5, 1 + 2.0**-17, 5614251 / 2**22, FP32_MAX,
                           -3.0e38, 2.0**-126, -(2.0**-149), 1.0e-40)
SPECIAL_COEFFICIENTS = (math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0e-46, -(2.0**-150), 2.0**-149, 1.0, -1.0, 1 / 3,
                        FP32_MAX)
ALPHA, BETA = 1 / 3, -0.7
LAYOUT_COUNT = 4099
LAUNCHES_PER_CALL = 50
BYTES_PER_ELEMENT = 1000
SLEEP_CYCLES = 100_000_000
# Rounding this integer to a double first and then to FP32 would give 2^60.
INTEGER_ALPHA, INTEGER_ALPHA_FP32 = 2**60 + 2**36 + 1, 2.0**60 + 2.0**37
COMPILED_SHAPES = ((7, 1029), (5, 600), (3, 333))


def declare_calls(calls, elements):
    """Declares `calls` axpby calls on `elements` elements in all, with their copies, references and checks."""
    declare_gpu_work(1, dram_bytes=BYTES_PER_ELEMENT * elements)
    declare_gpu_work(calls * LAUNCHES_PER_CALL)


def fp32(number):
    return float(np.float32(number))


def coefficient_column(numbers):
    return torch.tensor([fp32(number) for number in numbers], dtype=torch.float64, device="cuda").view(-1, 1)


def emulate(alpha, x, beta, y):
    """Returns `fma(a, p, fp32(b * q))` in FP32, rounded to the dtype of `x`. Here `a` and `b` are `alpha` and `beta`
    rounded to FP32, and `p` and `q` are `x` and `y` with +0 for a zero coefficient. The FP64 sum of the exact product
    and `fp32(b * q)` rounds to odd, so an inexact sum becomes the neighboring FP64 value whose last bit is 1. Rounding
    that value to FP32 gives the rounding of the exact sum. `alpha` and `beta` are Python numbers or FP64 tensors that
    broadcast against `x`."""
    a = alpha if isinstance(alpha, torch.Tensor) else torch.tensor(fp32(alpha), dtype=torch.float64, device="cuda")
    b = beta if isinstance(beta, torch.Tensor) else torch.tensor(fp32(beta), dtype=torch.float64, device="cuda")
    p = torch.where(a == 0, 0.0, x.double())
    t = (b * torch.where(b == 0, 0.0, y.double())).float().double()
    product = a * p
    total = product + t
    part = total - product
    error = (product - (total - part)) + (t - part)
    nudged = torch.isfinite(total) & (error != 0) & (total.view(torch.int64) % 2 == 0)
    toward = torch.where(error > 0, math.inf, -math.inf)
    return torch.where(nudged, torch.nextafter(total, toward), total).float().to(x.dtype)


def random_values(count, dtype, seed):
    """Returns `count` values of `dtype` with random signs and significands. Consecutive quarters of the values take
    their exponent fields from the ranges of `EXPONENT_FIELDS`, and every `ZERO_PERIOD`-th value is a signed zero."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    significand_bits = SIGNIFICAND_BITS[dtype]
    width = significand_bits + 9
    words = torch.randint(0, 1 << width, (count,), device="cuda", generator=generator)
    quarters = torch.arange(count, device="cuda") * len(EXPONENT_FIELDS) // count
    fields = torch.tensor(EXPONENT_FIELDS, device="cuda")[quarters]
    draws = torch.randint(0, 1 << 30, (count,), device="cuda", generator=generator)
    exponents = fields[:, 0] + draws % (fields[:, 1] - fields[:, 0] + 1)
    words = words & ~(0xFF << significand_bits) | exponents << significand_bits
    words[::ZERO_PERIOD] &= 1 << (width - 1)
    words = torch.where(words >= 1 << (width - 1), words - (1 << width), words)
    return words.to(BITS[dtype]).view(dtype)


def check_coefficient_pairs(name, coefficients, x, y):
    pairs = list(itertools.product(coefficients, repeat=2))
    declare_calls(len(pairs), len(pairs) * x.numel())
    results = torch.stack([axpby(alpha, x, beta, y) for alpha, beta in pairs])
    alphas, betas = coefficient_column([pair[0] for pair in pairs]), coefficient_column([pair[1] for pair in pairs])
    expected = emulate(alphas, x, betas, y)
    for (alpha, beta), result, row in zip(pairs, results, expected):
        with case(alpha=alpha, beta=beta):
            queue_equal_check(result, row, name)


@pytest.mark.parametrize("dtype", DTYPES)
def test_full_range_is_bitwise_equal_to_the_exact_emulation(dtype):
    x, y = random_values(FULL_RANGE_COUNT, dtype, 1), random_values(FULL_RANGE_COUNT, dtype, 2)
    check_coefficient_pairs("full range", FULL_RANGE_COEFFICIENTS, x, y)


def special_values(dtype):
    info = torch.finfo(dtype)
    tiny = info.smallest_normal * info.eps
    values = [math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0, -1.0, tiny, -tiny, info.smallest_normal - tiny,
              info.smallest_normal, info.max, -info.max, 1 / 3, 3.0]
    column = torch.tensor(values, dtype=dtype, device="cuda")
    return column.repeat_interleave(len(values)), column.repeat(len(values))


@pytest.mark.parametrize("dtype", DTYPES)
def test_special_values_and_zero_coefficients_follow_the_exact_emulation(dtype):
    check_coefficient_pairs("special values", SPECIAL_COEFFICIENTS, *special_values(dtype))


def allocation_count():
    return torch.cuda.memory_stats()["allocation.all.allocated"]


def layout_cases(dtype):
    """Yields the label, arguments and output of each call, and whether the call may allocate memory."""
    def values(*shape):
        return torch.randn(shape, device="cuda", dtype=dtype)

    def skipped(column):
        return torch.full((count, 2), math.nan, device="cuda", dtype=dtype)[:, column]

    count, offset = LAYOUT_COUNT, 16 // dtype.itemsize
    for size in (*range(1, 10), (1 << 20) - 9, (1 << 20) + 3):
        yield f"count {size}", ALPHA, values(size), BETA, values(size), None, True
    for shape in ((), (0,), (4, 0, 3)):
        yield f"shape {shape}", ALPHA, values(*shape), BETA, values(*shape), values(*shape), True
    yield "strided operands", ALPHA, values(64, 130)[:, 1:129], BETA, values(128, 64).t(), None, True
    yield "transposed operands", ALPHA, values(128, 64).t(), BETA, values(128, 64).t(), None, True
    yield "misaligned x", ALPHA, values(count + 1)[1:], BETA, values(count), values(count), True
    yield "strided out", ALPHA, values(64, 128), BETA, values(64, 128), values(64, 130)[:, 1:129], True
    yield "misaligned out", ALPHA, values(count), BETA, values(count), values(count + 1)[1:], True
    x, y = values(count), values(count)
    yield "out is x", ALPHA, x, BETA, y, x, False
    x, y = values(count), values(count)
    yield "out is y", ALPHA, x, BETA, y, y, False
    shared = values(2 * count)
    yield "out partly overlaps x", ALPHA, shared[:count], BETA, values(count), shared[offset:offset + count], True
    shared = values(2 * count)
    yield "out partly overlaps y", ALPHA, values(count), BETA, shared[:count], shared[offset:offset + count], True
    yield "skipped strided x", 0.0, skipped(0), BETA, values(count), values(count), False
    yield "x skipped by 1e-46", 1.0e-46, skipped(1), BETA, values(count), values(count), False
    yield "y skipped by -2^-150", ALPHA, values(count), -(2.0**-150), skipped(1), values(count), False
    shared = values(2 * count)
    out = shared[offset:offset + count]
    yield "out partly overlaps skipped y", ALPHA, values(count), -0.0, shared[:count], out, False


@pytest.mark.parametrize("dtype", DTYPES)
def test_layouts_and_aliases_give_the_exact_emulation(dtype):
    cuda_seed(3)
    cases = list(layout_cases(dtype))
    declare_calls(len(cases), sum(arguments[2].numel() for arguments in cases))
    for label, alpha, x, beta, y, out, may_allocate in cases:
        expected = emulate(alpha, x, beta, y)
        allocations = allocation_count()
        result = axpby(alpha, x, beta, y, out=out)
        assert may_allocate or allocation_count() == allocations, f"{label} copied a tensor"
        assert result is out if out is not None else result.is_contiguous(), label
        queue_equal_check(result, expected, label)


@pytest.mark.parametrize("dtype", DTYPES)
def test_calls_beyond_the_l2_cache_cap_the_resident_work_groups(dtype, monkeypatch):
    device_id = torch.cuda.current_device()
    count = runtime.l2_bytes(device_id) // 2 // dtype.itemsize + 27
    x, y = random_values(count, dtype, 4), random_values(count, dtype, 5)
    out = torch.empty_like(x)
    kernel, info = runtime.load_for(device_id, f"axpby_{ELEMENTS[dtype]}")
    calls = ((ALPHA, BETA), (0.0, BETA))
    shared_sizes = []
    launch = runtime.KernelCall.__call__

    def recording_launch(self, grid, threads, shared_bytes, values, stream):
        shared_sizes.append(shared_bytes)
        launch(self, grid, threads, shared_bytes, values, stream)

    monkeypatch.setattr(runtime.KernelCall, "__call__", recording_launch)
    declare_calls(len(calls), len(calls) * count)
    for alpha, beta in calls:
        axpby(alpha, x, beta, y, out=out)
        queue_equal_check(out, emulate(alpha, x, beta, y), f"alpha={alpha} beta={beta}")
    residency = [runtime.resident_work_groups(kernel, info.threads, shared_bytes) for shared_bytes in shared_sizes]
    assert residency == [launcher.RESIDENT_WORK_GROUPS[(alpha != 0) + (beta != 0)] for alpha, beta in calls]


@pytest.mark.parametrize("dtype", DTYPES)
def test_ops_pass_opcheck(dtype):
    cuda_seed(6)
    y = torch.randn(3, 1029, device="cuda", dtype=dtype)
    for x in (torch.randn_like(y), torch.randn(1029, 3, device="cuda", dtype=dtype).t()):
        check_op(axpby_op, ALPHA, x, BETA, y)
        check_op(axpby_out_op, ALPHA, x, BETA, y, torch.empty_like(y))


def test_out_calls_bump_the_version_of_the_outputs():
    cuda_seed(7)
    x = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    version = out._version
    axpby(ALPHA, x, BETA, y, out=out)
    assert out._version > version
    version = y._version
    axpby_(ALPHA, x, BETA, y)
    assert y._version > version


def test_fake_outputs_stay_symbolic_without_guards():
    x = torch.empty(6, 5, 64, device="cuda", dtype=torch.bfloat16)

    def calls(x, y, out):
        return axpby(ALPHA, x, BETA, y), axpby(ALPHA, x, BETA, y, out=out), axpby_(ALPHA, x, BETA, y)

    new, given, in_place = call_symbolic(calls, x, x.clone(), x.clone())
    for tensor in (new, given, in_place):
        assert_symbolic(tensor)
    assert new.dtype == given.dtype == x.dtype


@pytest.mark.usefixtures("fresh_compiler")
def test_compiled_calls_match_eager_calls():
    cuda_seed(7)
    pairs = [[torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)] for shape in COMPILED_SHAPES]
    declare_calls(6 * len(pairs), 6 * sum(pair[0].numel() for pair in pairs))
    functions = (lambda x, y: axpby(ALPHA, x, BETA, y), lambda x, y: axpby(ALPHA, x, BETA, y, out=x),
                 lambda x, y: axpby_(ALPHA, x, BETA, y))
    for function in functions:
        check_compiled(function, [lambda pair=pair: [tensor.clone() for tensor in pair] for pair in pairs])


def test_calls_never_synchronize_and_replay_in_graphs():
    cuda_seed(8)
    x, y, z = (torch.randn(LAYOUT_COUNT, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    strided = torch.randn(LAYOUT_COUNT, 2, device="cuda", dtype=torch.bfloat16)[:, 0]
    skipped = torch.full_like(x, math.inf)
    out = torch.empty_like(x)
    inputs = {}
    declare_calls(16, 16 * LAYOUT_COUNT)

    def run():
        return (axpby(ALPHA, x, BETA, y), axpby(ALPHA, strided, BETA, y, out=out), axpby_(0.5, x, 1.1, z),
                axpby(0.0, skipped, 2.0, y))

    def refresh():
        for tensor in (x, y, z, strided):
            tensor.copy_(torch.randn_like(tensor))
        inputs.update(x=x.clone(), y=y.clone(), z=z.clone(), strided=strided.clone())

    _, (new, given, in_place, skipping) = check_graph_replay(run, refresh)
    assert given is out and in_place is z
    queue_equal_check(new, emulate(ALPHA, inputs["x"], BETA, inputs["y"]), "new")
    queue_equal_check(out, emulate(ALPHA, inputs["strided"], BETA, inputs["y"]), "out")
    queue_equal_check(z, emulate(0.5, inputs["x"], 1.1, inputs["z"]), "in place")
    queue_equal_check(skipping, emulate(0.0, skipped, 2.0, inputs["y"]), "skipped")


class Exported:
    """Exports a tensor only through DLPack."""

    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, **options):
        return self.tensor.__dlpack__(**options)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def test_launcher_orders_exported_operands_and_validates_its_arguments():
    cuda_seed(9)
    x, y, fresh = (torch.randn(LAYOUT_COUNT, device="cuda") for _ in range(3))
    out = torch.empty_like(x)
    stream = torch.cuda.Stream()
    declare_calls(6, 6 * LAYOUT_COUNT)
    for source in (torch.randn_like(x), fresh):
        torch.cuda._sleep(SLEEP_CYCLES)
        x.copy_(source)
        launcher.axpby(ALPHA, Exported(x), BETA, Exported(y), Exported(out), element="f32", stream=stream)
    torch.cuda.current_stream().wait_stream(stream)
    queue_equal_check(out, emulate(ALPHA, fresh, BETA, y), "exported")
    launcher.axpby(INTEGER_ALPHA, fresh, 0, y, y, element="f32", stream=torch.cuda.current_stream())
    queue_equal_check(y, emulate(INTEGER_ALPHA_FP32, fresh, 0.0, y), "integer alpha")
    skipped = torch.full((LAYOUT_COUNT, 2), math.nan, device="cuda", dtype=torch.bfloat16)[:, 0]
    y, out = torch.randn(LAYOUT_COUNT, device="cuda", dtype=torch.bfloat16), skipped.new_empty(LAYOUT_COUNT)
    launcher.axpby(0, skipped, BETA, y, out, element="bf16", stream=torch.cuda.current_stream())
    queue_equal_check(out, emulate(0.0, skipped, BETA, y), "skipped strided x")
    rows = torch.randn(1, 5, 64, device="cuda", dtype=torch.bfloat16)[:, -1:]
    rows_out = torch.empty(1, 3, 64, device="cuda", dtype=torch.bfloat16)[:, 1:2]
    launcher.axpby(ALPHA, rows, BETA, rows, rows_out, element="bf16", stream=torch.cuda.current_stream())
    queue_equal_check(rows_out, emulate(ALPHA, rows, BETA, rows), "size-1 dimensions of any stride")
    with pytest.raises(ValueError, match="x must hold bfloat16 values, not float16"):
        launcher.axpby(ALPHA, y.half(), BETA, y, y, element="bf16", stream=torch.cuda.current_stream())
    shared = torch.randn(2 * LAYOUT_COUNT, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="out overlaps x without being x"):
        launcher.axpby(ALPHA, shared[:LAYOUT_COUNT], BETA, y, shared[8 : 8 + LAYOUT_COUNT], element="bf16",
                       stream=torch.cuda.current_stream())
