import contextlib
import fractions
import functools
import itertools
import math
import random
import threading
from typing import Optional

import numpy as np
import pytest
import torch
from cuda.core import Device
from torch._subclasses.fake_tensor import FakeTensorMode

from conftest import compile_function, cuda_seed, declare_gpu_work
from opcheck import (
    assert_same,
    assert_symbolic,
    call_symbolic,
    check_compiled,
    check_graph_replay,
    check_op,
    fresh_compiler,  # noqa: F401
    no_host_synchronization,
)
import zig_cu_kernels.torch as torch_layer
from zig_cu_kernels.torch import common
from zig_cu_kernels.torch.common import (
    FP32_OVERFLOW,
    PartialSumTensors,
    activation_block_scales,
    current_stream_handle,
    fp32_scalar,
    has_scale_shape,
    is_exact_alias,
    overlaps,
    require_choice,
    require_dtype,
    require_scale_dtype,
    require_tensorwise_scale,
    row_layout,
    scale_layout,
    shares_memory,
    shares_memory_within,
    workspace_for,
    writes_in_place,
)
from zig_cu_kernels.torch.ops import NAMESPACE, Output, define

FLOAT_DTYPES = (torch.bfloat16, torch.float32)
COMPILED_SIZES = ((3, 40), (5, 24), (7, 72))
LAYOUT_SAMPLES = 3000
SPLIT_COUNTERS = 8
GRAPH_SHAPES = ((64, 64), (32, 128))
GRAPH_ROUNDS = 4


def check_add_amax(x: torch.Tensor, residual: Optional[torch.Tensor], scale: float, out_dtype: torch.dtype):
    require_dtype("x", x, FLOAT_DTYPES)
    require_dtype("residual", residual, (x.dtype,))
    require_choice("out_dtype", out_dtype, FLOAT_DTYPES)
    if x.dim() < 1:
        raise ValueError("x needs at least one dimension")


def check_add_amax_sizes(x, residual, scale, out_dtype):
    if residual is not None and residual.shape != x.shape:
        raise ValueError(f"residual must have the shape {tuple(x.shape)}")


def scaled_like(x: torch.Tensor, residual: Optional[torch.Tensor], scale: float,
                out_dtype: torch.dtype) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=out_dtype)


def amax_like(x: torch.Tensor, residual: Optional[torch.Tensor], scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)


def sum_like(x: torch.Tensor, residual: Optional[torch.Tensor], scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    return x.new_empty(x.shape)


def add_amax_into(x: torch.Tensor, residual: Optional[torch.Tensor], scale: float, y: torch.Tensor,
                  amax: torch.Tensor, h: torch.Tensor) -> None:
    total = x if residual is None else x + residual
    amax.copy_(total.float().abs().amax(-1, keepdim=True))
    y.copy_(total.float() * scale)
    h.copy_(total)


ADD_AMAX = define(
    "add_amax",
    run=add_amax_into,
    outputs={"y": Output(scaled_like, "out_dtype", "out"), "amax": Output(amax_like, torch.float32, "amax_out"),
             "h": Output(sum_like, "x", "residual_out")},
    check=check_add_amax,
    check_sizes=check_add_amax_sizes,
    namespace="zig_cu_kernels_testing",
)


def add_amax(x, scale, *, residual=None, out_dtype=torch.float32, out=None, amax_out=None, residual_out=None):
    outputs = {"residual_out": residual_out, "out": out, "amax_out": amax_out}
    return ADD_AMAX(x, residual, scale, out_dtype, outputs=outputs)


def check_scale(x: torch.Tensor, scale: float):
    require_dtype("x", x, FLOAT_DTYPES)


def scale_like(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.new_empty(x.shape)


def scale_into(x: torch.Tensor, scale: float, out: torch.Tensor) -> None:
    out.copy_(x * scale)


SCALE = define("scale", run=scale_into, outputs={"out": Output(scale_like, "x")}, check=check_scale,
               namespace="zig_cu_kernels_testing")


def check_dual(x: torch.Tensor, fp8_scale: Optional[torch.Tensor], eps: float):
    require_dtype("x", x, (torch.bfloat16,))


def dual_dtype(x: torch.Tensor, fp8_scale: Optional[torch.Tensor], eps: float):
    return torch.bfloat16 if fp8_scale is None else torch.float8_e4m3fn


def normed_like(x: torch.Tensor, fp8_scale: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    return x.new_empty(x.shape, dtype=dual_dtype(x, fp8_scale, eps))


def residual_out_like(x: torch.Tensor, fp8_scale: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    return x.new_empty(x.shape)


def dual_reference(x, fp8_scale, eps):
    total = x + eps
    return total if fp8_scale is None else (total.float() / fp8_scale).to(torch.float8_e4m3fn), total


def dual_into(x: torch.Tensor, fp8_scale: Optional[torch.Tensor], eps: float, out: torch.Tensor,
              residual_out: torch.Tensor) -> None:
    normed, total = dual_reference(x, fp8_scale, eps)
    out.copy_(normed)
    residual_out.copy_(total)


DUAL = define("dual", run=dual_into, check=check_dual, namespace="zig_cu_kernels_testing",
              outputs={"out": Output(normed_like, dual_dtype), "residual_out": Output(residual_out_like, "x")})


def split_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape)


def split_into(x: torch.Tensor, out: torch.Tensor) -> None:
    device_index = x.get_device()
    with workspace_for(device_index, current_stream_handle(device_index)) as workspace:
        workspace.addresses(x.numel() * x.element_size(), SPLIT_COUNTERS)
        parts = workspace.parts[: x.numel() * x.element_size()].view(x.dtype).view(x.shape)
        counters = workspace.counters[:SPLIT_COUNTERS]
        parts.copy_(x)
        counters.add_(1)
        out.copy_(parts * (counters.sum() / SPLIT_COUNTERS).to(x.dtype))
        counters.zero_()


SPLIT = define("split", run=split_into, outputs={"out": Output(split_like, "x")}, check=lambda x: None,
               namespace="zig_cu_kernels_testing")


def add_amax_reference(x, scale, residual=None, out_dtype=torch.float32):
    total = x if residual is None else x + residual
    return (total.float() * scale).to(out_dtype), total.float().abs().amax(-1, keepdim=True), total.clone()


def test_schemas_follow_the_annotations():
    functional = torch.ops.zig_cu_kernels_testing.add_amax.default._schema
    out = torch.ops.zig_cu_kernels_testing.add_amax_out.default._schema
    assert str(functional) == ("zig_cu_kernels_testing::add_amax(Tensor x, Tensor? residual, float scale, "
                               "ScalarType out_dtype) -> (Tensor, Tensor, Tensor)")
    assert str(out) == ("zig_cu_kernels_testing::add_amax_out(Tensor x, Tensor? residual, float scale, "
                        "Tensor(a3!) y, Tensor(a4!) amax, Tensor(a5!) h) -> ()")
    assert str(torch.ops.zig_cu_kernels_testing.scale.default._schema) == (
        "zig_cu_kernels_testing::scale(Tensor x, float scale) -> Tensor")
    assert str(torch.ops.zig_cu_kernels_testing.scale_out.default._schema) == (
        "zig_cu_kernels_testing::scale_out(Tensor x, float scale, Tensor(a2!) out) -> ()")
    assert str(torch.ops.zig_cu_kernels_testing.dual_out.default._schema) == (
        "zig_cu_kernels_testing::dual_out(Tensor x, Tensor? fp8_scale, float eps, Tensor(a3!) out, "
        "Tensor(a4!) residual_out) -> ()")


@pytest.mark.parametrize("with_residual", [False, True])
def test_ops_pass_opcheck(with_residual):
    cuda_seed(1)
    x = torch.randn(4, 3, 40, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if with_residual else None
    check_op(ADD_AMAX.functional, x, residual, 0.5, torch.float32)
    outputs = [torch.empty_like(tensor) for tensor in add_amax_reference(x, 0.5, residual)]
    check_op(ADD_AMAX.out, x, residual, 0.5, *outputs)


def test_single_output_ops_return_tensors():
    cuda_seed(2)
    x = torch.randn(6, 24, device="cuda", dtype=torch.bfloat16)
    check_op(SCALE.functional, x, 3.0)
    check_op(SCALE.out, x, 3.0, torch.empty_like(x))
    expected = x * 3.0
    assert_same(SCALE(x, 3.0), expected)
    assert SCALE(x, 3.0, outputs={"out": x}) is x
    assert_same(x, expected)


def as_float(tensors):
    return tuple(tensor.float() for tensor in tensors)


def test_computed_dtypes_follow_the_arguments():
    cuda_seed(8)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    for fp8_scale in (None, torch.full((1, 1), 0.5, device="cuda")):
        expected = as_float(dual_reference(x, fp8_scale, 0.25))
        check_op(DUAL.functional, x, fp8_scale, 0.25)
        check_op(DUAL.out, x, fp8_scale, 0.25, torch.empty_like(x, dtype=dual_dtype(x, fp8_scale, 0.25)),
                 torch.empty_like(x))
        normed, total = DUAL(x, fp8_scale, 0.25)
        assert normed.dtype == dual_dtype(x, fp8_scale, 0.25)
        out = torch.empty_like(normed)
        given = DUAL(x, fp8_scale, 0.25, outputs={"out": out})
        assert given[0] is out
        for result in ((normed, total), given):
            assert_same(as_float(result), expected)
        with pytest.raises(ValueError, match=f"out must be a {normed.dtype} tensor"):
            DUAL(x, fp8_scale, 0.25, outputs={"out": torch.empty_like(x, dtype=torch.float32)})


def test_fake_outputs_stay_symbolic_without_guards():
    x = torch.empty(6, 5, 64, device="cuda", dtype=torch.bfloat16)

    def functional_and_out(x, residual, h):
        y, amax, h = add_amax(x, 0.5, residual=residual, out_dtype=torch.bfloat16, residual_out=h)
        return (y, amax, h), add_amax(x, 0.5, residual=residual)

    (y, amax, h), (new_y, new_amax, new_h) = call_symbolic(functional_and_out, x, x.clone(), x.clone())
    for tensor in (y, h, new_y, new_h):
        assert_symbolic(tensor)
    for tensor in (amax, new_amax):
        assert_symbolic(tensor, (0, 1))
    assert (y.dtype, new_y.dtype, amax.dtype, h.dtype) == (torch.bfloat16, torch.float32, torch.float32, x.dtype)


def test_activation_block_scales_have_the_blockwise_1x128_layout():
    x = torch.empty(3, 5, 384)
    scales = activation_block_scales(x, torch.float8_e8m0fnu)
    assert (scales.shape, scales.stride(), scales.dtype) == ((15, 3), (1, 15), torch.float8_e8m0fnu)
    symbolic_scales = call_symbolic(lambda x: activation_block_scales(x, torch.float32), x)
    assert_symbolic(symbolic_scales)
    assert symbolic_scales.stride(0) == 1


@pytest.mark.usefixtures("fresh_compiler")
def test_compiled_calls_match_eager_calls():
    cuda_seed(3)
    inputs = [[torch.randn(*size, device="cuda", dtype=torch.bfloat16) for _ in range(2)] for size in COMPILED_SIZES]

    def separate(x, residual):
        return add_amax(x, 0.5, residual=residual, out_dtype=torch.bfloat16)

    def into_x(x, residual):
        return add_amax(x, 0.5, residual=residual, residual_out=x)

    for function in (separate, into_x):
        check_compiled(function, [lambda pair=pair: [tensor.clone() for tensor in pair] for pair in inputs])


def test_calls_never_synchronize_and_replay_in_graphs():
    cuda_seed(4)
    x, residual = (torch.randn(8, 96, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    out = torch.empty(8, 96, device="cuda", dtype=torch.float32)

    def run():
        return add_amax(x, 0.25, residual=residual, out=out), add_amax(x, 0.25)

    def refresh():
        x.copy_(torch.randn_like(x))
        residual.copy_(torch.randn_like(residual))

    graph, (given, new) = check_graph_replay(run, refresh)
    assert given[0] is out
    assert_same(given, add_amax_reference(x, 0.25, residual))
    assert_same(new, add_amax_reference(x, 0.25))
    stream = Device(0).create_stream()
    event = torch.cuda.Event()
    event.record()
    waits = (stream.sync, lambda: stream.record().sync(), lambda: Device(0).sync(), functools.partial(stream.sync),
             torch.cuda.synchronize, event.synchronize, lambda: torch.ones(1, device="cuda").item())
    for wait in waits:
        with pytest.raises(RuntimeError, match="sync"), no_host_synchronization():
            wait()


def test_public_calls_return_the_given_outputs():
    cuda_seed(5)
    x, residual = (torch.randn(4, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    expected = add_amax_reference(x, 2.0, residual, torch.bfloat16)
    for given in itertools.product((False, True), repeat=3):
        outputs = [torch.empty_like(tensor) if present else None for tensor, present in zip(expected, given)]
        result = add_amax(x, 2.0, residual=residual, out_dtype=torch.bfloat16, out=outputs[0], amax_out=outputs[1],
                          residual_out=outputs[2])
        assert all(output is None or output is part for output, part in zip(outputs, result))
        assert_same(result, expected)
    with pytest.raises(ValueError, match="out must be a torch.float32 tensor"):
        add_amax(x, 2.0, residual=residual, out=torch.empty_like(expected[0]))
    with pytest.raises(TypeError, match="add_amax has no output y"):
        ADD_AMAX(x, residual, 2.0, torch.bfloat16, outputs={"y": torch.empty_like(expected[0])})
    halves = torch.empty(4, 64, device="cuda", dtype=torch.bfloat16)
    left, right = halves[:, :32], halves[:, 32:]
    assert_same(add_amax(x, 2.0, residual=residual, out_dtype=torch.bfloat16, out=left, residual_out=right), expected)
    with pytest.raises(ValueError, match="y and h share memory"):
        add_amax(x, 2.0, residual=residual, out_dtype=torch.bfloat16, out=left, residual_out=left)
    with pytest.raises(ValueError, match="the elements of y share memory"):
        add_amax(x, 2.0, residual=residual, out_dtype=torch.bfloat16, out=left[:1].expand(4, 32))


def test_the_package_exports_every_op():
    namespace = getattr(torch.ops, NAMESPACE)
    names = [name for name in dir(namespace) if isinstance(getattr(namespace, name), torch._ops.OpOverloadPacket)]
    assert names
    for name in names:
        exported = getattr(torch_layer, f"{name}_op", None)
        assert exported is getattr(namespace, name).default, name


def test_size_checks_come_before_allocations(monkeypatch):
    allocations = []

    def counted(*arguments):
        allocations.append(arguments[0].shape)
        return scaled_like(*arguments)

    monkeypatch.setattr(ADD_AMAX, "outputs", [Output(counted, "out_dtype"), *ADD_AMAX.outputs[1:]])
    x = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    add_amax(x, 1.0, residual=x)
    with pytest.raises(ValueError, match="residual must have the shape"):
        add_amax(x, 1.0, residual=x[:, :8])
    assert allocations == [x.shape]


def test_every_tensor_must_be_on_the_device_of_the_first():
    cuda_seed(6)
    source = torch.randn(4, 16, device="cuda", dtype=torch.bfloat16)
    expected = add_amax_reference(source, 0.5, source)[0]
    for x_device, residual_device, out_device in itertools.product(("cpu", "cuda"), repeat=3):
        x, residual = source.to(x_device), source.to(residual_device)
        out = torch.empty(4, 16, device=out_device)
        devices = {x_device, residual_device, out_device}
        if devices == {"cuda"}:
            assert_same(add_amax(x, 0.5, residual=residual, out=out)[0], expected)
            continue
        error = NotImplementedError if devices == {"cpu"} else ValueError
        with pytest.raises(error):
            add_amax(x, 0.5, residual=residual, out=out)
    assert SCALE.functional(source.to("meta"), 2.0).device.type == "meta"
    fake_mode = FakeTensorMode()
    x, residual = fake_mode.from_tensor(source), fake_mode.from_tensor(source.cpu())
    with fake_mode, pytest.raises(ValueError, match="residual must be on cuda:0"):
        ADD_AMAX.functional(x, residual, 0.5, torch.float32)


def fp32_nearest(value):
    exact = fractions.Fraction(value)
    with np.errstate(over="ignore"):
        guess = np.float32(float(exact))
        neighbors = {float(np.nextafter(guess, direction, dtype=np.float32)) for direction in (-np.inf, np.inf)}
    candidates = [candidate for candidate in neighbors | {float(guess)} if math.isfinite(candidate)]
    return min(candidates, key=lambda candidate: (abs(fractions.Fraction(candidate) - exact),
                                                  int(np.float32(candidate).view(np.uint32)) & 1))


def test_scalars_round_to_fp32():
    numbers = [0.1, -0.0, 1e-45, 3e-46, 2.0**-149, 3.4028234663852886e38, float(FP32_OVERFLOW),
               math.nextafter(float(FP32_OVERFLOW), 0.0), -float(FP32_OVERFLOW), math.inf, -math.inf, 7, -(2**24 + 1),
               2**64 + 2**40, 2**64 + 2**40 + 1, FP32_OVERFLOW, FP32_OVERFLOW - 1, -FP32_OVERFLOW, 10**400]
    for value in numbers:
        if not (isinstance(value, float) and math.isinf(value)) and abs(value) >= FP32_OVERFLOW:
            with pytest.raises(OverflowError):
                fp32_scalar("alpha", value)
            continue
        rounded = np.float32(fp32_scalar("alpha", value))
        assert float(rounded) == (value if math.isinf(value) else fp32_nearest(value)), value
        assert math.copysign(1.0, rounded) == math.copysign(1.0, value)
    assert math.isnan(fp32_scalar("alpha", math.nan))
    for value in (torch.tensor(1.0), True, "1", None, np.float32(1.0)):
        with pytest.raises(TypeError):
            fp32_scalar("alpha", value)
    with pytest.raises(TypeError, match="scale must be a Python number"):
        add_amax(torch.ones(2, 8), torch.tensor(0.5))
    with pytest.raises(TypeError, match="residual must be a tensor"):
        add_amax(torch.ones(2, 8), 0.5, residual=1.0)


def random_view(generator, bases):
    base = generator.choice(bases)
    rank = generator.randrange(4)
    sizes = [generator.choice((0, 1, 2, 3, 4, 5)) if generator.random() < 0.05 else generator.randrange(1, 6)
             for _ in range(rank)]
    if generator.random() < 0.5:
        strides = [1] * rank
        row_stride = generator.randrange(0, 12)
        for dim in range(rank - 2, -1, -1):
            strides[dim] = row_stride
            row_stride *= sizes[dim]
    else:
        strides = [generator.randrange(0, 10) for _ in range(rank)]
    return torch.as_strided(base, sizes, strides, generator.randrange(0, 40))


def element_bytes(tensor):
    offsets = [0]
    for size, stride in zip(tensor.shape, tensor.stride()):
        offsets = [offset + index * stride for offset in offsets for index in range(size)]
    element = tensor.element_size()
    return {tensor.data_ptr() + offset * element + byte for offset in offsets for byte in range(element)}, offsets


def is_superincreasing(tensor):
    span = 0
    for stride, size in sorted((stride, size) for size, stride in zip(tensor.shape, tensor.stride()) if size > 1):
        if stride <= span:
            return False
        span += (size - 1) * stride
    return True


def test_memory_tests_match_the_element_bytes():
    generator = random.Random(7)
    storage = torch.zeros(1024)
    bases = (storage, storage.view(torch.bfloat16))
    for _ in range(LAYOUT_SAMPLES):
        first, second = random_view(generator, bases), random_view(generator, bases)
        (first_bytes, first_offsets), (second_bytes, _) = element_bytes(first), element_bytes(second)
        spans_meet = bool(first_bytes and second_bytes) and min(first_bytes) <= max(second_bytes) and \
            min(second_bytes) <= max(first_bytes)
        assert overlaps(first, second) == spans_meet
        layouts = row_layout(first), row_layout(second)
        if layouts[0] is not None and first.numel():
            rows, width, stride = layouts[0]
            covered = {first.data_ptr() + row * stride + byte for row in range(rows) for byte in range(width)}
            assert covered == first_bytes
        exact = None not in layouts and (layouts[0][2] == layouts[1][2] or min(layouts[0][0], layouts[1][0]) <= 1)
        assert shares_memory(first, second) == (bool(first_bytes & second_bytes) if exact else spans_meet)
        repeated = len(set(first_offsets)) < len(first_offsets)
        within_exact = sum(size > 1 for size in first.shape) <= 2 or is_superincreasing(first)
        assert shares_memory_within(first) == repeated if within_exact else shares_memory_within(first)


def test_writes_in_place_allows_only_exact_aliases():
    buffer = torch.zeros(8, 32, dtype=torch.bfloat16)
    out, left, right = buffer[:4], buffer[4:, :16], buffer[4:, 16:]
    shifted = buffer[1:5]
    assert is_exact_alias(out, buffer[:4]) and not is_exact_alias(out, shifted)
    assert writes_in_place(out, (left, right), (buffer[:4],))
    assert not writes_in_place(out, (buffer[:4],))
    assert not writes_in_place(out, (), (shifted,))
    assert not shares_memory(left, right) and overlaps(left, right)
    even, odd = buffer[:, 0::2], buffer[:, 1::2]
    assert row_layout(even) is None and shares_memory(even, odd)


class ByteBuffer:
    def __init__(self, address, size):
        self.__cuda_array_interface__ = {"shape": (size,), "typestr": "|u1", "data": (address, False), "version": 3}


def test_scale_layouts_follow_the_scaling_modes():
    lead, groups, tile_rows = (2, 3), 3, 2
    scale_a = activation_block_scales(torch.empty(*lead, groups * 128), torch.float8_e8m0fnu)
    layouts = [(scale_a, lead, (1, 6)), (torch.empty(groups, 8)[:, :6].t(), lead, (1, 8)),
               (torch.empty(*lead, groups), lead, (groups, 1)), (torch.empty(*lead, 4)[..., :groups], lead, (4, 1)),
               (torch.empty(2, 4, groups)[:, :3], lead, None),
               (torch.empty(tile_rows, groups), (tile_rows,), (groups, 1)),
               (torch.empty(tile_rows, 4)[:, :groups], (tile_rows,), (4, 1)),
               (torch.empty(groups, tile_rows).t(), (tile_rows,), (1, tile_rows))]
    for scale, scale_lead, expected in layouts:
        assert has_scale_shape(scale, scale_lead, groups)
        assert scale_layout("scale", scale, scale_lead, groups) == expected
    assert scale_layout("scale_b", torch.empty(1, 4)[:, :1], (1,), 1, (torch.float32,)) == (0, 1)
    base = torch.zeros(64, dtype=torch.uint8, device="cuda")
    shifted = torch.as_tensor(ByteBuffer(base.data_ptr() + 1, 6), device="cuda").view(torch.float8_e8m0fnu)
    offset = torch.empty(7, dtype=torch.float8_e8m0fnu, device="cuda")[1:]
    assert shifted.untyped_storage().data_ptr() % 4 == 1 and offset.data_ptr() % 4 == 1
    assert scale_layout("scale_b", shifted.view(2, 3), (2,), 3) is None
    assert scale_layout("scale_b", offset.view(2, 3), (2,), 3) == (3, 1)
    assert not has_scale_shape(torch.empty(3, 6), lead, groups)
    with pytest.raises(ValueError, match=r"scale_a must have the shape \(6, 3\) or \(2, 3, 3\), got \(3, 6\)"):
        scale_layout("scale_a", torch.empty(3, 6), lead, groups)
    with pytest.raises(ValueError, match="scale_b must be a torch.float32 tensor"):
        scale_layout("scale_b", shifted.view(6, 1), (6,), 1, (torch.float32,))
    require_tensorwise_scale("scale", torch.ones(1, 1, 1))
    for value in (0.5, torch.ones(2), torch.ones(1, dtype=torch.bfloat16)):
        with pytest.raises(TypeError if isinstance(value, float) else ValueError):
            require_tensorwise_scale("scale", value)
    require_scale_dtype("scale_dtype", torch.float8_e8m0fnu, "block128")
    with pytest.raises(ValueError, match="scale_dtype must be one of torch.float32"):
        require_scale_dtype("scale_dtype", torch.float8_e8m0fnu, "rowwise")


def test_workspace_grows_to_twice_its_size_and_keeps_counters_zero():
    workspace = PartialSumTensors(0)
    sizes = []
    with workspace:
        for part_bytes, counters in ((100, 10), (150, 5), (10, 30), (90, 30)):
            addresses = workspace.addresses(part_bytes, counters)
            assert addresses == (workspace.parts.data_ptr(), workspace.counters.data_ptr())
            sizes.append((workspace.parts.numel(), workspace.counters.numel()))
    assert sizes == [(100, 10), (200, 10), (200, 30), (200, 30)]
    assert workspace.counters.count_nonzero().item() == 0
    for hold in (contextlib.nullcontext(), workspace.lock):
        with hold, pytest.raises(RuntimeError, match="must hold the workspace"):
            workspace.addresses(1000, 1)
    assert workspace.parts.numel() == 200


def test_workspace_belongs_to_one_thread_at_a_time():
    workspace = PartialSumTensors(0)
    started, refused = threading.Event(), []

    def grow():
        started.set()
        with workspace:
            workspace.addresses(1 << 20, 1)

    def take_without_holding():
        try:
            workspace.addresses(1 << 21, 1)
        except RuntimeError as error:
            refused.append(error)

    with workspace:
        workspace.addresses(16, 1)
        grower, taker = threading.Thread(target=grow), threading.Thread(target=take_without_holding)
        grower.start()
        started.wait()
        taker.start()
        taker.join()
        assert workspace.parts.numel() == 16 and len(refused) == 1
    grower.join()
    assert workspace.parts.numel() == 1 << 20


def pool_of(tensor):
    for segment in torch.cuda.memory_snapshot():
        if segment["address"] <= tensor.data_ptr() < segment["address"] + segment["total_size"]:
            return segment["segment_pool_id"]


def test_workspaces_follow_streams_and_graph_captures():
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    pools = []
    with torch.cuda.stream(first):
        eager = workspace_for(0, first.cuda_stream)
        assert workspace_for(0, first.cuda_stream) is eager and workspace_for(0, second.cuda_stream) is not eager
        for part_bytes, counters in ((1 << 22, 8), (1 << 23, 16)):
            with eager:
                eager.addresses(part_bytes, counters)
            pools.append(pool_of(eager.parts))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    segments = {segment["segment_pool_id"] for segment in torch.cuda.memory_snapshot()}
    assert pools[0] not in segments and pools[1] == pool_of(eager.counters) != (0, 0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=first):
        captured = [workspace_for(0, current_stream_handle(0)) for _ in range(2)]
        for workspace in captured:
            with workspace:
                workspace.addresses(64, 8)
        snapshot = captured[0].counters.clone()
        captured[0].counters.fill_(7)
    assert captured[0] is not captured[1] and eager not in captured
    assert pool_of(captured[0].counters) == graph.pool() != pools[1]
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert snapshot.count_nonzero().item() == 0 and bool(captured[0].counters.eq(7).all())
    assert workspace_for(0, first.cuda_stream) is eager


@pytest.mark.usefixtures("fresh_compiler")
def test_split_workspaces_work_in_cuda_graph_trees():
    cuda_seed(9)
    assert torch._inductor.config.triton.slow_path_cudagraph_asserts
    compiled = compile_function(lambda x: SPLIT(x * 2) + 1, mode="reduce-overhead", dynamic=False)
    declare_gpu_work(0, compiled_functions=len(GRAPH_SHAPES) - 1)
    streams = set(common.eager_workspaces)
    for _ in range(GRAPH_ROUNDS):
        for shape in GRAPH_SHAPES:
            x = torch.randn(shape, device="cuda")
            assert_same(compiled(x).clone(), x * 2 + 1)
    assert any(common.eager_workspaces[key].parts.numel() for key in set(common.eager_workspaces) - streams)
