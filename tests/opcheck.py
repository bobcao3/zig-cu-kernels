import contextlib
import functools
import sys

import pytest
import torch
from cuda.core import Device, Event, Stream
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv, StatelessSymbolicContext
from torch.library import opcheck

from conftest import compile_function, declare_gpu_work

OPCHECK_TESTS = ("test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic")
SYNCHRONIZING_TYPES = (Stream, Event, Device)
WAITING_FUNCTIONS = (torch.cuda.synchronize, torch.cuda.Stream.synchronize, torch.cuda.Event.synchronize,
                     torch.accelerator.synchronize)
FREE_TOOL_IDS = (3, 4)


def detached(value):
    return value.detach() if isinstance(value, torch.Tensor) else value


def is_one_byte_float(tensor):
    return tensor.dtype.is_floating_point and tensor.dtype.itemsize == 1


@contextlib.contextmanager
def one_byte_floats_compared_by_bits():
    """Lets `torch.allclose`, which the schema check of opcheck calls on every argument, compare bits of FP8 tensors."""
    allclose = torch.allclose

    def bits_or_allclose(first, second, *options, **named_options):
        if is_one_byte_float(first) and is_one_byte_float(second):
            return torch.equal(first.view(torch.uint8), second.view(torch.uint8))
        return allclose(first, second, *options, **named_options)

    torch.allclose = bits_or_allclose
    try:
        yield
    finally:
        torch.allclose = allclose


def check_op(op, *arguments):
    declare_gpu_work(0, compiled_functions=1)
    with one_byte_floats_compared_by_bits():
        opcheck(op, tuple(map(detached, arguments)), test_utils=OPCHECK_TESTS)


@pytest.fixture
def fresh_compiler():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def symbolic(fake_mode, tensor, static_dims=()):
    sizes = [DimDynamic.STATIC if dim in static_dims else DimDynamic.DYNAMIC for dim in range(tensor.dim())]
    return fake_mode.from_tensor(tensor, symbolic_context=StatelessSymbolicContext(dynamic_sizes=sizes))


def call_symbolic(call, *arguments, static_dims=None):
    """Calls `call` on fake tensor arguments with symbolic sizes and asserts that it adds no guard. The `ShapeEnv`
    specializes sizes 0 and 1 and the dimensions that `static_dims` lists by argument position."""
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    static_dims = static_dims or {}
    fakes = [symbolic(fake_mode, argument, static_dims.get(index, ())) if isinstance(argument, torch.Tensor)
             else argument for index, argument in enumerate(arguments)]
    guards = len(shape_env.guards)
    with fake_mode:
        result = call(*fakes)
    assert len(shape_env.guards) == guards, shape_env.guards[guards:]
    return result


def assert_symbolic(tensor, dims=None):
    sizes = [tensor.shape[dim] for dim in (range(tensor.dim()) if dims is None else dims)]
    assert all(isinstance(size, torch.SymInt) for size in sizes), tensor.shape


def assert_same(result, expected):
    if isinstance(result, torch.Tensor):
        torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
        return
    assert len(result) == len(expected)
    for part, expected_part in zip(result, expected):
        assert_same(part, expected_part)


def check_compiled(function, cases, compare=assert_same):
    """Compares compiled and eager calls of `function` on new arguments from each callable of `cases`."""
    torch._dynamo.reset()
    compiled = compile_function(function, fullgraph=True, dynamic=True)
    for index, case in enumerate(cases):
        with torch.compiler.set_stance("fail_on_recompile" if index else "default"):
            result = compiled(*case())
        compare(result, function(*case()))


@contextlib.contextmanager
def no_host_synchronization():
    """Raises at a call that waits for the device. PyTorch's sync debug mode raises inside PyTorch, and a monitoring
    hook raises at a call from Python code, also through `functools.partial`, of `WAITING_FUNCTIONS` and of `sync` on a
    cuda.core stream, event or device. The hook misses a call that other C code makes, for example inside `map`."""

    def on_call(code, offset, function, first_argument):
        while isinstance(function, functools.partial):
            function, first_argument = function.func, function.args[0] if function.args else None
        target = getattr(function, "__func__", function)
        if any(target is waiting for waiting in WAITING_FUNCTIONS):
            raise RuntimeError(f"{target.__qualname__}() waits for the device")
        owner = getattr(function, "__self__", first_argument)
        if getattr(function, "__name__", None) == "sync" and isinstance(owner, SYNCHRONIZING_TYPES):
            raise RuntimeError(f"{type(owner).__name__}.sync() waits for the device")

    tool = next(tool for tool in FREE_TOOL_IDS if sys.monitoring.get_tool(tool) is None)
    sys.monitoring.use_tool_id(tool, "no_host_synchronization")
    sys.monitoring.register_callback(tool, sys.monitoring.events.CALL, on_call)
    sys.monitoring.set_events(tool, sys.monitoring.events.CALL)
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")
        sys.monitoring.set_events(tool, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.CALL, None)
        sys.monitoring.free_tool_id(tool)


def check_graph_replay(run, refresh):
    run()
    with no_host_synchronization():
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    refresh()
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs
