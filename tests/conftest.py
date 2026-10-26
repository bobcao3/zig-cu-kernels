import faulthandler
import os
import time

import pytest
import torch

import checks
from zig_cu_kernels import roofline

UNDECLARED_SOL_SECONDS = 0.25
OUTSIDE_TESTS_SECONDS = 30.0


class Watchdog:
    """The faulthandler timer runs in a C thread without the GIL, so it fires even while a CUDA call holds the GIL."""

    def __init__(self):
        self.stderr = None
        self.scale = 1.0
        self.start = None
        self.sol_seconds = 0.0
        self.launches = 0
        self.compiled_functions = 0
        self.declared = False

    def arm(self, seconds, start=None):
        deadline = (time.monotonic() if start is None else start) + seconds * self.scale
        faulthandler.dump_traceback_later(max(deadline - time.monotonic(), 1e-3), exit=True, file=self.stderr)

    def time_limit(self):
        if not self.declared:
            return roofline.time_limit(UNDECLARED_SOL_SECONDS, 0)
        return roofline.time_limit(self.sol_seconds, self.launches, self.compiled_functions)

    def begin_test(self):
        self.start = time.monotonic()
        self.sol_seconds, self.launches, self.compiled_functions, self.declared = 0.0, 0, 0, False
        self.arm(self.time_limit(), self.start)

    def end_test(self):
        self.start = None
        self.arm(OUTSIDE_TESTS_SECONDS)

    def restart(self):
        if self.start is None:
            self.arm(OUTSIDE_TESTS_SECONDS)
        else:
            self.start = time.monotonic()
            self.arm(self.time_limit(), self.start)

    def declare(self, sol_seconds, launches, compiled_functions):
        assert self.start is not None, "declare GPU work inside a test"
        self.sol_seconds += sol_seconds
        self.launches += launches
        self.compiled_functions += compiled_functions
        self.declared = True
        self.arm(self.time_limit(), self.start)


watchdog = Watchdog()


def declare_gpu_work(launches=1, *, flops=0, dram_bytes=0, element="bf16", compiled_functions=0, device_id=None):
    """Adds `launches` launches of `flops` tensor-core FLOPs and `dram_bytes` DRAM bytes each to the running test.
    The first declaration replaces the time limit of a test without declared work."""
    sol_seconds = launches * roofline.sol_seconds(flops, dram_bytes, element, device_id) if launches else 0.0
    watchdog.declare(sol_seconds, launches, compiled_functions)


def compile_function(function, **options):
    declare_gpu_work(0, compiled_functions=1)
    return torch.compile(function, **options)


def pytest_addoption(parser):
    parser.addoption("--time-limit-scale", type=float, default=1.0,
                     help="multiply the time limit of every test, for example under compute-sanitizer")


def pytest_configure(config):
    if float(config.getini("faulthandler_timeout") or 0):
        raise pytest.UsageError("leave faulthandler_timeout unset, because tests/conftest.py sets the time limits")
    torch.set_float32_matmul_precision("high")
    watchdog.stderr = os.dup(2)
    watchdog.scale = config.getoption("--time-limit-scale")
    watchdog.arm(OUTSIDE_TESTS_SECONDS)


def pytest_unconfigure(config):
    faulthandler.cancel_dump_traceback_later()
    if watchdog.stderr is not None:
        os.close(watchdog.stderr)
        watchdog.stderr = None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    watchdog.begin_test()
    try:
        return (yield)
    finally:
        watchdog.end_test()


# The faulthandler plugin of pytest cancels the timer when pytest reports an error or enters the debugger.
@pytest.hookimpl(trylast=True)
def pytest_exception_interact():
    watchdog.restart()


def pytest_leave_pdb():
    watchdog.restart()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        result = yield
    except BaseException:
        checks.queued_checks.clear()
        raise
    checks.verify_queued_checks()
    return result


def cuda_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
