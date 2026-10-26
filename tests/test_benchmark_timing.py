import faulthandler
import functools
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from conftest import declare_gpu_work
from zig_cu_kernels import roofline, runtime

TIMING_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "timing.py"
FAKE_L2_BYTES = 50 << 20
ROTATION_L2_MULTIPLE = 4
CAPPED_FRACTION_OF_PEAK = 0.8
FAKE_CEILINGS = {
    "copy_bytes_per_second": 1e11,
    "read_bytes_per_second": 2e11,
    "write_bytes_per_second": 3e11,
    "bf16_flops_per_second": 4e13,
    "e4m3_flops_per_second": 5e13,
}
ROTATION_COPIES = 3


@pytest.fixture
def timing(monkeypatch):
    """Loads `benchmarks/timing.py` anew, without a ceilings file. The import arms a watchdog that ends the process 10 s
    after the process started, so the import runs with that call replaced."""
    spec = importlib.util.spec_from_file_location("timing", TIMING_PATH)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(faulthandler, "dump_traceback_later", lambda *arguments, **options: None)
        spec.loader.exec_module(module)
    monkeypatch.delenv(module.CEILINGS_VARIABLE, raising=False)
    monkeypatch.setattr(roofline, "measured_ceilings", {})
    return module


def test_rotation_copies_hold_four_times_the_l2_cache(timing, monkeypatch):
    monkeypatch.setattr(runtime, "l2_bytes", lambda device_id: FAKE_L2_BYTES)
    target = ROTATION_L2_MULTIPLE * FAKE_L2_BYTES
    for distinct_bytes in (1, 3 << 20, (7 << 20) + 1, FAKE_L2_BYTES, target, target + 1, 1 << 40):
        copies = timing.rotation_copies(distinct_bytes)
        assert copies * distinct_bytes >= target, (distinct_bytes, copies)
        assert copies == 1 or (copies - 1) * distinct_bytes < target, (distinct_bytes, copies)


def test_results_faster_than_the_dram_limits_end_the_benchmark_with_status_1(timing):
    capped_rate = CAPPED_FRACTION_OF_PEAK * roofline.peak_bandwidth()
    timing.known_ceilings.update(read_bytes_per_second=capped_rate, write_bytes_per_second=capped_rate)
    traffic = {"read_bytes": 3 << 30, "write_bytes": 1 << 30}
    capped_seconds = sum(traffic.values()) / capped_rate
    for fraction in (0.5, 1.0, 1.04):
        note = timing.roofline_note(capped_seconds / fraction, **traffic)
        assert f"{100 * fraction:5.1f}% of power-capped" in note and "ABOVE" not in note, note
    timing.run(lambda: None)
    assert "ABOVE THE POWER-CAPPED DRAM BOUND (106.0% > 105%)" in timing.roofline_note(capped_seconds / 1.06, **traffic)
    assert "ABOVE THE DRAM PEAK" in timing.roofline_note(capped_seconds / 1.3, **traffic)
    with pytest.raises(SystemExit, match="timing: 2 results"):
        timing.run(lambda: None)


def test_ceilings_missing_from_the_file_are_measured_once_and_stored(timing, monkeypatch, tmp_path):
    path = tmp_path / "ceilings.json"
    monkeypatch.setenv(timing.CEILINGS_VARIABLE, str(path))
    path.write_text(json.dumps({timing.device_key(): {"read_bytes_per_second": 7e11}}))
    measured = []

    def measurement(name):
        measured.append(name)
        return FAKE_CEILINGS[name]

    for name in FAKE_CEILINGS:
        timing.CEILING_MEASUREMENTS[name] = functools.partial(measurement, name)
    monkeypatch.setattr(timing, "unmeasurable_reason", lambda name: None)
    ceilings = timing.register_ceilings("bf16")
    assert timing.register_ceilings("bf16") == ceilings
    assert measured == ["write_bytes_per_second", "bf16_flops_per_second"]
    stored = {"read_bytes_per_second": 7e11, **{name: FAKE_CEILINGS[name] for name in measured}}
    assert {name: getattr(ceilings, name) for name in stored} == stored
    assert json.loads(path.read_text()) == {timing.device_key(): stored}


def test_a_graph_holds_whole_rotation_cycles_for_about_2_ms(timing, monkeypatch):
    """The fastest replay of a graph of one cycle sizes a graph of whole cycles of at most 1000 calls."""
    declare_gpu_work(2 * ROTATION_COPIES + 1000 + 10 * ROTATION_COPIES)
    counts = [torch.zeros((), device="cuda", dtype=torch.int64) for _ in range(ROTATION_COPIES)]
    rotation = timing.Rotation(functools.partial(torch.Tensor.add_, count, 1) for count in counts)
    launches = []

    def one_replay(function, warmup, duration, sol_seconds, calls):
        function()
        launches.append(calls)
        return 1.0

    monkeypatch.setattr(timing, "sustained_seconds", one_replay)
    for cycle_seconds, cycles in ((2e-4, 10), (1e-7, 1000 // ROTATION_COPIES)):
        monkeypatch.setattr(timing, "fastest_call_seconds", lambda function, seconds=cycle_seconds: seconds)
        before = [int(count) for count in counts]
        assert timing.graph_seconds(rotation) == 1.0 / (cycles * ROTATION_COPIES)
        assert launches[-1] == cycles * ROTATION_COPIES
        assert [int(count) - start for count, start in zip(counts, before)] == [1 + cycles] * ROTATION_COPIES


def test_a_measured_ceiling_keeps_the_highest_rate_and_the_latest_one(timing, monkeypatch, tmp_path):
    monkeypatch.setenv(timing.CEILINGS_VARIABLE, str(tmp_path / "ceilings.json"))
    name = timing.tensor_ceiling_name("bf16")
    for rate in (3e14, 2.5e14):
        timing.store_measured_ceiling(name, rate)
    assert timing.stored_ceiling(name) == 3e14
    assert timing.stored_ceiling(timing.latest_ceiling_name(name)) == 2.5e14
    timing.known_ceilings.update(read_bytes_per_second=1e12, write_bytes_per_second=1e12, bf16_flops_per_second=3e14)
    note = timing.roofline_note(1.0, flops=2.5e14)
    assert " 83.3% of power-capped" in note and "100.0% of latest power-capped" in note, note


@pytest.mark.parametrize("remaining, extended", ((10.0, True), (2.5, False)))
def test_the_power_cycle_extension_leaves_time_for_later_measurements(timing, monkeypatch, remaining, extended):
    durations = []

    def fake_windows(function, calls, seconds):
        durations.append(seconds)
        return [1.0, 1.0, 1.1, 1.1]

    class CappedTelemetry:
        capped = True

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            pass

        def means(self):
            return None

    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *arguments, **options: None)
    monkeypatch.setattr(timing, "windows", fake_windows)
    monkeypatch.setattr(timing, "Telemetry", CappedTelemetry)
    monkeypatch.setattr(timing, "fastest_call_seconds", lambda function: 1e-3)
    monkeypatch.setattr(timing, "fill", lambda function, seconds: None)
    monkeypatch.setattr(timing.watchdog, "remaining", lambda: remaining)
    timing.measure(lambda: None)
    assert (len(durations) == 3) == extended, durations


@pytest.fixture
def benchmark_fp8_mm(timing, monkeypatch):
    """Loads `benchmarks/benchmark_fp8_mm.py` on the `timing` module of the fixture of that name."""
    monkeypatch.setitem(sys.modules, "timing", timing)
    spec = importlib.util.spec_from_file_location("benchmark_fp8_mm", TIMING_PATH.with_name("benchmark_fp8_mm.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(timing, "CYCLE_RESERVE_SECONDS", timing.CYCLE_RESERVE_SECONDS)
    return module


def test_a_shape_name_stands_for_each_of_its_row_counts(benchmark_fp8_mm):
    shapes = benchmark_fp8_mm.parse_shapes(["gemma_qkv", "qwen_down:16384", "8192^3", "64x128x256"])
    assert [(shape.name, shape.m, shape.n, shape.k) for shape in shapes] == [
        *((f"gemma_qkv:{m}", m, 16384, 5376) for m in (1, 64, 256, 2048, 16384)),
        ("qwen_down:16384", 16384, 4096, 12288),
        ("8192^3", 8192, 8192, 8192),
        ("64x128x256", 64, 128, 256),
    ]
    with pytest.raises(SystemExit, match="no named shape 'gemma'"):
        benchmark_fp8_mm.parse_shapes(["gemma:64"])


def test_the_sol_time_is_the_slower_of_the_peak_rate_and_the_copy_rate(benchmark_fp8_mm):
    b300 = benchmark_fp8_mm.device_entry("NVIDIA B300 SXM6 AC")
    flops = 2 * 8192**3
    assert benchmark_fp8_mm.sol_seconds(b300, "fp8_rowwise", flops, 0) == flops / 3490e12
    assert benchmark_fp8_mm.sol_seconds(b300, "mxfp8", flops, 0) == flops / 3375e12
    assert benchmark_fp8_mm.sol_seconds(b300, "mxfp8", 2 * 5376 * 21504, 5376 * 21504) == 5376 * 21504 / 6614e9
    ada = benchmark_fp8_mm.device_entry("NVIDIA RTX 6000 Ada Generation")
    assert (ada.f8f6f4_tflops, ada.copy_gbps) == (640, 666)
    assert benchmark_fp8_mm.device_entry("NVIDIA RTX PRO 6000 Blackwell Server Edition").f8f6f4_tflops == 931
    assert benchmark_fp8_mm.sol_seconds(benchmark_fp8_mm.device_entry("NVIDIA A100"), "mxfp8", flops, 0) is None


def test_the_budget_refuses_a_measurement_past_it_and_keeps_measurements_short(benchmark_fp8_mm, timing,
                                                                               monkeypatch):
    budget = benchmark_fp8_mm.Budget(9.0)
    monkeypatch.setattr(timing.watchdog, "remaining", lambda: 3.0)
    assert budget.remaining() == 1.0
    assert budget.allows(0.9) and not budget.allows(1.1)
    assert timing.CYCLE_RESERVE_SECONDS == math.inf
    budget.skip("8192^3 mm_abt mxfp8")
    assert budget.skipped == ["8192^3 mm_abt mxfp8"]
