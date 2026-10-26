"""Measures the device time per call of benchmark functions. `python timing.py` measures the power-capped ceiling.

After an idle gap, a GPU at its power limit runs at its boost clock until its power controller lowers the clock, so
`measure` warms the GPU up first. README, section Benchmarks, describes the limits and the ceilings file."""

import argparse
import contextlib
import faulthandler
import fcntl
import functools
import importlib.util
import json
import math
import os
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import torch
from cuda.bindings import nvml

if importlib.util.find_spec("zig_cu_kernels") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zig_cu_kernels import roofline, runtime

PROCESS_SECONDS = 10.0
TEARDOWN_SECONDS = 1.0
PREHEAT_SECONDS = 1.2
REHEAT_SECONDS = 0.15
MEASURE_SECONDS = 0.15
IDLE_SECONDS = 1.0
CYCLE_SECONDS = 1.0
CYCLE_RESERVE_SECONDS = 1.0
CLOCK_STEP_SPREAD = 0.007
SOFTWARE_POWER_CAP = 0x4
FILL_SECONDS = 0.05
WINDOW_SECONDS = 0.01
SIZE_SAMPLES = 5
SIZE_SECONDS = 0.01
CEILING_BYTES = 1 << 30
TELEMETRY_PERIOD_SECONDS = 0.02
GRAPH_SECONDS = 2e-3
MAX_GRAPH_CALLS = 1000
ROTATION_L2_MULTIPLE = 4
CAPPED_DRAM_MARGIN = 1.05
HOST_CALLS = 2000
HOST_REPEATS = 5
MMA_RATE_FLOPS_PER_INSTRUCTION = {"bf16": 2 * 16 * 8 * 16, "e4m3": 2 * 16 * 8 * 32}
MMA_RATE_CHAINS = 8
MMA_RATE_WORK_GROUPS_PER_SM = 2
MMA_RATE_ITERATIONS = 4096
MMA_RATE_SEED = 1
CEILINGS_VARIABLE = "ZIG_CU_KERNELS_CEILINGS"
DRAM_CEILINGS = ("read_bytes_per_second", "write_bytes_per_second")
Measurement = namedtuple("Measurement", "seconds drift windows calls_per_window clock_mhz power_w temperature_c")
last_telemetry = None
last_measurement_end = None
known_ceilings = {}
results_above_dram_limits = []


def process_age():
    try:
        with open("/proc/self/stat") as file:
            start_ticks = int(file.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as file:
            uptime = float(file.read().split()[0])
    except (OSError, IndexError, ValueError):
        return 0.0
    return max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))


class Watchdog:
    def __init__(self, seconds):
        self.deadline = time.monotonic() - process_age() + seconds
        self.release()

    def remaining(self):
        return self.deadline - time.monotonic()

    def limit(self, seconds):
        deadline = min(self.deadline, time.monotonic() + seconds)
        faulthandler.dump_traceback_later(max(deadline - time.monotonic(), 1e-3), exit=True)

    def release(self):
        self.limit(math.inf)

    def require(self, seconds, what):
        available = self.remaining() - TEARDOWN_SECONDS
        if available < seconds:
            raise SystemExit(f"timing: {what} needs {seconds:.2f} s, but the process has {available:.2f} s left of "
                             f"its {PROCESS_SECONDS:.0f} s after {TEARDOWN_SECONDS:.0f} s for its exit; run fewer "
                             "cases per process")


watchdog = Watchdog(PROCESS_SECONDS)


@functools.cache
def nvml_device():
    nvml.init_v2()
    uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
    return nvml.device_get_handle_by_uuid(f"GPU-{uuid}")


class Telemetry:
    """Samples the SM clock in MHz, the board power in W and the GPU temperature in degrees Celsius, and records
    whether the power controller limited the clock."""

    def __enter__(self):
        self.handle = nvml_device()
        self.samples = []
        self.capped = False
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.done.wait(TELEMETRY_PERIOD_SECONDS):
            self.samples.append((
                nvml.device_get_clock_info(self.handle, nvml.ClockType.CLOCK_SM),
                nvml.device_get_power_usage(self.handle) * 1e-3,
                nvml.device_get_temperature_v(self.handle, nvml.TemperatureSensors.TEMPERATURE_GPU),
            ))
            reasons = nvml.device_get_current_clocks_event_reasons(self.handle)
            self.capped = self.capped or bool(reasons & SOFTWARE_POWER_CAP)

    def __exit__(self, *exception):
        self.done.set()
        self.thread.join()

    def means(self):
        if not self.samples:
            return None
        return tuple(sum(values) / len(values) for values in zip(*self.samples))


def single_call_seconds(function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    end.synchronize()
    return max(start.elapsed_time(end) * 1e-3, 1e-7)


def fastest_call_seconds(function):
    function()
    times = [single_call_seconds(function)]
    while len(times) < SIZE_SAMPLES and sum(times) < SIZE_SECONDS:
        times.append(single_call_seconds(function))
    return min(times)


def windows(function, calls, seconds):
    """The host waits only for the window two places behind, so the GPU stays busy between windows."""
    pairs = []
    deadline = time.perf_counter() + seconds
    while not pairs or time.perf_counter() < deadline:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            function()
        end.record()
        pairs.append((start, end))
        if len(pairs) > 2:
            pairs[-3][1].synchronize()
    pairs[-1][1].synchronize()
    return [start.elapsed_time(end) * 1e-3 for start, end in pairs]


def drift_of(times):
    half = len(times) // 2
    return sum(times[half:]) / (len(times) - half) / (sum(times[:half]) / half) - 1 if half else 0.0


def measure(function, sol_seconds=None, *, launches=1, reheat=REHEAT_SECONDS, duration=MEASURE_SECONDS):
    """Returns a `Measurement` whose `seconds` is the mean device time of one call of `function` in `duration` seconds
    of back-to-back calls. When the power controller limits the clock and the second-slowest and second-fastest
    windows differ by more than `CLOCK_STEP_SPREAD`, the measurement lasts `CYCLE_SECONDS` if the process keeps
    `CYCLE_RESERVE_SECONDS` for its later measurements. With `sol_seconds`, the SoL time of one call of `launches`
    launches, the watchdog ends the process when the planned calls take longer than their `roofline.time_limit`."""
    global last_telemetry, last_measurement_end
    idle = last_measurement_end is None or time.monotonic() - last_measurement_end > IDLE_SECONDS
    warm = PREHEAT_SECONDS if idle else reheat
    watchdog.require(warm + duration, "the next measurement")
    if sol_seconds is not None:
        planned = 1 + SIZE_SAMPLES
        watchdog.limit(roofline.time_limit(planned * sol_seconds, planned * launches))
    single = fastest_call_seconds(function)
    calls = max(1, round(WINDOW_SECONDS / single))
    if sol_seconds is not None:
        planned = calls * (2 + math.ceil((warm + max(duration, CYCLE_SECONDS)) / (calls * single)))
        watchdog.limit(roofline.time_limit(planned * sol_seconds, planned * launches))
    warm_times = windows(function, calls, warm)
    with Telemetry() as telemetry:
        times = windows(function, calls, duration)
        cycle_rest = CYCLE_SECONDS - duration
        spare = watchdog.remaining() - TEARDOWN_SECONDS - FILL_SECONDS - CYCLE_RESERVE_SECONDS
        ordered = sorted(warm_times + times)
        if telemetry.capped and ordered[-2] / ordered[1] - 1 > CLOCK_STEP_SPREAD and 0 < cycle_rest <= spare:
            times += windows(function, calls, cycle_rest)
    watchdog.release()
    seconds = sum(times) / (len(times) * calls)
    fill(function, seconds)
    last_measurement_end = time.monotonic()
    last_telemetry = telemetry.means()
    clock_mhz, power_w, temperature_c = last_telemetry or (None, None, None)
    return Measurement(seconds, drift_of(times), len(times), calls, clock_mhz, power_w, temperature_c)


def fill(function, seconds_per_call):
    deadline = time.perf_counter() + FILL_SECONDS
    for _ in range(math.ceil(FILL_SECONDS / seconds_per_call)):
        if time.perf_counter() > deadline:
            break
        function()


def sustained_seconds(function, warmup=None, duration=None, sol_seconds=None, launches=1):
    return measure(function, sol_seconds, launches=launches, reheat=REHEAT_SECONDS if warmup is None else warmup,
                   duration=MEASURE_SECONDS if duration is None else duration).seconds


def rotation_copies(distinct_bytes):
    """Returns the number of copies of `distinct_bytes` that fill the L2 cache `ROTATION_L2_MULTIPLE` times."""
    l2_bytes = runtime.l2_bytes(runtime.current_device().device_id)
    return max(1, -(-ROTATION_L2_MULTIPLE * l2_bytes // max(1, distinct_bytes)))


class Rotation:
    """Calls the functions of `calls` in turn. With one copy of the operands per function and `rotation_copies`
    functions, one cycle of `period` calls moves `ROTATION_L2_MULTIPLE` times the L2 cache size or more."""

    def __init__(self, calls):
        self.calls = tuple(calls)
        self.period = len(self.calls)
        self.position = -1

    def __call__(self):
        self.position = (self.position + 1) % self.period
        return self.calls[self.position]()


def captured(function, calls, stream):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls):
            function()
    return graph


def graph_seconds(function, warmup=None, duration=None, sol_seconds=None, launches=1):
    """Returns the device time per call of `function` from replays of a CUDA graph. `GRAPH_SECONDS` and
    `MAX_GRAPH_CALLS` set the number of whole `Rotation` cycles in the graph."""
    period = getattr(function, "period", 1)
    if sol_seconds is not None:
        planned = period * (2 + SIZE_SAMPLES)
        watchdog.limit(roofline.time_limit(planned * sol_seconds, planned * launches))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(period):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    cycle = captured(function, period, stream)
    cycle_seconds = fastest_call_seconds(cycle.replay)
    del cycle
    cycles = max(1, min(MAX_GRAPH_CALLS // period, round(GRAPH_SECONDS / cycle_seconds)))
    graph = captured(function, cycles * period, stream)
    replay_sol_seconds = None if sol_seconds is None else sol_seconds * cycles * period
    seconds = sustained_seconds(graph.replay, warmup, duration, replay_sol_seconds, launches * cycles * period)
    return seconds / (cycles * period)


def host_seconds(function, calls=HOST_CALLS):
    for _ in range(calls // 10):
        function()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(HOST_REPEATS):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        best = min(best, (time.perf_counter() - start) / calls)
        torch.cuda.synchronize()
    return best


def clock_note():
    if last_telemetry is None:
        return ""
    return f"@ {last_telemetry[0]:4.0f} MHz {last_telemetry[1]:3.0f} W {last_telemetry[2]:2.0f} C"


def copy_rate():
    source = torch.empty(CEILING_BYTES, dtype=torch.uint8, device="cuda")
    target = torch.empty_like(source)
    return 2 * CEILING_BYTES / sustained_seconds(lambda: target.copy_(source))


def read_rate():
    values = torch.empty(CEILING_BYTES // 4, dtype=torch.float32, device="cuda")
    total = torch.empty((), device="cuda")
    return CEILING_BYTES / sustained_seconds(lambda: torch.sum(values, dim=0, out=total))


def write_rate():
    values = torch.empty(CEILING_BYTES // 4, dtype=torch.float32, device="cuda")
    return CEILING_BYTES / sustained_seconds(lambda: values.fill_(1.0))


def mma_rate_unavailable_reason(element):
    if roofline.tensor_flops_per_clock(element) is None:
        return f"the device has no {element} tensor-core rate in the table of roofline.py"
    return runtime.unavailable_reason(f"mma_rate_{element}")


def mma_rate(element, work_groups_per_sm=MMA_RATE_WORK_GROUPS_PER_SM, iterations=MMA_RATE_ITERATIONS,
                     duration=MEASURE_SECONDS):
    """Returns the FLOP rate and the `Measurement` of independent `mma` instructions on random register operands."""
    kernel, info = runtime.load(f"mma_rate_{element}")
    work_groups = runtime.num_sm(runtime.current_device().device_id) * work_groups_per_sm
    out = torch.empty(info.threads, device="cuda")
    values = (out.data_ptr(), iterations, MMA_RATE_SEED)
    instructions = work_groups * info.threads // 32 * MMA_RATE_CHAINS * iterations
    flops = instructions * MMA_RATE_FLOPS_PER_INSTRUCTION[element]

    def run():
        runtime.run(kernel, (work_groups, 1, 1), info.threads, 0, "puu", values,
                    torch.cuda.current_stream().cuda_stream)

    measurement = measure(run, roofline.compute_seconds(flops, element), duration=duration)
    return flops / measurement.seconds, measurement


def tensor_ceiling_name(element):
    return f"{element}_flops_per_second"


def main_loop_ceiling_name(element):
    return f"{element}_main_loop_flops_per_second"


def main_loop_source_name(element):
    return f"{element}_main_loop_source"


CEILING_MEASUREMENTS = {
    "copy_bytes_per_second": copy_rate,
    "read_bytes_per_second": read_rate,
    "write_bytes_per_second": write_rate,
    tensor_ceiling_name("bf16"): lambda: mma_rate("bf16")[0],
    tensor_ceiling_name("e4m3"): lambda: mma_rate("e4m3")[0],
}
CEILING_ELEMENTS = {tensor_ceiling_name(element): element for element in MMA_RATE_FLOPS_PER_INSTRUCTION}


def unmeasurable_reason(name):
    element = CEILING_ELEMENTS.get(name)
    return None if element is None else mma_rate_unavailable_reason(element)


@functools.cache
def power_limit_w():
    return nvml.device_get_enforced_power_limit(nvml_device()) / 1000


@functools.cache
def device_key():
    uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
    return f"GPU-{uuid} {power_limit_w():.0f} W driver {nvml.system_get_driver_version()}"


@contextlib.contextmanager
def ceilings_file():
    """Yields the stored ceilings of all devices under a file lock and stores them again, or yields None."""
    path = os.environ.get(CEILINGS_VARIABLE)
    if not path:
        yield None
        return
    with open(path, "a+") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        text = file.read()
        stored = json.loads(text) if text.strip() else {}
        before = json.dumps(stored, sort_keys=True)
        yield stored
        if json.dumps(stored, sort_keys=True) != before:
            file.seek(0)
            file.truncate()
            json.dump(stored, file, indent=1, sort_keys=True)


def stored_ceiling(name):
    with ceilings_file() as stored:
        return None if stored is None else stored.get(device_key(), {}).get(name)


def store_ceiling(name, value):
    with ceilings_file() as stored:
        if stored is not None:
            stored.setdefault(device_key(), {})[name] = value


def latest_ceiling_name(name):
    return f"{name}_latest"


def store_measured_ceiling(name, value):
    """Stores `value` as the latest rate of the ceiling `name`, and as `name` when the file holds no higher rate."""
    with ceilings_file() as stored:
        if stored is not None:
            device = stored.setdefault(device_key(), {})
            device[latest_ceiling_name(name)] = value
            device[name] = max(value, device.get(name, value))


def ceiling(name):
    """Returns the ceiling `name` from the ceilings file, or measures and stores it, or returns None."""
    if name not in known_ceilings:
        value = stored_ceiling(name)
        if value is None and name in CEILING_MEASUREMENTS and unmeasurable_reason(name) is None:
            value = CEILING_MEASUREMENTS[name]()
            store_ceiling(name, value)
        known_ceilings[name] = value
    return known_ceilings[name]


@functools.cache
def main_loop_ceiling(element):
    """Returns the stored main-loop rate of the tiled GEMMs of `element` and the text that names its source."""
    return stored_ceiling(main_loop_ceiling_name(element)), stored_ceiling(main_loop_source_name(element))


def register_ceilings(*elements):
    for name in DRAM_CEILINGS + tuple(tensor_ceiling_name(element) for element in elements):
        ceiling(name)
    measured = roofline.Ceilings(**{name: known_ceilings.get(name) for name in roofline.Ceilings._fields})
    roofline.set_ceilings(runtime.current_device().device_id, measured)
    return measured


def print_limits(element=None, main_loop=False):
    """Prints the SoL rates, the power-capped ceilings and, with `main_loop`, the main-loop ceiling of `element`."""
    measured = register_ceilings(element) if element else register_ceilings()
    sol = [f"DRAM {roofline.peak_bandwidth() * 1e-9:.0f} GB/s"]
    capped = [f"DRAM read {measured.read_bytes_per_second * 1e-9:.0f} GB/s",
              f"write {measured.write_bytes_per_second * 1e-9:.0f} GB/s"]
    if element is not None and roofline.tensor_flops_per_clock(element) is not None:
        rate = getattr(measured, tensor_ceiling_name(element))
        sol.append(f"{element.upper()} tensor cores {roofline.peak_flops(element) * 1e-12:.0f} TF/s")
        capped.append(f"{element.upper()} mma on register operands "
                      f"{'unmeasured' if rate is None else f'{rate * 1e-12:.0f} TF/s'}")
        latest = latest_tensor_ceiling(element)
        if latest is not None:
            capped.append(f"latest measurement {latest * 1e-12:.0f} TF/s")
    print(f"SoL (data sheet): {', '.join(sol)}", flush=True)
    print(f"power-capped ceiling at {power_limit_w():.0f} W: {', '.join(capped)}", flush=True)
    if main_loop:
        rate, source = main_loop_ceiling(element)
        text = f"no {element.upper()} rate in {CEILINGS_VARIABLE}" if rate is None else \
            f"{element.upper()} {rate * 1e-12:.1f} TF/s, {source}"
        print(f"main-loop ceiling of the tiled GEMMs: {text}", flush=True)


@functools.cache
def latest_tensor_ceiling(element):
    return stored_ceiling(latest_ceiling_name(tensor_ceiling_name(element)))


def roofline_note(seconds, flops=0, read_bytes=0, write_bytes=0, element="bf16", main_loop=False):
    """Returns the percentage of each limit of README, section Benchmarks, that a call of `seconds` reaches.
    `roofline_note` also flags a result above the DRAM limits of that section, and `run` then exits with status 1."""
    measured = register_ceilings(element) if flops else register_ceilings()
    memory = read_bytes / measured.read_bytes_per_second + write_bytes / measured.write_bytes_per_second
    notes = [f"{100 * roofline.sol_seconds(flops, read_bytes + write_bytes, element) / seconds:5.1f}% of SoL"]
    capped = roofline.power_capped_seconds(flops, read_bytes, write_bytes, element)
    if capped is not None:
        notes.append(f"{100 * capped / seconds:5.1f}% of power-capped")
    loop_rate = main_loop_ceiling(element)[0] if main_loop and flops else None
    if loop_rate is not None:
        notes.append(f"{100 * max(flops / loop_rate, memory) / seconds:5.1f}% of main loop")
    latest = latest_tensor_ceiling(element) if flops else None
    if latest is not None:
        notes.append(f"{100 * max(flops / latest, memory) / seconds:5.1f}% of latest power-capped")
    rate, peak = (read_bytes + write_bytes) / seconds, roofline.peak_bandwidth()
    if rate > peak:
        results_above_dram_limits.append(rate)
        notes.append(f"ABOVE THE DRAM PEAK ({rate * 1e-9:.0f} > {peak * 1e-9:.0f} GB/s)")
    elif memory > CAPPED_DRAM_MARGIN * seconds:
        results_above_dram_limits.append(rate)
        notes.append(f"ABOVE THE POWER-CAPPED DRAM BOUND ({100 * memory / seconds:.1f}% > "
                     f"{100 * CAPPED_DRAM_MARGIN:.0f}%)")
    return ", ".join(notes)


def run(main):
    """Runs `main` and exits with status 1 when `roofline_note` flagged a result."""
    main()
    if results_above_dram_limits:
        raise SystemExit(f"timing: {len(results_above_dram_limits)} results moved their compulsory DRAM bytes faster "
                         f"than the DRAM peak bandwidth or more than {100 * CAPPED_DRAM_MARGIN:.0f}% as fast as the "
                         "power-capped DRAM rates")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--main-loop", nargs=3, metavar=("ELEMENT", "FLOPS_PER_SECOND", "SOURCE"),
                        help="store a main-loop rate of the tiled GEMMs of ELEMENT, measured elsewhere, with its "
                        "source, and measure nothing")
    arguments = parser.parse_args()
    print(device_key(), flush=True)
    if arguments.main_loop:
        if not os.environ.get(CEILINGS_VARIABLE):
            raise SystemExit(f"timing: --main-loop needs {CEILINGS_VARIABLE}, the JSON file of the ceilings")
        element, rate, source = arguments.main_loop
        store_ceiling(main_loop_ceiling_name(element), float(rate))
        store_ceiling(main_loop_source_name(element), source)
        return
    for name, measurement in CEILING_MEASUREMENTS.items():
        reason = unmeasurable_reason(name)
        if reason is not None:
            print(f"{name:28} skipped: {reason}", flush=True)
            continue
        value = measurement()
        store_measured_ceiling(name, value)
        print(f"{name:28} {value:.4g} {clock_note()}", flush=True)
    rates = [f"{element.upper()} {roofline.peak_flops(element) * 1e-12:.0f} TF/s"
             for element in MMA_RATE_FLOPS_PER_INSTRUCTION if roofline.tensor_flops_per_clock(element) is not None]
    print(f"SoL rates: {roofline.peak_bandwidth() * 1e-9:.0f} GB/s", *rates, sep=", ", flush=True)


if __name__ == "__main__":
    main()
