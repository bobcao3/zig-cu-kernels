"""Speed-of-light (SoL) times use the peak rates of the device, and power-capped times use the power-capped ceiling
that `set_ceilings` stores (README section "Benchmarks")."""

import math
from collections import namedtuple

from cuda.core import Device

from . import runtime

# The data sheets of the A100, RTX A6000 and RTX 6000 Ada give these dense `mma` FLOPs per clock and
# streaming multiprocessor (SM) with FP32 accumulation. Each GeForce board in the GA102 and AD102 whitepapers,
# and the L40 in the AD102 whitepaper, has half these rates. The H100 SXM data sheet gives 989 BF16 and 1979 FP8
# dense TFLOP/s, the rates of 132 SMs at 1.83 GHz. The B200 data sheet gives 4.5 dense FP8 PFLOP/s, the rate of 148
# SMs at 1.86 GHz. The RTX PRO 6000 Blackwell Max-Q data sheet gives 878 dense FP8 TFLOP/s at 188 SMs and 2.28 GHz.
TENSOR_FLOPS_PER_CLOCK = {
    80: {"bf16": 2048},
    86: {"bf16": 1024},
    89: {"bf16": 1024, "e4m3": 2048},
    90: {"bf16": 4096, "e4m3": 8192},
    100: {"bf16": 8192, "e4m3": 16384},
    120: {"bf16": 1024, "e4m3": 2048},
}

HOST_SECONDS = 5.0
SOL_FACTOR = 20.0
LAUNCH_SECONDS = 50e-6
COMPILE_SECONDS = 60.0

Ceilings = namedtuple(
    "Ceilings",
    "copy_bytes_per_second read_bytes_per_second write_bytes_per_second bf16_flops_per_second "
    "e4m3_flops_per_second",
    defaults=(None, None),
)

measured_ceilings = {}


def device_of(device_id):
    return Device().device_id if device_id is None else device_id


def table_rates(capability):
    if capability in TENSOR_FLOPS_PER_CLOCK:
        return TENSOR_FLOPS_PER_CLOCK[capability]
    same_major_below = [
        rates
        for known, rates in TENSOR_FLOPS_PER_CLOCK.items()
        if known // 10 == capability // 10 and known < capability
    ]
    elements = {element for rates in same_major_below for element in rates}
    return {element: max(rates.get(element, 0) for rates in same_major_below) for element in elements}


def runs_at_half_rate(device_name):
    return "GeForce" in device_name or device_name == "NVIDIA L40"


def tensor_flops_per_clock(element, device_id=None):
    device_id = device_of(device_id)
    per_clock = table_rates(runtime.compute_capability(device_id)).get(element)
    if per_clock is not None and runs_at_half_rate(Device(device_id).name):
        return per_clock // 2
    return per_clock


def peak_flops(element, device_id=None):
    device_id = device_of(device_id)
    per_clock = tensor_flops_per_clock(element, device_id)
    if per_clock is None:
        capability = runtime.compute_capability(device_id)
        raise ValueError(f"no {element} tensor-core rate is known for compute capability {capability}")
    properties = Device(device_id).properties
    return per_clock * properties.multiprocessor_count * properties.clock_rate * 1e3


def peak_bandwidth(device_id=None):
    properties = Device(device_of(device_id)).properties
    return 2 * properties.memory_clock_rate * 1e3 * properties.global_memory_bus_width / 8


def compute_seconds(flops, element="bf16", device_id=None):
    return flops / peak_flops(element, device_id) if flops else 0.0


def memory_seconds(dram_bytes, device_id=None):
    return dram_bytes / peak_bandwidth(device_id)


def sol_seconds(flops=0, dram_bytes=0, element="bf16", device_id=None):
    return max(compute_seconds(flops, element, device_id), memory_seconds(dram_bytes, device_id))


def set_ceilings(device_id, ceilings):
    measured_ceilings[device_id] = ceilings


def ceilings(device_id=None):
    return measured_ceilings.get(device_of(device_id))


def power_capped_seconds(flops=0, read_bytes=0, write_bytes=0, element="bf16", device_id=None):
    measured = ceilings(device_id)
    if measured is None:
        return None
    rate = getattr(measured, f"{element}_flops_per_second") if flops else math.inf
    if rate is None:
        return None
    memory = read_bytes / measured.read_bytes_per_second + write_bytes / measured.write_bytes_per_second
    return max(flops / rate, memory)


def time_limit(sol_seconds_total, launches, compiled_functions=0):
    return (
        HOST_SECONDS
        + SOL_FACTOR * sol_seconds_total
        + LAUNCH_SECONDS * launches
        + COMPILE_SECONDS * compiled_functions
    )
