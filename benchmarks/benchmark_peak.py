"""Measures the power-capped tensor-core rates with the `mma_rate` kernels into `ZIG_CU_KERNELS_CEILINGS`."""

import argparse

import peak_sm100
import peak_sm120
import peak_sm90
import timing
from zig_cu_kernels import roofline, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", nargs="+", choices=timing.MMA_RATE_FLOPS_PER_INSTRUCTION,
                        default=list(timing.MMA_RATE_FLOPS_PER_INSTRUCTION))
    parser.add_argument("--work-groups-per-sm", type=int, default=timing.MMA_RATE_WORK_GROUPS_PER_SM)
    parser.add_argument("--iterations", type=int, default=timing.MMA_RATE_ITERATIONS)
    parser.add_argument("--seconds", type=float, default=timing.MEASURE_SECONDS, help="the time of one measurement")
    arguments = parser.parse_args()
    sms = runtime.num_sm(runtime.current_device().device_id)
    print(f"mma_rate: {sms} SMs x {arguments.work_groups_per_sm} work groups, {timing.MMA_RATE_CHAINS} chains of "
          f"{arguments.iterations} mma per warp; SoL = SMs x CU_DEVICE_ATTRIBUTE_CLOCK_RATE x the FLOPs per clock of "
          "the table in roofline.py", flush=True)
    for element in arguments.elements:
        reason = timing.mma_rate_unavailable_reason(element)
        if reason is not None:
            print(f"{element:5} skipped: {reason}", flush=True)
            continue
        rate, measurement = timing.mma_rate(element, arguments.work_groups_per_sm, arguments.iterations,
                                                    arguments.seconds)
        timing.store_measured_ceiling(timing.tensor_ceiling_name(element), rate)
        peak = roofline.peak_flops(element)
        per_clock = "" if measurement.clock_mhz is None else \
            f"  {rate / (sms * measurement.clock_mhz * 1e6):5.0f} FLOP/clk/SM at the sampled clock"
        print(f"{element:5} {rate * 1e-12:6.1f} TF/s  {100 * rate / peak:5.1f}% of SoL ({peak * 1e-12:.0f} TF/s)"
              f"{per_clock}  {timing.clock_note()}", flush=True)
    peak_sm100.report()
    peak_sm120.report()
    peak_sm90.report()


if __name__ == "__main__":
    main()
