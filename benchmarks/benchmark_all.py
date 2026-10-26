"""Runs the default cases of every benchmark with the PyTorch baselines, a few cases per process. README, section
Benchmarks, describes the run and `--lock`. A measurement at the power limit lasts about 1.3 s, so a process holds at
most 4 such measurements. The BF16 GEMMs, the FP8 GEMMs with at least `POWER_BOUND_ROWS` rows and the axpby cases that
fit in the L2 cache run at the power limit."""

import argparse
import contextlib
import fcntl
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROCESS_SECONDS = 10.0
KILL_GRACE_SECONDS = 1.0
AXPBY_COUNTS = (("1048576",), ("4194304",), ("67108864", "268435456"))
AXPBY_IN_PLACE_COUNTS = (("1048576", "4194304"), ("67108864", "268435456"))
RMS_NORM_SHAPES = ("16384x4096", "8192x8192", "4096x16384", "65536x1024")
RMS_NORM_VARIANTS = (
    ("plain", "f32", "residual", "fp8"),
    ("fp8_blockwise", "residual_fp8_blockwise", "adaptive", "dual"),
    ("fp8_rowwise", "fp8_blockwise_e8m0", "adaptive_fp8_blockwise"),
)
GEMM_SHAPES = (
    "4096x4096x4096", "8192x8192x8192", "16384x14336x4096", "8192x14336x8192", "16384x4096x14336",
    "129x4096x4096", "192x4096x4096", "256x4096x4096", "512x14336x4096", "1024x4096x4096", "2048x8192x4096",
    "1024x14336x4096", "2048x4096x14336", "1x4096x4096", "16x8192x8192", "32x4096x4096", "64x4096x4096",
    "128x8192x8192", "128x28672x4096", "8x1024x1024", "64x1024x1024", "256x1024x1024", "1x2048x2048",
    "129x2048x2048", "256x2048x2048",
)
FP8_GEMM_SHAPES = (
    "8192x8192x8192", "16384x8192x8192", "8192x14336x4096", "4096x4096x4096", "2048x4096x4096", "512x4096x4096",
    "128x4096x4096", "64x8192x8192", "64x6144x4096", "32x4096x4096", "1x4096x4096", "32x14336x4096",
)
FP8_GATED_SHAPES = ("8192x14336x4096", "4096x4096x4096", "32x14336x4096")
FP8_QUANTIZE_SHAPES = ("16384x4096", "8192x8192", "16384x16384")
POWER_BOUND_ROWS = 128
GEMM_EPILOGUE_GROUPS = (("store", "bias_silu"), ("residual", "f32"), ("gated_silu",))


def power_bound(shape):
    return int(shape.split("x")[0]) >= POWER_BOUND_ROWS


def invocations():
    yield "benchmark_peak.py", "--seconds", "0.15"
    for dtype in ("bf16", "f32"):
        for counts in AXPBY_COUNTS:
            yield "benchmark_axpby.py", "--baselines", "--dtypes", dtype, "--counts", *counts
        for counts in AXPBY_IN_PLACE_COUNTS:
            yield "benchmark_axpby.py", "--baselines", "--dtypes", dtype, "--cases", "in_place", "--counts", *counts
    for shape in RMS_NORM_SHAPES:
        for variants in RMS_NORM_VARIANTS:
            yield "benchmark_rms_norm.py", "--baselines", "--shapes", shape, "--variants", *variants
    for shape in GEMM_SHAPES:
        for epilogues in GEMM_EPILOGUE_GROUPS:
            yield "benchmark_gemm.py", "--baselines", "--shapes", shape, "--epilogues", *epilogues
    fp8 = ("benchmark_fp8_gemm.py", "--baselines", "--e8m0")
    for shape in FP8_GEMM_SHAPES:
        if power_bound(shape):
            yield "benchmark_fp8_gemm.py", "--e8m0", "--shapes", shape, "--gated-shapes", "--quantize-shapes"
            yield "benchmark_fp8_gemm.py", "--baselines-only", "--shapes", shape, "--gated-shapes", "--quantize-shapes"
        else:
            yield *fp8, "--shapes", shape, "--gated-shapes", "--quantize-shapes"
    for shape in FP8_GATED_SHAPES:
        yield *fp8, "--shapes", "--gated-shapes", shape, "--quantize-shapes"
    for shape in FP8_QUANTIZE_SHAPES:
        yield "benchmark_fp8_gemm.py", "--e8m0", "--shapes", "--gated-shapes", "--quantize-shapes", shape
        yield "benchmark_fp8_gemm.py", "--baselines-only", "--shapes", "--gated-shapes", "--quantize-shapes", shape


@contextlib.contextmanager
def locked(path):
    if path is None:
        yield
        return
    with open(path, "a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(file, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=[], help="run only the scripts whose names contain one of these")
    parser.add_argument("--list", action="store_true", help="print the commands without running them")
    parser.add_argument("--lock", help="hold an exclusive flock of this file while each process runs")
    arguments = parser.parse_args()
    directory = Path(__file__).resolve().parent
    commands = [command for command in invocations()
                if not arguments.only or any(word in command[0] for word in arguments.only)]
    if arguments.list:
        print("\n".join(" ".join(command) for command in commands))
        return
    environment = dict(os.environ)
    with tempfile.TemporaryDirectory() as scratch:
        environment.setdefault("ZIG_CU_KERNELS_CEILINGS", str(Path(scratch) / "ceilings.json"))
        failures, durations, run_start = [], [], time.monotonic()
        for command in commands:
            with locked(arguments.lock):
                print(f"== {' '.join(command)}", flush=True)
                start = time.monotonic()
                try:
                    status = subprocess.run([sys.executable, str(directory / command[0]), *command[1:]],
                                            cwd=directory, env=environment,
                                            timeout=PROCESS_SECONDS + KILL_GRACE_SECONDS).returncode
                except subprocess.TimeoutExpired:
                    status = "killed after timeout"
                durations.append((time.monotonic() - start, " ".join(command)))
            print(f"== {durations[-1][0]:.1f} s, exit status {status}", flush=True)
            if status != 0:
                failures.append(f"{' '.join(command)}: {status}")
    slowest = max(durations, default=(0.0, "none"))
    print(f"== {len(commands)} processes, {len(failures)} failed, {sum(seconds for seconds, _ in durations):.0f} s in "
          f"the processes, {time.monotonic() - run_start:.0f} s in all, slowest {slowest[0]:.1f} s: {slowest[1]}",
          flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} of {len(commands)} benchmark processes failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
