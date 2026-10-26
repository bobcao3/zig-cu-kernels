"""Times the FP8 pack operators against the speed of light (SoL) and the `torch._scaled_mm` references.

SoL is max(F / P, D / W), where F = 2·M·N·K is the FLOP count, D is the byte count of the packs and outputs, P is the
measured dense FP8 rate and W is the measured copy rate of the device (README, section "FP8 operators"). A reference
that differs from the FP32 product of the dequantized packs by more than `AGREEMENT` (relative Frobenius norm) gets
the mark "disagrees". The run stops before the time budget and prints the cases that it skipped."""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

import dse
import timing
import zig_cu_kernels.torch  # noqa: F401
from zig_cu_kernels import fp8_mm, fp8_pack, packs, runtime

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
RECIPES = fp8_mm.RECIPES
OPS = ("mm_abt", "mm_glu")
GROUP = packs.GROUP
BLOCKWISE_K_MULTIPLE = 512
GRAPH_ROWS = 512
AGREEMENT = 2.0**-8
CHECK_ROWS = 128
SEED = 0
BUDGET_SECONDS = 9.0
SETUP_SECONDS = 0.5
MEASUREMENT_OVERHEAD_SECONDS = 0.1


@dataclass(frozen=True)
class Device:
    f8f6f4_tflops: float
    block_scaled_tflops: float
    copy_gbps: float


DEVICES = {
    "RTX 6000 Ada": Device(640, 640, 666),
    "RTX PRO 6000": Device(931, 931, 1466),
    "H100": Device(1667, 1667, 3025),
    "B300": Device(3490, 3375, 6614),
}
BLOCK_SCALED = ("mxfp8", "fp8_b128_e8m0", "torch mxfp8")
GEMMA_M = (1, 64, 256, 2048, 16384)
DECODE_M = (1, 16, 64, 128, 256)
QWEN_M = (4096, 16384)
JVT_M = (18432,)
NAMED_SHAPES = {
    "8192^3": ((8192,), 8192, 8192),
    "gemma_qkv": (GEMMA_M, 16384, 5376),
    "gemma_o": (GEMMA_M, 5376, 8192),
    "gemma_gate_up": (GEMMA_M, 2 * 21504, 5376),
    "gemma_down": (GEMMA_M, 5376, 21504),
    "gemma_qkv_decode": (DECODE_M, 16384, 5376),
    "gemma_o_decode": (DECODE_M, 5376, 8192),
    "gemma_gate_up_decode": (DECODE_M, 2 * 21504, 5376),
    "gemma_down_decode": (DECODE_M, 5376, 21504),
    "qwen_qkv": (QWEN_M, 12288, 4096),
    "qwen_out": (QWEN_M, 4096, 4096),
    "qwen_up_gate": (QWEN_M, 2 * 12288, 4096),
    "qwen_down": (QWEN_M, 4096, 12288),
    "jvt_qkv": (JVT_M, 10752, 3584),
    "jvt_fc1": (JVT_M, 2 * 14336, 3584),
    "jvt_fc2": (JVT_M, 3584, 14336),
}
REFERENCE_CHOICES = {
    "mxfp8": (("torch mxfp8", "torch rowwise fast"),),
    "fp8_b128_e8m0": (("torch blockwise", "torch rowwise fast"),),
    "fp8_b128_fp32": (("torch blockwise", "torch rowwise fast"),),
    "fp8_rowwise": (("torch rowwise fast",), ("torch rowwise",)),
}
REFERENCE_RECIPES = {
    "torch rowwise": "fp8_rowwise",
    "torch rowwise fast": "fp8_rowwise",
    "torch blockwise": "fp8_b128_fp32",
    "torch mxfp8": "mxfp8",
}


@dataclass
class Shape:
    name: str
    m: int
    n: int
    k: int


@dataclass
class Result:
    device: str
    shape: str
    m: int
    n: int
    k: int
    op: str
    recipe: str
    kernel: str
    status: str
    seconds: Optional[float] = None
    tflops: Optional[float] = None
    sol_percent: Optional[float] = None
    dram_gbps: Optional[float] = None
    reference_percents: dict = field(default_factory=dict)
    relative_error: Optional[float] = None
    agrees: Optional[bool] = None
    clock_mhz: Optional[float] = None
    power_w: Optional[float] = None


def parse_shapes(texts):
    shapes = []
    for text in texts:
        if text in NAMED_SHAPES or ":" in text:
            name, _, rows = text.partition(":")
            if name not in NAMED_SHAPES:
                raise SystemExit(f"benchmark_fp8_mm: no named shape {name!r}; the names are {', '.join(NAMED_SHAPES)}")
            sizes, n, k = NAMED_SHAPES[name]
            for m in (int(rows),) if rows else sizes:
                shapes.append(Shape(name if name == "8192^3" else f"{name}:{m}", m, n, k))
        else:
            m, n, k = (int(size) for size in text.split("x"))
            shapes.append(Shape(text, m, n, k))
    return shapes


def device_entry(name):
    return next((entry for key, entry in DEVICES.items() if key in name), None)


def peak_flops(device, recipe):
    if device is None:
        return None
    return 1e12 * (device.block_scaled_tflops if recipe in BLOCK_SCALED else device.f8f6f4_tflops)


def sol_seconds(device, recipe, flops, dram_bytes):
    if device is None:
        return None
    return max(flops / peak_flops(device, recipe), dram_bytes / (1e9 * device.copy_gbps))


def nbytes(*tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None)


def relative(y, oracle):
    return (torch.linalg.norm(y.float() - oracle) / torch.linalg.norm(oracle)).item()


def quantized(recipe, x, w):
    if recipe == "mxfp8":
        return fp8_pack.mxfp8_quant(x), fp8_pack.mxfp8_quant(w)
    if recipe == "fp8_rowwise":
        return fp8_pack.fp8_rowwise_quant(x), fp8_pack.fp8_rowwise_quant(w)
    e8m0 = recipe == "fp8_b128_e8m0"
    return fp8_pack.fp8_b128_quant(x, e8m0=e8m0), fp8_pack.fp8_b128_quant_weight(w, e8m0=e8m0)


def pack_recipe(recipe):
    return "fp8_b128" if recipe.startswith("fp8_b128") else recipe


def row_bands(m):
    last = GROUP * ((m - 1) // GROUP)
    return sorted({(0, min(m, CHECK_ROWS)), (last, m)})


def band_of(pack, start, end):
    values, scales = pack
    scales = scales[start:end] if scales.dim() == 2 else scales[start // GROUP:start // GROUP + 1]
    return values[start:end], scales


def oracle(recipe, a, b, m):
    """Returns the FP32 product of the dequantized packs on the rows of `row_bands(m)`."""
    torch.backends.cuda.matmul.allow_tf32 = False
    dequantized_b = packs.dequantize(*b, pack_recipe(recipe))
    rows = [packs.dequantize(*band_of(a, start, end), pack_recipe(recipe)) @ dequantized_b.t()
            for start, end in row_bands(m)]
    return torch.cat(rows)


def checked_rows(y, m):
    return torch.cat([y[start:end] for start, end in row_bands(m)])


def kernel_call(op, recipe, a, s_a, b, s_b):
    ops = torch.ops.zig_cu_kernels
    prefix = "fp8_b128" if recipe.startswith("fp8_b128") else recipe
    if op == "mm_glu":
        if recipe == "mxfp8":
            return lambda: ops.mxfp8_mm_glu(a, s_a, b, s_b)
        return lambda: getattr(ops, f"{prefix}_mm_glu")(a, s_a, b, s_b, "swiglu", False)
    if recipe == "fp8_rowwise":
        return lambda: ops.fp8_rowwise_mm_abt(a, s_a, b, s_b, None)
    return lambda: getattr(ops, f"{prefix}_mm_abt")(a, s_a, b, s_b)


def output_bytes(op, recipe, m, n):
    if op == "mm_abt":
        return 2 * m * n
    outputs = fp8_mm.glu_shapes(recipe, m, n // 2, n // 2, False)
    return sum(torch.empty(shape, dtype=dtype, device="meta").nbytes for shape, dtype in filter(None, outputs))


def blockwise_operands(a, s_a, b, s_b, m, n, k):
    """Returns the operands of `torch._scaled_mm` blockwise scaling for an FP32-scaled block128 pack. cuBLAS needs
    16-byte aligned scale strides, so K grows to a multiple of `BLOCKWISE_K_MULTIPLE` with zero values."""
    padded_k = packs.round_up(k, BLOCKWISE_K_MULTIPLE)
    a_padded = torch.zeros((m, padded_k), dtype=torch.uint8, device="cuda")
    a_padded[:, :k] = a.view(torch.uint8)
    b_padded = torch.zeros((n, padded_k), dtype=torch.uint8, device="cuda")
    b_padded[:, :k] = b.view(torch.uint8)
    scale_a = torch.ones((padded_k // GROUP, m), device="cuda")
    scale_a[:k // GROUP] = packs.scale_grid(s_a, m, k // GROUP).t()
    scale_b = torch.ones((n // GROUP, padded_k // GROUP), device="cuda")
    scale_b[:, :k // GROUP] = s_b[:, :k // GROUP]
    return a_padded.view(FP8), scale_a.t(), b_padded.view(FP8), scale_b.t()


def reference_calls(packed, m, n, k):
    """Returns the `torch._scaled_mm` calls by name, each as its tensors and a function of the tensors."""
    calls = {}
    if "fp8_rowwise" in packed:
        (a, s_a), (b, s_b) = packed["fp8_rowwise"]
        for name, fast in (("torch rowwise", False), ("torch rowwise fast", True)):
            calls[name] = ((a, s_a, b, s_b), lambda a, s_a, b, s_b, fast=fast: torch._scaled_mm(
                a, b.t(), s_a, s_b.t(), out_dtype=BF16, use_fast_accum=fast))
    if "fp8_b128_fp32" in packed:
        (a, s_a), (b, s_b) = packed["fp8_b128_fp32"]
        calls["torch blockwise"] = (blockwise_operands(a, s_a, b, s_b, m, n, k), lambda a, s_a, b, s_b: (
            torch._scaled_mm(a, b.t(), s_a, s_b, out_dtype=BF16)))
    if "mxfp8" in packed:
        (a, s_a), (b, s_b) = packed["mxfp8"]
        e8m0 = torch.float8_e8m0fnu
        calls["torch mxfp8"] = ((a, s_a.view(e8m0).flatten(), b, s_b.view(e8m0).flatten()),
                                lambda a, s_a, b, s_b: torch._scaled_mm(a, b.t(), s_a, s_b, out_dtype=BF16))
    return calls


def rotated(make, tensors):
    """Returns a `timing.Rotation` over clones of `tensors` that together hold several times the L2 cache."""
    copies = [tensors] + [tuple(tensor.clone() for tensor in tensors)
                          for _ in range(timing.rotation_copies(nbytes(*tensors)) - 1)]
    return timing.Rotation(make(*copy) for copy in copies)


def measurement_seconds(window_seconds, idle=False):
    warm = timing.PREHEAT_SECONDS if idle else timing.REHEAT_SECONDS
    return warm + window_seconds + timing.FILL_SECONDS + MEASUREMENT_OVERHEAD_SECONDS


class Budget:
    """Keeps the process within `seconds` of its start. Each measurement lasts `timing.MEASURE_SECONDS`, because the
    budget leaves `timing.measure` no reserve to lengthen a measurement to a power cycle of a power-capped GPU."""

    def __init__(self, seconds):
        self.margin = timing.PROCESS_SECONDS - seconds
        self.skipped = []
        timing.CYCLE_RESERVE_SECONDS = math.inf

    def remaining(self):
        return timing.watchdog.remaining() - self.margin - timing.TEARDOWN_SECONDS

    @staticmethod
    def next_measurement_seconds():
        idle = timing.last_measurement_end is None or \
            time.monotonic() - timing.last_measurement_end > timing.IDLE_SECONDS
        return measurement_seconds(timing.MEASURE_SECONDS, idle)

    def allows(self, seconds):
        return self.remaining() >= seconds

    def skip(self, what):
        self.skipped.append(what)
        print(f"skipped (time budget): {what}", flush=True)


def measured_seconds(call, rows, sol):
    if rows <= GRAPH_ROWS:
        return timing.graph_seconds(call, sol_seconds=sol)
    return timing.measure(call, sol).seconds


def print_result(result):
    if result.status != "ok":
        print(f"{result.shape:20} {result.op:7} {result.recipe:18} {result.kernel:31} {result.status}", flush=True)
        return
    sol = "" if result.sol_percent is None else f"{result.sol_percent:5.1f}% of SoL"
    versus = ", ".join(f"{percent:5.1f}% of {name}" for name, percent in result.reference_percents.items())
    check = "" if result.relative_error is None else \
        f"err {result.relative_error:.1e}{'' if result.agrees else ' DISAGREES'}"
    clock = "" if result.clock_mhz is None else f"{result.clock_mhz:4.0f} MHz {result.power_w:3.0f} W"
    print(f"{result.shape:20} {result.op:7} {result.recipe:18} {result.kernel:31} {result.seconds * 1e3:8.3f} ms "
          f"{result.tflops:7.1f} TF/s {result.dram_gbps:6.0f} GB/s {sol:15} {versus:32} {check:20} {clock}", flush=True)


class Run:
    def __init__(self, arguments):
        self.ops, self.recipes, self.json = arguments.ops, arguments.recipes, arguments.json
        self.budget = Budget(arguments.budget)
        self.device_name = torch.cuda.get_device_name()
        self.device = device_entry(self.device_name)
        self.arch = fp8_mm.arch_of(runtime.compute_capability(torch.cuda.current_device()))
        self.results = []

    def record(self, result):
        self.results.append(result)
        print_result(result)
        if self.json:
            with open(self.json, "w") as file:
                json.dump([asdict(result) for result in self.results], file, indent=1)
        if result.status == "ok":
            temperature = timing.last_telemetry[2] if timing.last_telemetry else None
            dse.log(f"{result.kernel} {result.shape}", f"{result.tflops:.1f}", "TF/s", temperature_c=temperature)

    def result(self, shape, op, recipe, kernel, status):
        return Result(self.device_name, shape.name, shape.m, shape.n, shape.k, op, recipe, kernel, status)

    def timed(self, result, call, flops, dram_bytes):
        sol = sol_seconds(self.device, result.recipe, flops, dram_bytes)
        if not self.budget.allows(self.budget.next_measurement_seconds()):
            self.budget.skip(f"{result.shape} {result.op} {result.recipe} {result.kernel}")
            result.status = "skipped (time budget)"
            return None
        try:
            result.seconds = measured_seconds(call, result.m, sol)
        except Exception as error:  # noqa: BLE001
            result.status = f"failed: {type(error).__name__}: {str(error).splitlines()[0][:120]}"
            return None
        result.tflops = flops / result.seconds * 1e-12
        result.dram_gbps = dram_bytes / result.seconds * 1e-9
        result.sol_percent = None if sol is None else 100 * sol / result.seconds
        if timing.last_telemetry is not None:
            result.clock_mhz, result.power_w = timing.last_telemetry[:2]
        result.status = "ok"
        return result.seconds

    def shape(self, shape):
        m, n, k = shape.m, shape.n, shape.k
        if not self.budget.allows(SETUP_SECONDS + self.budget.next_measurement_seconds()):
            self.budget.skip(f"{shape.name} (every case)")
            return
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        x = torch.randn((m, k), device="cuda", generator=generator, dtype=BF16)
        w = torch.randn((n, k), device="cuda", generator=generator, dtype=BF16)
        packed = {}
        for recipe in self.quantized_recipes():
            try:
                packed[recipe] = quantized(recipe, x, w)
            except Exception as error:  # noqa: BLE001
                print(f"{shape.name:20} quantizer {recipe} unavailable: {type(error).__name__}: {error}", flush=True)
        del x, w
        oracles = {recipe: oracle(recipe, *pair, m) for recipe, pair in packed.items()}
        references = self.references(shape, packed, oracles)
        plan = self.plan(shape, packed, references)
        print(f"{shape.name:20} M {m} N {n} K {k}: {len(plan)} measurements", flush=True)
        reference_seconds = {}
        for kind, op, recipe in plan:
            if kind == "reference":
                reference_seconds[recipe] = self.reference(shape, recipe, references[recipe])
            else:
                self.kernel(shape, op, recipe, packed[recipe], oracles[recipe], reference_seconds, references)

    def quantized_recipes(self):
        """Returns the recipes of the kernels and of their references, whose packs `reference_calls` reads."""
        recipes = dict.fromkeys(self.recipes + ["fp8_rowwise"])
        if any(recipe.startswith("fp8_b128") for recipe in self.recipes):
            recipes["fp8_b128_fp32"] = None
        return list(recipes)

    def references(self, shape, packed, oracles):
        """Runs each reference once and returns the ones that ran, with their tensors, call and agreement."""
        ready = {}
        for name, (tensors, function) in reference_calls(packed, shape.m, shape.n, shape.k).items():
            try:
                y = function(*tensors)
            except Exception as error:  # noqa: BLE001
                reason = str(error).splitlines()[0][:120]
                print(f"{shape.name:20} {name} unavailable: {type(error).__name__}: {reason}", flush=True)
                continue
            error = relative(checked_rows(y, shape.m), oracles[REFERENCE_RECIPES[name]])
            ready[name] = (tensors, function, error)
        return ready

    def plan(self, shape, packed, references):
        plan = []
        for op in self.ops:
            for recipe in self.recipes:
                for reference in self.matched_references(recipe, references):
                    if ("reference", "mm_abt", reference) not in plan:
                        plan.append(("reference", "mm_abt", reference))
                if recipe in packed:
                    plan.append(("kernel", op, recipe))
        return plan + [("reference", "mm_abt", name) for name in references if ("reference", "mm_abt", name) not in plan]

    @staticmethod
    def matched_references(recipe, references):
        """Returns the first reference that ran of each choice of `REFERENCE_CHOICES`."""
        matches = (next((name for name in names if name in references), None) for names in REFERENCE_CHOICES[recipe])
        return [name for name in matches if name is not None]

    def reference(self, shape, name, prepared):
        tensors, function, error = prepared
        result = self.result(shape, "mm_abt", name, f"torch._scaled_mm {name.removeprefix('torch ')}", "")
        result.relative_error, result.agrees = error, error <= AGREEMENT
        dram_bytes = nbytes(*tensors) + 2 * shape.m * shape.n
        seconds = self.timed(result, rotated(lambda *t: lambda: function(*t), tensors), 2 * shape.m * shape.n * shape.k,
                             dram_bytes)
        self.record(result)
        return seconds

    def kernel(self, shape, op, recipe, pair, exact, reference_seconds, references):
        name = fp8_mm.kernel_name(op, recipe, self.arch)
        result = self.result(shape, op, recipe, name, "")
        if not fp8_mm.available(recipe, op):
            result.status = f"unavailable: no {name} kernel on {self.device_name}"
            self.record(result)
            return
        (a, s_a), (b, s_b) = pair
        try:
            y = kernel_call(op, recipe, a, s_a, b, s_b)()
        except Exception as error:  # noqa: BLE001
            result.status = f"failed: {type(error).__name__}: {str(error).splitlines()[0][:120]}"
            self.record(result)
            return
        if op == "mm_abt":
            result.relative_error = relative(checked_rows(y, shape.m), exact)
            result.agrees = result.relative_error <= AGREEMENT
        del y
        dram_bytes = nbytes(a, s_a, b, s_b) + output_bytes(op, recipe, shape.m, shape.n)
        call = rotated(lambda *t: kernel_call(op, recipe, *t), (a, s_a, b, s_b))
        seconds = self.timed(result, call, 2 * shape.m * shape.n * shape.k, dram_bytes)
        for reference in self.matched_references(recipe, references) if seconds is not None else ():
            if reference_seconds.get(reference) is not None:
                label = reference if references[reference][2] <= AGREEMENT else f"{reference} (disagrees)"
                result.reference_percents[label] = 100 * reference_seconds[reference] / seconds
        self.record(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ops", nargs="+", choices=OPS, default=["mm_abt"])
    parser.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    parser.add_argument("--shapes", nargs="+", default=["8192^3"],
                        help=f"MxNxK, a name ({', '.join(NAMED_SHAPES)}) for all its M, or NAME:M")
    parser.add_argument("--budget", type=float, default=BUDGET_SECONDS,
                        help="seconds after the process start by which the measurements end")
    parser.add_argument("--json", help="write the results to this file")
    arguments = parser.parse_args()
    shapes = parse_shapes(arguments.shapes)
    run = Run(arguments)
    capability = runtime.compute_capability(torch.cuda.current_device())
    peaks = "no peak rates" if run.device is None else \
        f"P {run.device.f8f6f4_tflops:.0f} TF/s ({run.device.block_scaled_tflops:.0f} block-scaled), " \
        f"W {run.device.copy_gbps:.0f} GB/s"
    print(f"{run.device_name} (compute capability {capability // 10}.{capability % 10}, power limit "
          f"{timing.power_limit_w():.0f} W): {peaks}", flush=True)
    for shape in shapes:
        run.shape(shape)
    print(f"process time {timing.process_age():.1f} s", flush=True)
    if run.budget.skipped:
        print(f"skipped {len(run.budget.skipped)} cases to stay within {arguments.budget:.1f} s: "
              f"{'; '.join(run.budget.skipped)}", flush=True)


if __name__ == "__main__":
    main()
