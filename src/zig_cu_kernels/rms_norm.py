import functools
import os
import re
import itertools
from collections import namedtuple
from dataclasses import astuple, dataclass

from . import runtime
from .runtime import fp32
from .tensors import is_aligned, require, require_device, require_dtype, rows_view, view_of

SIGNATURE = "pupupupppupupupupuuuuf"
X_DTYPES = {"bf16": "bfloat16", "f32": "float32"}
ADDS = ("none", "residual", "gated", "dual", "dual_gated")
WEIGHT_DTYPES = {"none": None, "bf16": "bfloat16", "f32": "float32", "zero_centered_bf16": "bfloat16",
                 "zero_centered_f32": "float32"}
MODULATIONS = ("none", "scale", "scale_shift")
MODULATION_NAMES = ("gate", "scale", "shift")
OUTPUT_DTYPES = {"bf16": "bfloat16", "f32": "float32", "fp8_tensorwise": "float8_e4m3fn",
                 "fp8_rowwise": "float8_e4m3fn", "fp8_blockwise": "float8_e4m3fn",
                 "fp8_blockwise_e8m0": "float8_e4m3fn"}
SCALE_DTYPES = {"fp8_tensorwise": "float32", "fp8_rowwise": "float32", "fp8_blockwise": "float32",
                "fp8_blockwise_e8m0": "float8_e8m0fnu"}
DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "float8_e4m3fn": 1, "float8_e8m0fnu": 1}
BLOCKWISE_OUTPUTS = ("fp8_blockwise", "fp8_blockwise_e8m0")
ADD_NAMES = {
    "none": "",
    "residual": "_add",
    "gated": "_gated",
    "dual": "_dual",
    "dual_gated": "_dual_gated",
}
VALUES_PER_CHUNK = 8
SCALE_GROUP = 128
CHUNK_BYTES = 16
FP8_CHUNK_BYTES = 8
MAX_U32 = (1 << 32) - 1
MAX_GRID_X = (1 << 31) - 1
WARP_THREADS = 32
WORK_GROUP_THREADS = 128
KIB = 1024

Matrix = namedtuple("Matrix", "pointer rows width stride")
ABSENT = Matrix(0, 0, 0, 0)
# A footprint describes the memory of an operand as `rows` runs of `row_bytes` bytes that start `stride` bytes apart.
Footprint = namedtuple("Footprint", "pointer rows row_bytes stride")


@dataclass(frozen=True)
class Options:
    """The options of an RMS norm instance, as `zig/rms_norm_configurations.zig` defines them. The build compiles each
    instance into one module that holds one kernel per size, where a size is a count of chunks per thread."""

    x: str = "bf16"
    add: str = "none"
    weight: str = "none"
    modulation: str = "none"
    output: str = "bf16"

    def __post_init__(self):
        for name, allowed in (("x", X_DTYPES), ("add", ADDS), ("weight", WEIGHT_DTYPES),
                              ("modulation", MODULATIONS), ("output", OUTPUT_DTYPES)):
            require(getattr(self, name) in allowed, f"{name} must be one of {tuple(allowed)}")

    @functools.cached_property
    def name(self):
        weight = ""
        if self.weight != "none":
            weight = f"_{self.weight}" if self.weight.startswith("zero_centered") else f"_weight_{self.weight}"
        modulation = "" if self.modulation == "none" else f"_{self.modulation}"
        return f"rms_norm_{self.x}{ADD_NAMES[self.add]}{weight}{modulation}_to_{self.output}"

    @property
    def gated(self):
        return self.add in ("gated", "dual_gated")

    @property
    def normalized_branch(self):
        return self.add in ("dual", "dual_gated")

    @property
    def modulated(self):
        return self.gated or self.modulation != "none"

    @property
    def fp8(self):
        return self.output in SCALE_DTYPES

    @property
    def blockwise(self):
        return self.output in BLOCKWISE_OUTPUTS

    @functools.cached_property
    def operands(self):
        """The names of the operands of `plan` besides `x` and `y` that the instance reads or writes."""
        used = set()
        if self.add != "none":
            used |= {"residual" if self.add == "residual" else "branch", "residual_out"}
        if self.weight != "none":
            used |= {"weight", "branch_weight"} if self.normalized_branch else {"weight"}
        if self.gated:
            used.add("gate")
        if self.modulation != "none":
            used |= {"scale", "shift"} if self.modulation == "scale_shift" else {"scale"}
        if self.fp8:
            used.add("fp8_scales")
        return frozenset(used)


@dataclass
class Tuning:
    """The launch choices that are not fixed by the operands. The register engine loads rows with `ld.global`; the bulk
    engine stages rows in shared memory with bulk copies on compute capability 9.0 and up. `auto` picks the bulk
    engine where it runs."""

    engine: str = os.environ.get("ZIG_CU_KERNELS_RMS_NORM_ENGINE", "registers")
    bulk_threads: int = 512
    bulk_stages: int = 0
    bulk_work_groups_per_sm: int = 1


tuning = Tuning()
# The suffix of a kernel name that names its size and engine.
SIZE_SUFFIX = re.compile(r"_c\d+_max\d+(_bulk)?$")
ENGINES = ("auto", "registers", "bulk")
ENGINE_SUFFIXES = {"registers": "", "bulk": "_bulk"}
BULK_CAPABILITY = 90
MAX_STAGES = 32
UNLIMITED_CAPABILITY = 90
FP8_CHUNKS = 4
EIGHT_CHUNKS = 8
SIX_CHUNKS = 6
EIGHT_CHUNKS_CAPABILITY = 100
MIN_EIGHT_CHUNK_THREADS = 16
MIN_TWO_CHUNK_WIDTH = 128
MIN_STAGES = 2


@functools.cache
def sizes(instance, engine="registers"):
    _, entries = runtime.manifest()
    pattern = re.compile(rf"{re.escape(instance)}_c\d+_max\d+{ENGINE_SUFFIXES[engine]}")

    def matches(name):
        return pattern.fullmatch(name) is not None

    return tuple(sorted((entry for name, entry in entries.items() if matches(name)),
                        key=lambda entry: entry.max_width))


def preferred_chunks(device_id, options, width):
    """Returns the chunks per work item that the launch prefers, or None for the smallest size. On compute capability
    9.0 and up, an FP8 output costs more instructions per byte than a BF16 output, and more chunks per work item spread
    the per-row costs over more values: on the H100 and the B300, fp8_rowwise ran 7% and 27% faster with four chunks
    than with two. `exact_chunks` chooses for the FP8 outputs without an addend. On the H100, eight chunks ran up to
    20% slower than four except for the tensorwise output, whose quantization needs no reduction, so the other FP8
    outputs there start from four. On compute capability 10.0 and up, rows of 128 to 255 values of the other outputs
    take two chunks, with which the plain norm of rows of 128 values ran 8% faster on the B300 than with one."""
    capability = runtime.compute_capability(device_id)
    if not options.fp8:
        narrow = MIN_TWO_CHUNK_WIDTH <= width < 2 * MIN_TWO_CHUNK_WIDTH
        return 2 if narrow and capability >= EIGHT_CHUNKS_CAPABILITY else None
    if capability < UNLIMITED_CAPABILITY:
        return None
    if options.add != "none":
        return FP8_CHUNKS
    if capability >= EIGHT_CHUNKS_CAPABILITY or options.output == "fp8_tensorwise":
        return exact_chunks(options, width, (EIGHT_CHUNKS, FP8_CHUNKS, SIX_CHUNKS, 2))
    return exact_chunks(options, width, (FP8_CHUNKS, SIX_CHUNKS, 2))


def exact_chunks(options, width, choices):
    """Returns the first of `choices` that splits a row of `width` values among work items without padding, and for a
    blockwise output into whole scale groups, or 4 when none does. Eight chunks per work item also need at least 16
    work items per row. On the B300, the exact split ran up to 33% faster than a padded one, eight chunks ran up to 15%
    faster than four, rows of 128 values ran 17-24% faster with 4 work items of 4 chunks than with 2 of 8, and rows of
    1536 values ran 17-23% faster with 32 work items of 6 chunks than with 96 of 2. On the H100, 6 chunks ran up to 8%
    faster than a padded split of 4 at widths 768 and 1536."""
    chunks = width // VALUES_PER_CHUNK
    for chunks_per_thread in choices:
        threads_per_row, _ = row_layout(chunks, chunks_per_thread)
        exact = threads_per_row * chunks_per_thread == chunks
        wide = chunks_per_thread < EIGHT_CHUNKS or threads_per_row >= MIN_EIGHT_CHUNK_THREADS
        grouped = not options.blockwise or threads_per_row % (SCALE_GROUP // VALUES_PER_CHUNK) == 0
        if exact and wide and grouped:
            return chunks_per_thread
    return FP8_CHUNKS


def engine_for(device_id, options):
    require(tuning.engine in ENGINES, f"the engine must be one of {ENGINES}")
    if tuning.engine != "auto":
        return tuning.engine
    built = sizes(options.name, "bulk")
    if built and runtime.compute_capability(device_id) >= BULK_CAPABILITY and \
            runtime.unavailable_reason(built[0].kernel, device_id) is None:
        return "bulk"
    return "registers"


def is_built(options):
    return bool(sizes(options.name))


def unavailable_reason(options, device_id=None):
    """Returns why the device cannot run `options`, or None when it can, from the module that all sizes share."""
    built = sizes(options.name)
    if not built:
        return f"no built kernel combines these options: {options.name}"
    return runtime.unavailable_reason(built[0].kernel, device_id)


def is_available(options, device_id=None):
    return unavailable_reason(options, device_id) is None


@functools.cache
def built_options():
    _, entries = runtime.manifest()
    names = {SIZE_SUFFIX.sub("", name) for name in entries}
    every = (Options(*values) for values in itertools.product(X_DTYPES, ADDS, WEIGHT_DTYPES, MODULATIONS,
                                                                OUTPUT_DTYPES))
    return tuple(options for options in every if options.name in names)


def neighbors(options):
    values = astuple(options)
    return [other.name for other in built_options()
            if sum(a != b for a, b in zip(astuple(other), values)) == 1]


def require_built(options):
    if not is_built(options):
        raise ValueError(f"no built kernel combines these options: {options.name}; the built kernels that differ in "
                         f"one option are {neighbors(options)}")


def size_for(options, width, engine="registers", chunks=None):
    """Returns the smallest size of at least two chunks per work item, or of `chunks` chunks when a size of that many
    chunks takes `width`. The row count never selects the size, so the RMS norms stay batch invariant (README section
    "Determinism")."""
    built = sizes(options.name, engine)
    require_built(options)
    for entry in built:
        if entry.chunks_per_thread == chunks and width <= entry.max_width and fits_scale_groups(options, width, entry):
            return entry
    least = 2 if width >= 32 * VALUES_PER_CHUNK else 1
    for entry in built:
        if width <= entry.max_width and entry.chunks_per_thread >= least:
            return entry
    raise ValueError(f"{options.name} takes rows of at most {built[-1].max_width} values, got {width}")


def matrix_of(name, tensor, dtype, stream=-1, row_bytes=CHUNK_BYTES, output=False):
    view = view_of(tensor, stream, name)
    require_dtype(view, name, dtype)
    rows, width, stride = rows_view(view)
    if not is_aligned(view):
        raise ValueError(f"{name} must start at a multiple of {CHUNK_BYTES} bytes")
    if rows > 1 and width and stride * view.itemsize % row_bytes:
        raise ValueError(f"the row stride of {name} must be a multiple of {row_bytes} bytes")
    if output and rows > 1 and stride < width:
        raise ValueError(f"the rows of {name} overlap each other")
    if rows > MAX_U32 or stride > MAX_U32:
        raise ValueError(f"the row count and the row stride of {name} must lie below 2^32")
    return Matrix(view.pointer, rows, width, stride if rows > 1 else 0)


def fits(check, *arguments, **options):
    try:
        check(*arguments, **options)
    except ValueError:
        return False
    return True


def row_matrix_of(name, tensor, dtype, rows, width, stream=-1, row_bytes=CHUNK_BYTES, output=False):
    matrix = matrix_of(name, tensor, dtype, stream, row_bytes, output)
    if (matrix.rows, matrix.width) != (rows, width):
        raise ValueError(f"{name} must have {rows} rows of {width} values")
    return matrix


def vector_of(name, tensor, dtype, width, stream=-1):
    view = view_of(tensor, stream, name)
    require_dtype(view, name, dtype)
    if view.shape != (width,):
        raise ValueError(f"{name} must hold {width} values")
    if width > 1 and view.strides != (1,):
        raise ValueError(f"{name} must be contiguous")
    if not is_aligned(view):
        raise ValueError(f"{name} must start at a multiple of {CHUNK_BYTES} bytes")
    return Matrix(view.pointer, 1, width, 0)


def modulation_of(name, tensor, width, samples_needed, stream=-1):
    matrix = matrix_of(name, tensor, "bfloat16", stream)
    if matrix.width != width:
        raise ValueError(f"{name} must have rows of {width} values")
    if matrix.rows < samples_needed:
        raise ValueError(f"{name} must have at least {samples_needed} rows")
    return matrix


def fp8_scales_of(options, scales, rows, width, stream=-1):
    """Returns the footprint of the FP8 scales and the distance in elements between the scales of two rows for rowwise
    scales, or between the scales of two scale groups for blockwise scales."""
    view = view_of(scales, stream, "the FP8 scales")
    require_dtype(view, "the FP8 scales", SCALE_DTYPES[options.output])
    if view.pointer % view.itemsize:
        raise ValueError(f"the FP8 scales must start at a multiple of {view.itemsize} bytes")
    count = 1
    for size in view.shape:
        count *= size
    if options.output == "fp8_tensorwise":
        if count != 1:
            raise ValueError("the tensorwise FP8 scale must hold one value")
        return Footprint(view.pointer, 1, view.itemsize, 0), 0
    if options.output == "fp8_rowwise":
        if view.shape != (rows, 1):
            raise ValueError(f"the rowwise FP8 scales must have the shape {(rows, 1)}")
        stride = view.strides[0] if rows > 1 else 1
        if not 1 <= stride <= MAX_U32:
            raise ValueError("the rowwise FP8 scales of two rows must not overlap")
        return Footprint(view.pointer, rows, view.itemsize, stride * view.itemsize), stride
    scale_groups = width // SCALE_GROUP
    if view.shape != (rows, scale_groups):
        raise ValueError(f"the blockwise FP8 scales must have the shape {(rows, scale_groups)}")
    if rows > 1 and scale_groups and view.strides[0] != 1:
        raise ValueError("the blockwise FP8 scales of a scale group must be contiguous")
    stride = view.strides[1] if scale_groups > 1 else rows
    if not rows <= stride <= MAX_U32:
        raise ValueError(f"the blockwise FP8 scales of two scale groups must lie at least {rows} elements apart")
    return Footprint(view.pointer, scale_groups if rows else 0, rows * view.itemsize, stride * view.itemsize), stride


def footprint(matrix, dtype):
    itemsize = DTYPE_BYTES[dtype]
    return Footprint(matrix.pointer, matrix.rows, matrix.width * itemsize, matrix.stride * itemsize)


def byte_range(span):
    if span.rows == 0 or span.row_bytes == 0:
        return None
    return span.pointer, span.pointer + (span.rows - 1) * span.stride + span.row_bytes


def overlaps(first, second):
    first_range, second_range = byte_range(first), byte_range(second)
    if first_range is None or second_range is None:
        return False
    return first_range[0] < second_range[1] and second_range[0] < first_range[1]


def shares_memory(first, second):
    """Returns whether a byte belongs to both footprints. The answer is exact when the footprints have the same stride
    or one of them has one row, and otherwise says whether their byte ranges overlap."""
    if not overlaps(first, second):
        return False
    if first.rows > 1 and second.rows > 1 and first.stride != second.stride:
        return True
    stride = first.stride if first.rows > 1 else second.stride
    if first.rows == 1 and second.rows == 1:
        return True
    if stride <= 0:
        return True
    offset = second.pointer - first.pointer
    lowest = max((offset - first.row_bytes) // stride + 1, 1 - second.rows)
    highest = min(-(-(offset + second.row_bytes) // stride) - 1, first.rows - 1)
    return lowest <= highest


def require_operands(options, operands):
    for name, tensor in operands.items():
        if tensor is None and name in options.operands:
            raise TypeError(f"{options.name} needs {name}")
        if tensor is not None and name not in options.operands:
            raise ValueError(f"{options.name} takes no {name}")


def require_separate(outputs, inputs):
    for index, (name, span, dtype, aliases) in enumerate(outputs):
        for input_name, input_span, input_dtype in inputs:
            exact = input_name in aliases and input_span == span and input_dtype == dtype
            if not exact and overlaps(span, input_span):
                raise ValueError(f"{name} overlaps {input_name}")
        for other_name, other_span, _, _ in outputs[index + 1 :]:
            if shares_memory(span, other_span):
                raise ValueError(f"{name} and {other_name} share memory")


def plan(options, x, y, *, residual=None, branch=None, residual_out=None, weight=None, branch_weight=None, gate=None,
         scale=None, shift=None, fp8_scales=None, rows_per_sample=1, stream=-1):
    """Checks the arguments of `rms_norm` and returns its launch and the pointers of the operands, or None when `x` has
    no values. `view_of` passes `stream` to the producers of the operands."""
    operands = {"residual": residual, "branch": branch, "residual_out": residual_out, "weight": weight,
                "branch_weight": branch_weight, "gate": gate, "scale": scale, "shift": shift, "fp8_scales": fp8_scales}
    require_operands(options, operands)
    x_dtype = X_DTYPES[options.x]
    source = matrix_of("x", x, x_dtype, stream)
    rows, width = source.rows, source.width
    if width % VALUES_PER_CHUNK:
        raise ValueError(f"the row length must be a multiple of {VALUES_PER_CHUNK}")
    if options.blockwise and width % SCALE_GROUP:
        raise ValueError(f"the blockwise FP8 outputs need a row length that is a multiple of {SCALE_GROUP}")
    if not 1 <= rows_per_sample <= MAX_U32:
        raise ValueError("rows_per_sample must lie between 1 and 2^32 - 1")
    out_dtype = OUTPUT_DTYPES[options.output]
    row_bytes = FP8_CHUNK_BYTES if options.fp8 else CHUNK_BYTES
    target = row_matrix_of("y", y, out_dtype, rows, width, stream, row_bytes, output=True)
    inputs = [("x", footprint(source, x_dtype), x_dtype)]
    outputs = []
    other = residual_out_matrix = ABSENT
    if options.add != "none":
        addend, addend_dtype = ("residual", x_dtype) if options.add == "residual" else ("branch", "bfloat16")
        other = row_matrix_of(addend, operands[addend], addend_dtype, rows, width, stream)
        inputs.append((addend, footprint(other, addend_dtype), addend_dtype))
        residual_out_matrix = row_matrix_of("residual_out", residual_out, x_dtype, rows, width, stream, output=True)
        outputs.append(("residual_out", footprint(residual_out_matrix, x_dtype), x_dtype,
                        ("x", "residual", "branch")))
    outputs.append(("y", footprint(target, out_dtype), out_dtype, ("x", "residual", "branch")))
    weight_dtype = WEIGHT_DTYPES[options.weight]
    weights = branch_weights = ABSENT
    if weight_dtype:
        weights = vector_of("weight", weight, weight_dtype, width, stream)
        inputs.append(("weight", footprint(weights, weight_dtype), weight_dtype))
    if "branch_weight" in options.operands:
        branch_weights = vector_of("branch_weight", branch_weight, weight_dtype, width, stream)
        inputs.append(("branch_weight", footprint(branch_weights, weight_dtype), weight_dtype))
    samples = -(-rows // rows_per_sample) if options.modulated else 0
    modulation = {}
    for name in MODULATION_NAMES:
        used = name in options.operands
        modulation[name] = modulation_of(name, operands[name], width, samples, stream) if used else ABSENT
        if used:
            inputs.append((name, footprint(modulation[name], "bfloat16"), "bfloat16"))
    fp8_pointer, fp8_stride = 0, 0
    if options.fp8:
        scale_span, fp8_stride = fp8_scales_of(options, fp8_scales, rows, width, stream)
        fp8_pointer = scale_span.pointer
        scale_dtype = SCALE_DTYPES[options.output]
        if options.output == "fp8_tensorwise":
            inputs.append(("fp8_scales", scale_span, scale_dtype))
        else:
            outputs.append(("fp8_scales", scale_span, scale_dtype, ()))
    require_separate(outputs, inputs)
    if rows == 0 or width == 0:
        return None
    device_id = runtime.current_device().device_id
    tensors = dict(operands, x=x, y=y)
    for name, span, *_ in inputs + outputs:
        require_device(name, tensors[name], span.pointer, device_id)
    strides = (source.stride, other.stride, residual_out_matrix.stride, modulation["gate"].stride,
               modulation["scale"].stride, modulation["shift"].stride, target.stride, fp8_stride)
    pointers = (source.pointer, other.pointer, residual_out_matrix.pointer, weights.pointer, branch_weights.pointer,
                modulation["gate"].pointer, modulation["scale"].pointer, modulation["shift"].pointer, target.pointer,
                fp8_pointer)
    return launch_for(device_id, options, strides, (rows, width, rows_per_sample)), pointers


@functools.lru_cache(maxsize=4096)
def launch_for(device_id, options, strides, sizes):
    """Returns the launch for operands that `plan` accepted, where `sizes` is `(rows, width, rows_per_sample)`."""
    rows, width, _ = sizes
    engine = engine_for(device_id, options)
    entry = size_for(options, width, engine, preferred_chunks(device_id, options, width))
    kernel, _ = runtime.load_for(device_id, entry.kernel)
    bulk = bulk_geometry(device_id, options, kernel, entry, rows, width) if engine == "bulk" else None
    if bulk is None and engine == "bulk":
        entry = size_for(options, width, chunks=preferred_chunks(device_id, options, width))
        kernel, _ = runtime.load_for(device_id, entry.kernel)
    if bulk is not None:
        grid, threads, shared_bytes, stages = bulk
        return Launch(runtime.kernel_call(kernel, SIGNATURE + "u"), entry.kernel, grid, threads, shared_bytes,
                      strides, sizes, (stages,))
    grid, threads, shared_bytes = geometry(device_id, options, kernel, entry, rows, width)
    return Launch(runtime.kernel_call(kernel, SIGNATURE), entry.kernel, grid, threads, shared_bytes, strides, sizes)


def rms_norm(options, x, y, *, epsilon, residual=None, branch=None, residual_out=None, weight=None,
             branch_weight=None, gate=None, scale=None, shift=None, fp8_scales=None, rows_per_sample=1,
             stream=None):
    """Launches the RMS norm instance that the `Options` value `options` selects on the rows of `x`, on `stream`.
    `y` receives the result, and `fp8_scales` holds the scale that a tensorwise FP8 output reads or the scales that
    the other FP8 outputs write. The README sections "Launchers", "RMS norm" and "Scalar arguments" describe the
    operands and the errors. A missing operand of `options` raises TypeError, and an unused operand raises
    ValueError."""
    epsilon = fp32("epsilon", epsilon)
    handle = runtime.stream_handle(stream)
    planned = plan(options, x, y, residual=residual, branch=branch, residual_out=residual_out, weight=weight,
                   branch_weight=branch_weight, gate=gate, scale=scale, shift=shift, fp8_scales=fp8_scales,
                   rows_per_sample=rows_per_sample, stream=handle)
    if planned is not None:
        launch, pointers = planned
        launch.launch(pointers, epsilon, handle)


class Launch:
    """The launch of one kernel for fixed sizes and strides. The method `launch` takes the pointers of each call."""

    def __init__(self, kernel_call, kernel, grid, threads, shared_bytes, strides, sizes, extra=()):
        self.extra = extra
        self.kernel_call = kernel_call
        self.kernel = kernel
        self.grid = grid
        self.threads = threads
        self.shared_bytes = shared_bytes
        self.strides = strides
        self.sizes = sizes

    def launch(self, pointers, epsilon, stream):
        x, other, residual_out, weight, branch_weight, gate, scale, shift, y, fp8 = pointers
        s = self.strides
        rows, width, rows_per_sample = self.sizes
        values = (x, s[0], other, s[1], residual_out, s[2], weight, branch_weight, gate, s[3], scale, s[4], shift, s[5],
                  y, s[6], fp8, s[7], rows, width, rows_per_sample, epsilon, *self.extra)
        self.kernel_call(self.grid, self.threads, self.shared_bytes, values, stream)


def row_layout(chunks, chunks_per_thread):
    threads = -(-chunks // chunks_per_thread)
    if threads <= WARP_THREADS:
        threads_per_row = 1 << (threads - 1).bit_length()
    else:
        threads_per_row = -(-threads // WARP_THREADS) * WARP_THREADS
    return threads_per_row, max(1, WORK_GROUP_THREADS // threads_per_row)


def input_bytes_per_row(options, width):
    x_bytes = DTYPE_BYTES[X_DTYPES[options.x]]
    addend_bytes = x_bytes if options.add == "residual" else 2 * (options.add != "none")
    return width * (x_bytes + addend_bytes)


def output_bytes_per_row(options, width):
    x_bytes = DTYPE_BYTES[X_DTYPES[options.x]] if options.add != "none" else 0
    return width * (DTYPE_BYTES[OUTPUT_DTYPES[options.output]] + x_bytes)


def in_flight_bytes(options, work_group_bytes):
    if options.add == "none" and work_group_bytes <= 4 * KIB:
        if options.blockwise:
            return 16 * KIB
        return 8 * KIB if options.fp8 or options.modulation != "none" else 4 * KIB
    return 8 * KIB if options.add != "none" and options.fp8 else 16 * KIB


def fits_scale_groups(options, width, entry):
    """A blockwise FP8 output needs a multiple of the work items of a scale group on each row."""
    threads_per_row, _ = row_layout(width // VALUES_PER_CHUNK, entry.chunks_per_thread)
    return not options.blockwise or threads_per_row % (SCALE_GROUP // VALUES_PER_CHUNK) == 0


def in_flight_limit(device_id, options, work_group_bytes):
    """Returns the bytes of loads in flight per SM that a launch keeps at most, or None for no limit. The limit comes
    from measurements on an RTX 6000 Ada (compute capability 8.9). On the H100, the B300 and the RTX PRO 6000
    (compute capability 9.0, 10.3 and 12.0), every variant ran fastest without it."""
    if runtime.compute_capability(device_id) >= UNLIMITED_CAPABILITY:
        return None
    return in_flight_bytes(options, work_group_bytes)


@functools.lru_cache(maxsize=4096)
def geometry(device_id, options, kernel, entry, rows, width):
    """Returns the grid, the work group shape and the dynamic shared memory of a launch. When the tensors exceed the L2
    cache, the shared memory limits each streaming multiprocessor to the fewest work groups that keep `in_flight_bytes`
    bytes of loads in flight."""
    threads_per_row, rows_per_work_group = row_layout(width // VALUES_PER_CHUNK, entry.chunks_per_thread)
    assert threads_per_row % WARP_THREADS == 0 or WARP_THREADS % threads_per_row == 0
    assert not options.blockwise or threads_per_row % (SCALE_GROUP // VALUES_PER_CHUNK) == 0
    threads = threads_per_row * rows_per_work_group
    work_groups = -(-rows // rows_per_work_group)
    require(work_groups <= MAX_GRID_X, f"{rows} rows need more than 2^31 - 1 work groups")
    work_group_bytes = rows_per_work_group * input_bytes_per_row(options, width)
    limit = in_flight_limit(device_id, options, work_group_bytes)
    shared_bytes = 0
    if limit is not None:
        cap = -(-limit // work_group_bytes)
        resident = runtime.resident_work_groups(kernel, threads, 0)
        tensor_bytes = rows * (input_bytes_per_row(options, width) + output_bytes_per_row(options, width))
        if cap < resident and tensor_bytes > runtime.l2_bytes(device_id):
            shared_bytes = runtime.shared_bytes_for(kernel, device_id, cap)
    return (work_groups, 1, 1), (threads_per_row, rows_per_work_group, 1), shared_bytes


@functools.lru_cache(maxsize=4096)
def bulk_geometry(device_id, options, kernel, entry, rows, width):
    """Returns the grid, the work group shape, the dynamic shared memory and the stage count of a bulk launch, or None
    when the shared memory of a work group holds fewer than `MIN_STAGES` stages of one row. A work group takes tiles
    of `rows_per_work_group` rows, and each stage holds the input rows of one tile."""
    threads_per_row, _ = row_layout(width // VALUES_PER_CHUNK, entry.chunks_per_thread)
    assert not options.blockwise or threads_per_row % (SCALE_GROUP // VALUES_PER_CHUNK) == 0
    per_sm = tuning.bulk_work_groups_per_sm
    budget = min(runtime.allow_dynamic_shared(kernel, device_id), runtime.shared_bytes_for(kernel, device_id, per_sm))
    row_bytes = input_bytes_per_row(options, width)
    tile_rows = max(1, min(tuning.bulk_threads, entry.threads) // threads_per_row)
    tile_rows = min(tile_rows, max(1, budget // (MIN_STAGES * row_bytes)))
    stage_bytes = tile_rows * row_bytes
    stages = tuning.bulk_stages or min(MAX_STAGES, budget // stage_bytes)
    if stages < MIN_STAGES or stages * stage_bytes > budget:
        return None
    threads = threads_per_row * tile_rows
    shared_bytes = stages * stage_bytes
    resident = min(per_sm, runtime.resident_work_groups(kernel, threads, shared_bytes))
    if resident < 1:
        return None
    tiles = -(-rows // tile_rows)
    grid = min(tiles, runtime.num_sm(device_id) * resident)
    return (grid, 1, 1), (threads_per_row, tile_rows, 1), shared_bytes, stages
