"""Launchers of the BF16 GEMM kernels (README section "Launchers")."""

import functools
import math
import threading
from collections import namedtuple

from cuda.bindings import driver
from cuda.core import Device

from . import runtime
from .tensors import require, require_device, require_dtype, view_of

ACTIVATIONS = ("none", "silu", "gelu_tanh", "gelu")
SIGNATURES = {"store_bf16": "p" * 7 + "u" * 11, "store_f32": "p" * 6 + "u" * 9, "gated": "p" * 9 + "u" * 11}
KIND_NAMES = {"store_bf16": "a BF16 output", "store_f32": "an FP32 output", "gated": "gated_gemm"}
CHUNK_BYTES = 16
MAX_INDEX = (1 << 31) - 1
# The constants of `modeled_seconds`. A streaming multiprocessor (SM) reaches `TENSOR_EFFICIENCY` of its tensor-core
# rate with at least `SATURATING_WARPS` warps. Under the power limit, all SMs sustain `SUSTAINED_FRACTION` of the
# tensor-core rate at the peak clock divided by `relative_energy`, and the memory traffic adds `MEMORY_POWER_SHARE` of
# the memory time to that bound. An SM copies tiles from the L2 cache at `SM_L2_BYTES_PER_CLOCK`. A split product
# writes FP32 partial sums to the L2 cache, and the last work group of a tile reads them at `SM_L2_BYTES_PER_CLOCK`
# and spends `SPLIT_SECONDS` more.
TENSOR_FLOPS_PER_CLOCK = {80: 2048}
DEFAULT_TENSOR_FLOPS_PER_CLOCK = 1024
L2_BYTES_PER_CLOCK = 2048
SM_L2_BYTES_PER_CLOCK = 43.8
TENSOR_EFFICIENCY = 0.674
MEMORY_EFFICIENCY = 0.75
SATURATING_WARPS = 6.54
SUSTAINED_FRACTION = 0.53
L2_ENERGY = 22.3
SHARED_ENERGY = 0.586
SMOOTH_MAX_EXPONENT = 13.6
SPLIT_SECONDS = 4.28e-7
MEMORY_POWER_SHARE = 0.372
SPLIT_CHOICES = (1, 2, 4, 8)
MIN_SPLIT_K_TILES = 8
L2_BUDGET_FRACTION = 0.75
PARTIAL_BYTES = 4
COUNTER_BYTES = 4
WORKSPACES_PER_DEVICE = 16
SQUARE_TILE = (192, 192)
SQUARE_MIN_WAVES = 8
GATED_SQUARE_MIN_ROWS = 4096

Matrix = namedtuple("Matrix", "pointer rows width stride itemsize")
Plan = namedtuple("Plan", "info tiles_m tiles_n panel by_columns splits")


def rows_of(name, view):
    require(len(view.shape) >= 1, f"{name} must have at least one dimension")
    width = view.shape[-1]
    if 0 in view.shape:
        return math.prod(view.shape[:-1]), width, 0
    require(width == 1 or view.strides[-1] == 1, f"the last dimension of {name} must be contiguous")
    rows, stride = 1, 0
    for size, step in zip(reversed(view.shape[:-1]), reversed(view.strides[:-1])):
        if size == 1:
            continue
        require(rows == 1 or step == stride * rows, f"the leading dimensions of {name} must flatten into rows of one "
                                                    f"stride")
        stride = stride if rows > 1 else step
        rows *= size
    return rows, width, stride


def matrix_of(name, tensor, dtypes, device_id, stream):
    view = view_of(tensor, stream, name)
    dtype = require_dtype(view, name, *dtypes)
    rows, width, stride = rows_of(name, view)
    require(width % 8 == 0, f"the rows of {name} must hold a multiple of 8 values, got {width}")
    require(max(rows, width, stride) <= MAX_INDEX and stride >= 0,
            f"the row count, the row length and the row stride of {name} must lie in 0 ... 2^31 - 1")
    if rows and width:
        require_device(name, tensor, view.pointer, device_id)
        require(view.pointer % CHUNK_BYTES == 0, f"{name} must start at a multiple of {CHUNK_BYTES} bytes")
        require(stride * view.itemsize % CHUNK_BYTES == 0,
                f"the row stride of {name} must be a multiple of {CHUNK_BYTES} bytes")
    return Matrix(view.pointer, rows, width, stride, view.itemsize), dtype


def output_of(name, tensor, dtypes, device_id, stream, rows, width):
    matrix, dtype = matrix_of(name, tensor, dtypes, device_id, stream)
    require((matrix.rows, matrix.width) == (rows, width), f"{name} must have {rows} rows of {width} values")
    require(rows <= 1 or matrix.stride >= width, f"the rows of {name} must not overlap")
    return matrix, dtype


def input_of(name, tensor, device_id, stream, rows=None, width=None):
    matrix, _ = matrix_of(name, tensor, ("bfloat16",), device_id, stream)
    if rows is not None:
        require((matrix.rows, matrix.width) == (rows, width), f"{name} must have {rows} rows of {width} values")
    return matrix


def vector_of(name, tensor, length, device_id, stream):
    if tensor is None:
        return None
    view = view_of(tensor, stream, name)
    require_dtype(view, name, "bfloat16")
    require(view.shape == (length,), f"{name} must have the shape ({length},), got {view.shape}")
    require(view.strides == (1,) or length <= 1, f"{name} must be contiguous")
    if length:
        require_device(name, tensor, view.pointer, device_id)
        require(view.pointer % CHUNK_BYTES == 0, f"{name} must start at a multiple of {CHUNK_BYTES} bytes")
    return Matrix(view.pointer, 1, length, 0, view.itemsize)


def pointer_of(matrix):
    return 0 if matrix is None else matrix.pointer


def byte_range(matrix):
    if matrix is None or not matrix.rows or not matrix.width:
        return None
    return matrix.pointer, matrix.pointer + ((matrix.rows - 1) * matrix.stride + matrix.width) * matrix.itemsize


def overlaps(first, second):
    first_range, second_range = byte_range(first), byte_range(second)
    return first_range is not None and second_range is not None and \
        first_range[0] < second_range[1] and second_range[0] < first_range[1]


def require_separate(out, inputs, residual):
    """`residual` may be `out`, because each work item reads its residual chunks before it writes the same chunks."""
    for name, matrix in inputs.items():
        require(not overlaps(out, matrix), f"out overlaps {name}")
    require(residual == out or not overlaps(out, residual), "out overlaps residual without being residual")


def activation_index(activation):
    require(activation in ACTIVATIONS, f"activation must be one of {', '.join(ACTIVATIONS)}, got {activation!r}")
    return ACTIVATIONS.index(activation)


@functools.cache
def tilings(kind):
    _, entries = runtime.manifest()
    prefix = f"gemm_{kind}_"
    built = [entry for name, entry in entries.items() if name.startswith(prefix)]
    return tuple(sorted(built, key=is_few_rows))


def is_few_rows(info):
    return info.warp_m == 0


def tiling_name(info):
    return f"m{info.tile_m}" if is_few_rows(info) else f"{info.tile_m}x{info.tile_n}x{info.tile_k}"


def plan_name(product_plan):
    if is_few_rows(product_plan.info):
        return tiling_name(product_plan.info)
    order = "c" if product_plan.by_columns else "r"
    split = f" split {product_plan.splits}" if product_plan.splits > 1 else ""
    return f"{tiling_name(product_plan.info)} panel {product_plan.panel}{order}{split}"


def block_columns(kind, info):
    return info.tile_n // 2 if kind == "gated" else info.tile_n


@functools.cache
def resident_work_groups(device_id, info):
    kernel, _ = runtime.load_for(device_id, info.kernel)
    return max(1, kernel.occupancy.max_active_blocks_per_multiprocessor(info.threads, info.shared_bytes))


@functools.cache
def device_rates(device_id):
    properties = Device(device_id).properties
    clock_hz = properties.clock_rate * 1e3
    flops_per_clock = TENSOR_FLOPS_PER_CLOCK.get(runtime.compute_capability(device_id), DEFAULT_TENSOR_FLOPS_PER_CLOCK)
    memory = 2 * properties.memory_clock_rate * 1e3 * properties.global_memory_bus_width / 8
    return flops_per_clock * clock_hz, L2_BYTES_PER_CLOCK * clock_hz, SM_L2_BYTES_PER_CLOCK * clock_hz, memory


def smooth_max(*times):
    return sum(time**SMOOTH_MAX_EXPONENT for time in times) ** (1 / SMOOTH_MAX_EXPONENT)


def relative_energy(info):
    l2_bytes = 2 * (info.tile_m + info.tile_n) / (info.tile_m * info.tile_n)
    shared_bytes = 2 * (info.warp_m + info.warp_n) / (info.warp_m * info.warp_n)
    return 1 + L2_ENERGY * l2_bytes + SHARED_ENERGY * shared_bytes


def k_tiles_of(info, k, splits):
    all_k_tiles = max(1, -(-k // info.tile_k))
    return -(-all_k_tiles // splits)


def memory_seconds(device_id, kind, m, columns, k, output_bytes):
    *_, memory_rate = device_rates(device_id)
    b_rows = 2 * columns if kind == "gated" else columns
    return (2 * (m + b_rows) * k + output_bytes * m * columns) / (memory_rate * MEMORY_EFFICIENCY)


def live_fraction(info, m):
    tiles_m = max(1, -(-m // info.tile_m))
    fragments = info.tile_m // 16
    last = -(-(m - (tiles_m - 1) * info.tile_m) // 16)
    return ((tiles_m - 1) * fragments + last) / (tiles_m * fragments)


def modeled_seconds(device_id, kind, info, m, columns, k, output_bytes, splits):
    """Returns the smooth maximum of the time of the busiest SM, the time that the power limit allows and
    the memory time, plus the time of the split sums."""
    flops_rate, l2_rate, sm_l2_rate, _ = device_rates(device_id)
    tiles = -(-m // info.tile_m) * -(-columns // block_columns(kind, info))
    work_groups = tiles * splits
    work_group_k = k_tiles_of(info, k, splits) * info.tile_k
    work_group_flops = 2 * info.tile_m * info.tile_n * work_group_k
    work_group_seconds = work_group_flops / (flops_rate * TENSOR_EFFICIENCY)
    work_group_load_seconds = 2 * (info.tile_m + info.tile_n) * work_group_k / sm_l2_rate
    resident = resident_work_groups(device_id, info)
    warps = info.threads // 32

    def round_seconds(work_groups):
        if not work_groups:
            return 0
        compute = work_groups * work_group_seconds / min(1, work_groups * warps / SATURATING_WARPS)
        return smooth_max(compute, work_groups * work_group_load_seconds)

    sms = runtime.num_sm(device_id)
    full_rounds, last = divmod(-(-work_groups // sms), resident)
    busiest = full_rounds * round_seconds(resident) + round_seconds(last)
    power = (live_fraction(info, m) * work_groups * work_group_flops * relative_energy(info) /
             (sms * flops_rate * SUSTAINED_FRACTION))
    memory = memory_seconds(device_id, kind, m, columns, k, output_bytes)
    seconds = smooth_max(busiest, power + MEMORY_POWER_SHARE * memory, memory)
    if splits > 1:
        tile_bytes = info.tile_m * info.tile_n * PARTIAL_BYTES
        seconds += SPLIT_SECONDS + work_groups * tile_bytes / l2_rate + splits * tile_bytes / sm_l2_rate
    return seconds


def panel_of(tiles, other_tiles, limit, wave):
    most = min(max(limit, 1), tiles)
    if most == tiles:
        return tiles
    for panel in range(most, most // 2, -1):
        if panel * other_tiles % wave == 0:
            return panel
    panels = -(-tiles // most)
    return -(-tiles // panels)


def raster_of(device_id, info, tiles_m, tiles_n, k):
    budget = int(runtime.l2_bytes(device_id) * L2_BUDGET_FRACTION)
    wave = runtime.num_sm(device_id) * resident_work_groups(device_id, info)
    a_panel_bytes = info.tile_m * max(k, 1) * 2
    b_panel_bytes = info.tile_n * max(k, 1) * 2
    rows = panel_of(tiles_m, tiles_n, budget // a_panel_bytes, wave)
    columns = panel_of(tiles_n, tiles_m, budget // b_panel_bytes, wave)
    by_rows = tiles_m * a_panel_bytes + tiles_n * b_panel_bytes * -(-tiles_m // rows)
    by_columns = tiles_m * a_panel_bytes * -(-tiles_n // columns) + tiles_n * b_panel_bytes
    return (rows, 0) if by_rows <= by_columns else (columns, 1)


def allows_square_tiling(device_id, kind, info, m, columns):
    if (info.tile_m, info.tile_n) != SQUARE_TILE:
        return True
    if kind == "gated" and m < GATED_SQUARE_MIN_ROWS:
        return False
    tiles = -(-m // info.tile_m) * -(-columns // block_columns(kind, info))
    return tiles >= SQUARE_MIN_WAVES * runtime.num_sm(device_id) * resident_work_groups(device_id, info)


def split_choices(info, k, splits):
    if splits is not None:
        return (splits,) if splits <= info.max_splits else ()
    all_k_tiles = max(1, -(-k // info.tile_k))
    return tuple(s for s in SPLIT_CHOICES if s <= info.max_splits and (s == 1 or all_k_tiles // s >= MIN_SPLIT_K_TILES))


def checked_tilings(kind, tiling, raster, splits):
    built = tilings(kind)
    if not built:
        raise RuntimeError(f"no {kind} GEMM kernel is built")
    if tiling is not None:
        names = [tiling_name(info) for info in built]
        require(tiling in names, f"tiling must be one of {', '.join(names)} for {KIND_NAMES[kind]}, got {tiling!r}")
        built = tuple(info for info in built if tiling_name(info) == tiling)
    if raster is not None:
        require(isinstance(raster, tuple) and len(raster) == 2 and all(isinstance(value, int) for value in raster) and
                raster[0] >= 1 and raster[1] in (0, 1),
                f"raster must be a pair (panel, order) of a panel >= 1 and an order 0 or 1, got {raster!r}")
    if splits is None:
        return built
    require(isinstance(splits, int) and splits >= 1, f"splits must be a positive int, got {splits!r}")
    if tiling is not None and is_few_rows(built[0]):
        require(splits == 1, f"the tiling {tiling} takes no splits of k, because each of its work groups adds all of "
                             f"k, got splits={splits}")
    most = max(info.max_splits for info in built)
    require(splits <= most, f"splits must lie in 1 ... {most}, the most that a built kernel takes, got {splits}")
    return built


@functools.lru_cache(maxsize=4096)
def plan(device_id, kind, m, columns, k, tiling=None, raster=None, splits=None):
    """Returns the plan of a product of `m` rows and `columns` output columns. A `tiling`, `raster` or `splits` that the
    caller passes overrides the modeled choice. A few-rows kernel serves every product of at most its rows, unless the
    caller passes `splits` without `tiling`."""
    built = checked_tilings(kind, tiling, raster, splits)
    candidates = [info for info in built if runtime.is_available(info.kernel, device_id)]
    if not candidates:
        raise RuntimeError(runtime.unavailable_reason(built[0].kernel, device_id))
    few_rows = [info for info in candidates if is_few_rows(info)]
    if tiling is not None and few_rows:
        require(m <= few_rows[0].tile_m, f"the tiling {tiling} takes at most {few_rows[0].tile_m} rows of a, got {m}")
    fitting = [info for info in few_rows if m <= info.tile_m]
    if fitting and (tiling is not None or splits is None):
        info = min(fitting, key=lambda info: info.tile_m)
        return Plan(info, 1, -(-columns // block_columns(kind, info)), 1, 0, 1)
    eligible = [info for info in candidates if not is_few_rows(info) and
                (tiling is not None or allows_square_tiling(device_id, kind, info, m, columns))]
    choices = [(info, s) for info in eligible for s in split_choices(info, k, splits)]
    if not choices:
        tiled = [info for info in built if not is_few_rows(info)]
        raise RuntimeError(runtime.unavailable_reason(tiled[0].kernel, device_id) if tiled else
                           f"no tiled {kind} GEMM kernel is built")
    output_bytes = 4 if kind == "store_f32" else 2
    info, splits = min(choices, key=lambda choice: modeled_seconds(device_id, kind, choice[0], m, columns, k,
                                                                    output_bytes, choice[1]))
    tiles_m = -(-m // info.tile_m)
    tiles_n = -(-columns // block_columns(kind, info))
    require(tiles_m * tiles_n * splits <= MAX_INDEX, "the product needs more than 2^31 - 1 work groups")
    panel, by_columns = raster or raster_of(device_id, info, tiles_m, tiles_n, k)
    return Plan(info, tiles_m, tiles_n, min(panel, tiles_n if by_columns else tiles_m), by_columns, splits)


class Launch:
    def __init__(self, kind, product_plan, sizes, activation, device_id):
        kernel, info = runtime.load_for(device_id, product_plan.info.kernel)
        self.run = runtime.kernel_call(kernel, SIGNATURES[kind])
        tiles = product_plan.tiles_m * product_plan.tiles_n
        self.grid = (tiles * product_plan.splits, 1, 1)
        self.threads = info.threads
        self.shared_bytes = info.shared_bytes
        self.sizes = (*sizes, product_plan.panel, product_plan.by_columns, product_plan.splits, *activation)
        split = product_plan.splits > 1
        self.partial_bytes = tiles * product_plan.splits * info.tile_m * info.tile_n * PARTIAL_BYTES if split else 0
        self.counter_count = tiles if split else 0

    def __call__(self, pointers, buffers, stream):
        self.run(self.grid, self.threads, self.shared_bytes, (*pointers, *buffers, *self.sizes), stream)


NO_BUFFERS = (0, 0)


def checked(result):
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {status}")
    return values[0] if values else None


def is_capturing(stream):
    return checked(driver.cuStreamIsCapturing(stream)) != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE


class Workspace:
    """The partial sums and counters of the split products that one stream runs outside a CUDA graph capture. A thread
    holds `lock` until it has queued its launch, so that a larger allocation frees the old buffer, in stream order,
    after every launch that reads the old buffer. The event `used` follows the last launch that used the buffers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.parts = None
        self.counters = None
        self.used = None

    def addresses(self, device, stream, part_bytes, counter_count):
        if self.parts is None or self.parts.size < part_bytes:
            size = max(part_bytes, 2 * (self.parts.size if self.parts else 0))
            if self.parts is not None:
                self.parts.close(stream)
            self.parts = device.allocate(size, stream=stream)
        counter_bytes = COUNTER_BYTES * counter_count
        if self.counters is None or self.counters.size < counter_bytes:
            size = max(counter_bytes, 2 * (self.counters.size if self.counters else 0))
            if self.counters is not None:
                self.counters.close(stream)
            self.counters = device.allocate(size, stream=stream)
            self.counters.fill(0, stream=stream)
        return int(self.parts.handle), int(self.counters.handle)

    def release(self, stream):
        """Frees the buffers on `stream` after the last launch that used them. The event `used` marks that launch even
        when its stream no longer exists."""
        if self.used is not None:
            stream.wait(self.used)
        for buffer in (self.parts, self.counters):
            if buffer is not None:
                buffer.close(stream)
        self.parts = self.counters = None


workspaces = {}
workspaces_lock = threading.Lock()


def workspace_of(device_id, handle, stream):
    """A new workspace on a device with `WORKSPACES_PER_DEVICE` workspaces frees the oldest one on `stream`."""
    key = device_id, handle
    with workspaces_lock:
        workspace = workspaces.get(key)
        if workspace is not None:
            return workspace
        same_device = [other for other in workspaces if other[0] == device_id]
        oldest = workspaces.pop(same_device[0]) if len(same_device) >= WORKSPACES_PER_DEVICE else None
        workspace = workspaces[key] = Workspace()
    if oldest is not None:
        with oldest.lock:
            oldest.release(stream)
    return workspace


def run(launch, pointers, stream):
    """A split product in a CUDA graph capture allocates, zeroes and frees its partial sums and counters in the graph.
    Other split products use the workspace of their device and stream."""
    handle = runtime.stream_handle(stream)
    if not launch.partial_bytes:
        launch(pointers, NO_BUFFERS, handle)
        return
    if is_capturing(handle):
        address = int(checked(driver.cuMemAllocAsync(launch.partial_bytes + COUNTER_BYTES * launch.counter_count,
                                                     handle)))
        try:
            counters = address + launch.partial_bytes
            checked(driver.cuMemsetD32Async(counters, 0, launch.counter_count, handle))
            launch(pointers, (address, counters), handle)
        finally:
            checked(driver.cuMemFreeAsync(address, handle))
        return
    device, launch_stream = runtime.current_device(), runtime.stream_of(handle)
    workspace = workspace_of(device.device_id, handle, launch_stream)
    with workspace.lock:
        buffers = workspace.addresses(device, launch_stream, launch.partial_bytes, launch.counter_count)
        launch(pointers, buffers, handle)
        workspace.used = launch_stream.record(workspace.used)


def plan_arguments(tiling, raster, splits):
    return tiling, tuple(raster) if isinstance(raster, (tuple, list)) else raster, splits


def store_launch(a, b, out, *, bias=None, residual=None, activation="none", tiling=None, raster=None, splits=None,
                 stream=None):
    device_id = runtime.current_device().device_id
    a_matrix = input_of("a", a, device_id, stream)
    b_matrix = input_of("b", b, device_id, stream)
    m, k = a_matrix.rows, a_matrix.width
    n = b_matrix.rows
    require(b_matrix.width == k, f"b must have rows of {k} values, got {b_matrix.width}")
    require(n % 8 == 0, f"b must have a multiple of 8 rows, got {n}")
    out_matrix, out_dtype = output_of("out", out, ("bfloat16", "float32"), device_id, stream, m, n)
    kind = "store_bf16" if out_dtype == "bfloat16" else "store_f32"
    bias_vector = vector_of("bias", bias, n, device_id, stream)
    index = activation_index(activation)
    residual_matrix = None if residual is None else input_of("residual", residual, device_id, stream, m, n)
    if kind == "store_f32":
        require(index == 0 and residual is None, "an FP32 output takes no activation and no residual")
    require_separate(out_matrix, {"a": a_matrix, "b": b_matrix, "bias": bias_vector}, residual_matrix)
    if m == 0 or n == 0:
        checked_tilings(kind, *plan_arguments(tiling, raster, splits))
        return None
    product_plan = plan(device_id, kind, m, n, k, *plan_arguments(tiling, raster, splits))
    pointers = (a_matrix.pointer, b_matrix.pointer, out_matrix.pointer, pointer_of(bias_vector))
    if kind == "store_f32":
        sizes = (a_matrix.stride, b_matrix.stride, out_matrix.stride, m, n, k)
        return Launch(kind, product_plan, sizes, (), device_id), pointers
    residual_stride = 0 if residual_matrix is None else residual_matrix.stride
    sizes = (a_matrix.stride, b_matrix.stride, out_matrix.stride, residual_stride, m, n, k)
    return Launch(kind, product_plan, sizes, (index,), device_id), (*pointers, pointer_of(residual_matrix))


def gemm(a, b, out, *, bias=None, residual=None, activation="none", tiling=None, raster=None, splits=None,
         stream=None):
    """Queues the product of `zig_cu_kernels.torch.gemm` into a BF16 or FP32 `out` on `stream`. `tiling`, `raster`
    and `splits` override the plan (README sections "BF16 GEMM", "Launchers", "Work groups, tiles and splits" and
    "Determinism")."""
    launched = store_launch(a, b, out, bias=bias, residual=residual, activation=activation, tiling=tiling,
                            raster=raster, splits=splits, stream=stream)
    if launched is not None:
        run(*launched, stream)


def gated_launch(a, w_gate, w_up, out, *, bias_gate=None, bias_up=None, residual=None, activation="silu",
                 tiling=None, raster=None, splits=None, stream=None):
    device_id = runtime.current_device().device_id
    a_matrix = input_of("a", a, device_id, stream)
    gate_matrix = input_of("w_gate", w_gate, device_id, stream)
    m, k = a_matrix.rows, a_matrix.width
    f = gate_matrix.rows
    require(gate_matrix.width == k, f"w_gate must have rows of {k} values, got {gate_matrix.width}")
    up_matrix = input_of("w_up", w_up, device_id, stream, f, k)
    require(f % 8 == 0, f"w_gate and w_up must have a multiple of 8 rows, got {f}")
    require(gate_matrix.stride == up_matrix.stride, "w_gate and w_up must have the same row stride")
    out_matrix, _ = output_of("out", out, ("bfloat16",), device_id, stream, m, f)
    biases = tuple(vector_of(name, bias, f, device_id, stream) for name, bias in (("bias_gate", bias_gate),
                                                                                  ("bias_up", bias_up)))
    index = activation_index(activation)
    residual_matrix = None if residual is None else input_of("residual", residual, device_id, stream, m, f)
    inputs = {"a": a_matrix, "w_gate": gate_matrix, "w_up": up_matrix, "bias_gate": biases[0], "bias_up": biases[1]}
    require_separate(out_matrix, inputs, residual_matrix)
    if m == 0 or f == 0:
        checked_tilings("gated", *plan_arguments(tiling, raster, splits))
        return None
    product_plan = plan(device_id, "gated", m, f, k, *plan_arguments(tiling, raster, splits))
    pointers = (a_matrix.pointer, gate_matrix.pointer, up_matrix.pointer, out_matrix.pointer, *map(pointer_of, biases),
                pointer_of(residual_matrix))
    residual_stride = 0 if residual_matrix is None else residual_matrix.stride
    sizes = (a_matrix.stride, gate_matrix.stride, out_matrix.stride, residual_stride, m, f, k)
    return Launch("gated", product_plan, sizes, (index,), device_id), pointers


def gated_gemm(a, w_gate, w_up, out, *, bias_gate=None, bias_up=None, residual=None, activation="silu", tiling=None,
               raster=None, splits=None, stream=None):
    """Queues the gated linear unit of `zig_cu_kernels.torch.gated_gemm` into `out`. `w_gate` and `w_up` must share
    one row stride, and the other rules are those of `gemm`."""
    launched = gated_launch(a, w_gate, w_up, out, bias_gate=bias_gate, bias_up=bias_up, residual=residual,
                            activation=activation, tiling=tiling, raster=raster, splits=splits, stream=stream)
    if launched is not None:
        run(*launched, stream)
