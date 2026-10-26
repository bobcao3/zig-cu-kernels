"""Launchers of the FP8 GEMM and quantizer kernels (README section "Launchers")."""

import functools
import math
from collections import namedtuple

from cuda.core import Device

from . import runtime
from .tensors import is_aligned, is_contiguous, require, require_device, require_dtype, rows_view, view_of

SCALE_GROUP = 128
ACTIVATIONS = {"none": 0, "silu": 1, "gelu_tanh": 2, "gelu": 3}
SIGNATURES = {"store": "p" * 9 + "u" * 16, "low_rank": "p" * 11 + "u" * 17, "gated": "p" * 12 + "u" * 18,
              "glu": "p" * 12 + "u" * 17 + "f" + "u" * 3}
GLU_ACTIVATIONS = ("swiglu", "geglu_tanh")
GLU_TILINGS = ("128x256",)
FAMILY_GLU_TILINGS = {"mxfp8": ("128x128",)}
RASTER_ORDERS = ("along_n", "along_m")
QUANTIZE_SIGNATURES = {"rows": "puppuuu", "tiles": "puppuuu", "tensorwise": "pupuupp", "amax": "puuup"}
ELEMENTS = {"bfloat16": "bf16", "float32": "f32"}
SCALE_SUFFIXES = {"float32": "", "float8_e8m0fnu": "_e8m0"}
SCALE_BYTES = 4
WORD_BYTES = 4
CODE_ALIGNMENT_BYTES = 8
ROW_ALIGNMENT_BYTES = 16
MAX_U32 = (1 << 32) - 1
MAX_WORK_GROUPS = (1 << 31) - 1
AMAX_GROUPS_PER_SM = 8
K_MULTIPLES = {"tensorwise": 16, "rowwise": 16, "block128": SCALE_GROUP}
N_MULTIPLE = 8
TILINGS = ("32x64", "64x128", "128x256", "128x128", "128x64")
FAMILY_TILINGS = {"mxfp8": ("32x64", "64x128", "128x128")}
COLUMNS_PER_TILE = {"store": lambda tile_n: tile_n, "low_rank": lambda tile_n: tile_n,
                    "gated": lambda tile_n: tile_n // 2, "glu": lambda tile_n: tile_n // 2}
STEP_SECONDS = {"128x256": 0.87e-6, "64x128": 1.45e-6, "32x64": 0.48e-6, "128x128": 0.87e-6, "128x64": 0.6e-6}
STEP_TILE_K = {"128x256": 128, "64x128": 256, "32x64": 256, "128x128": 128, "128x64": 256}
MAC_ENERGY = {"mxfp8": 1.5}
POWER_MAC_RATE = {"128x256": 114e12, "64x128": 75e12, "32x64": 53e12, "128x128": 99e12, "128x64": 72e12}
SM_STREAM_RATE = {"128x256": 29.7e9, "64x128": 40e9, "32x64": 40e9, "128x128": 15.4e9, "128x64": 40e9}
SM_HIT_RATE = 40e9
L2_STREAM_RATE = {"128x256": 1.54e12, "64x128": 2.2e12, "32x64": 2.2e12, "128x128": 1.8e12, "128x64": 2.25e12}
STREAM_SHARE = 0.2
MEMORY_EFFICIENCY = {"128x256": 0.66, "64x128": 0.79, "32x64": 0.82, "128x128": 0.48, "128x64": 0.8}
SM_L2_RATE = 0.1e12
EXCHANGE_SECONDS = 3e-6
FIXED_SECONDS = 3.5e-6
MAX_SHARE = 4


def scaling_type_of(lead, columns, k, scale_a_shape, weight_scale_shapes, e8m0):
    m = math.prod(lead)
    a_shape, weight_shapes = tuple(scale_a_shape), [tuple(shape) for shape in weight_scale_shapes]
    scale_groups = k // SCALE_GROUP
    block = a_shape in ((m, scale_groups), (*lead, scale_groups)) and all(
        shape == (-(-columns // SCALE_GROUP), scale_groups) for shape in weight_shapes)
    if e8m0 and block:
        return "block128"
    if math.prod(a_shape) == 1 and all(math.prod(shape) == 1 for shape in weight_shapes):
        return "tensorwise"
    if a_shape in ((m, 1), (*lead, 1)) and all(shape == (columns, 1) for shape in weight_shapes):
        return "rowwise"
    if block:
        return "block128"
    raise ValueError(f"the scale shapes {a_shape} and {', '.join(map(str, weight_shapes))} match no scaling mode for "
                     f"m = {m}, n = {columns} and k = {k}: tensorwise takes one-element scales, rowwise "
                     f"{(m, 1)} and {(columns, 1)}, block128 {(m, scale_groups)} and "
                     f"{(-(-columns // SCALE_GROUP), scale_groups)}")


def check_product(scaling_type, columns, k, e8m0, use_fast_accum):
    require(not e8m0 or scaling_type == "block128", "E8M0 scales need block128 scaling")
    require(k > 0 and k % K_MULTIPLES[scaling_type] == 0,
            f"{scaling_type} scaling needs a positive k % {K_MULTIPLES[scaling_type]} == 0")
    require(columns % N_MULTIPLE == 0, f"the weights need a row count that is a multiple of {N_MULTIPLE}")
    require(not (use_fast_accum and scaling_type == "block128"), "use_fast_accum does not take block128 scales")
    require(max(columns, k) <= MAX_U32, "the weights are too large")


def family_of(scaling_type, use_fast_accum):
    if scaling_type == "block128":
        return "block128"
    return "rowwise_fast" if use_fast_accum else "rowwise"


def require_devices(device_id, **tensors):
    for name, tensor in tensors.items():
        if tensor is not None:
            require_device(name, tensor, None, device_id)


def byte_range(view):
    if 0 in view.shape:
        return None
    low = sum(min(0, (size - 1) * stride) for size, stride in zip(view.shape, view.strides))
    high = sum(max(0, (size - 1) * stride) for size, stride in zip(view.shape, view.strides))
    return view.pointer + low * view.itemsize, view.pointer + (high + 1) * view.itemsize


def overlaps(first, second):
    first_range, second_range = byte_range(first), byte_range(second)
    if first_range is None or second_range is None:
        return False
    return first_range[0] < second_range[1] and second_range[0] < first_range[1]


def is_exact_alias(first, second):
    return (first.pointer, first.shape, first.strides, first.dtype) == (
        second.pointer, second.shape, second.strides, second.dtype)


def require_separate(output_name, output, aliases=(), **inputs):
    """Raises ValueError when the view `output` overlaps an input other than as an exact alias in `aliases`."""
    for name, view in inputs.items():
        if view is not None and overlaps(output, view) and not (name in aliases and is_exact_alias(output, view)):
            raise ValueError(f"{output_name} overlaps {name}")


def matrix(view, name, *dtypes):
    """Returns the row count, width and row stride of `view`; an empty view may have any address and row stride."""
    rows, width, stride = rows_view(view)
    require_dtype(view, name, *dtypes)
    require(max(rows, width) <= MAX_U32, f"the row count and the row length of {name} must lie below 2^32")
    if rows * width == 0:
        return rows, width, 0
    require(is_aligned(view), f"{name} must start at a multiple of 16 bytes")
    require(rows <= 1 or stride * view.itemsize % ROW_ALIGNMENT_BYTES == 0,
            f"the row stride of {name} must be a multiple of 16 bytes")
    require(stride <= MAX_U32, f"the row stride of {name} must lie below 2^32")
    return rows, width, stride


def output_matrix(view, name, rows, width):
    out_rows, out_width, stride = matrix(view, name, "bfloat16")
    require((out_rows, out_width) == (rows, width), f"{name} must have {rows} rows of {width} values")
    require(rows <= 1 or stride >= width, f"the elements of {name} share memory")
    return stride


def scale_strides(view, name, lead, scale_groups):
    """Returns the row stride and the scale group stride in elements of a scale view, with the stride 0 for a single
    row or a single scale group."""
    rows, row_stride = 1, 0
    for size, stride in zip(reversed(view.shape[:-1]), reversed(view.strides[:-1])):
        if size == 1:
            continue
        if rows == 1:
            row_stride = stride
        else:
            require(stride == row_stride * rows, f"{name} needs leading dimensions that flatten into one row stride")
        rows *= size
    scale_group_stride = view.strides[-1] if scale_groups > 1 else 0
    require(min(row_stride, scale_group_stride) >= 0 and max(row_stride, scale_group_stride) <= MAX_U32,
            f"{name} needs strides in [0, 2^32)")
    return row_stride, scale_group_stride


ScaleOperands = namedtuple("ScaleOperands", "scaling_type views strides e8m0")


def scale_operands(scale_a, weight_scales, lead, columns, k, use_fast_accum, stream):
    """Checks `scale_a` and the dict `weight_scales` and returns the scaling type, views and strides of the scales.
    Bit i of `e8m0` marks scale i as E8M0, where `scale_a` is scale 0 and the weight scales follow in dict order."""
    names = ("scale_a", *weight_scales)
    views = [view_of(scale, stream, name) for scale, name in zip((scale_a, *weight_scales.values()), names)]
    dtypes = [require_dtype(view, name, *SCALE_SUFFIXES) for view, name in zip(views, names)]
    e8m0 = sum(1 << index for index, dtype in enumerate(dtypes) if dtype == "float8_e8m0fnu")
    scaling_type = scaling_type_of(lead, columns, k, views[0].shape, [view.shape for view in views[1:]], e8m0 != 0)
    check_product(scaling_type, columns, k, e8m0 != 0, use_fast_accum)
    strides = []
    for index, (view, name) in enumerate(zip(views, names)):
        require(view.pointer % WORD_BYTES == 0, f"{name} must start at a multiple of 4 bytes")
        if scaling_type == "tensorwise":
            strides.append((0, 0))
        elif scaling_type == "rowwise":
            strides.append(scale_strides(view, name, lead if index == 0 else (columns,), 1))
        else:
            scale_lead = lead if index == 0 else (-(-columns // SCALE_GROUP),)
            strides.append(scale_strides(view, name, scale_lead, k // SCALE_GROUP))
    return ScaleOperands(scaling_type, views, strides, e8m0)


def raster(info, tiles_m, tiles_n, k, work_groups, device_id):
    """Returns the panel and raster order under which the work groups read the fewest bytes from memory. The count
    assumes that the L2 cache keeps a panel of A or B tiles that fits in three quarters of the cache."""
    budget = runtime.l2_bytes(device_id) // 4 * 3
    rows = panel_of(tiles_m, tiles_n, budget // (info.tile_m * k), work_groups)
    columns = panel_of(tiles_n, tiles_m, budget // (info.tile_n * k), work_groups)
    a_bytes = tiles_m * info.tile_m * k
    b_bytes = tiles_n * info.tile_n * k
    along_n_bytes = a_bytes + b_bytes * -(-tiles_m // rows)
    along_m_bytes = a_bytes * -(-tiles_n // columns) + b_bytes
    return (rows, "along_n") if along_n_bytes <= along_m_bytes else (columns, "along_m")


def panel_of(tiles, other_tiles, limit, work_groups):
    """Returns the size of a panel of at most `limit` tiles. The largest size above half the limit whose panel fills
    whole waves of `work_groups` work groups wins, and otherwise the panels split the tiles evenly."""
    most = min(max(limit, 1), tiles)
    if most == tiles:
        return tiles
    for panel in range(most, most // 2, -1):
        if panel * other_tiles % work_groups == 0:
            return panel
    panels = -(-tiles // most)
    return -(-tiles // panels)


Plan = namedtuple("Plan", "kernel info work_groups panel raster_order data_parallel_tiles part_bytes counters")
Choice = namedtuple("Choice", "tiling work_groups data_parallel_tiles")


@functools.cache
def memory_rate(device_id):
    """Returns the peak memory bandwidth of a device in bytes per second, with two transfers per memory clock."""
    properties = Device(device_id).properties
    return 2 * properties.memory_clock_rate * 1e3 * properties.global_memory_bus_width / 8


def estimate(info, columns_per_tile, rows, columns, b_rows, k, work_groups, data_parallel_tiles, memory_bytes_rate,
             mac_energy=1):
    """Returns the seconds that a performance model predicts for a launch that reads A and B from memory. The work
    groups first compute waves of whole tiles and then share the k tiles of the remaining tiles. `mac_energy` is the
    energy of one multiply-accumulate relative to the rowwise kernels, for which `POWER_MAC_RATE` holds."""
    tiles = -(-rows // info.tile_m) * -(-columns // columns_per_tile(info.tile_n))
    k_tiles = -(-k // info.tile_k)
    total_k_tiles = tiles * k_tiles
    stream_k_tiles = total_k_tiles - data_parallel_tiles * k_tiles
    tiling = f"{info.tile_m}x{info.tile_n}"
    step_bytes = (info.tile_m + info.tile_n) * info.tile_k
    memory_bytes = (rows + b_rows) * k
    misses = min(1, memory_bytes / (total_k_tiles * step_bytes))
    step_stream = step_bytes * (misses / SM_STREAM_RATE[tiling] + (1 - misses) / SM_HIT_RATE)
    step = max(STEP_SECONDS[tiling] * info.tile_k / STEP_TILE_K[tiling], step_stream)
    k_tile_memory = memory_bytes / (MEMORY_EFFICIENCY[tiling] * memory_bytes_rate) / total_k_tiles
    k_tile_seconds = max(k_tile_memory, step_bytes / L2_STREAM_RATE[tiling])
    waves, last = divmod(data_parallel_tiles, work_groups)
    phases = ((waves, work_groups * k_tiles, k_tiles), (1, last * k_tiles, k_tiles))
    phases += ((1, stream_k_tiles, -(-stream_k_tiles // work_groups)),)
    seconds = 0
    steps = 0
    for count, phase_k_tiles, phase_steps in phases:
        if phase_k_tiles:
            seconds += count * max(phase_steps * step, phase_k_tiles * k_tile_seconds)
            steps += count * phase_steps
    seconds = max(seconds, -(-rows // info.tile_m) * info.tile_m * b_rows * k * mac_energy / POWER_MAC_RATE[tiling])
    exchange = 0
    if stream_k_tiles:
        parts = -(-k_tiles // max(stream_k_tiles // work_groups, 1)) + 1
        exchange = EXCHANGE_SECONDS + parts * min(rows, info.tile_m) * info.tile_n * SCALE_BYTES / SM_L2_RATE
    return seconds + STREAM_SHARE * steps * step_stream + exchange + FIXED_SECONDS


def choices(info, columns_per_tile, rows, columns, k, sms, stream_k):
    """Yields the work group count and whole tile count of each candidate launch of one tiling. With `stream_k`, the
    candidates after the first spread all k tiles over 2 to `MAX_SHARE` work groups per tile, up to `sms` in all."""
    tiles = -(-rows // info.tile_m) * -(-columns // columns_per_tile(info.tile_n))
    total_k_tiles = tiles * -(-k // info.tile_k)
    yield min(tiles, sms), tiles
    if stream_k and total_k_tiles * (sms + 1) <= MAX_U32:
        for share in range(2, min(MAX_SHARE, sms // tiles) + 1):
            yield min(share * tiles, total_k_tiles), 0


def gemm_name(family, kind, tiling):
    if family == "mxfp8":
        operation = {"glu": "glu", "low_rank": "abt_low_rank"}.get(kind, "abt")
        return f"mxfp8_mm_{operation}_sm89_{tiling}"
    return f"fp8_gemm_{family}_{kind}_{tiling}"


@functools.lru_cache(maxsize=4096)
def choose(family, kind, rows, columns, k, sms, memory_bytes_rate, stream_k):
    best = None
    columns_per_tile = COLUMNS_PER_TILE[kind]
    b_rows = columns * (2 if kind in ("gated", "glu") else 1)
    glu_tilings = FAMILY_GLU_TILINGS.get(family, GLU_TILINGS)
    for tiling in glu_tilings if kind == "glu" else FAMILY_TILINGS.get(family, TILINGS):
        info = runtime.info(gemm_name(family, kind, tiling))
        for work_groups, data_parallel_tiles in choices(info, columns_per_tile, rows, columns, k, sms, stream_k):
            seconds = estimate(info, columns_per_tile, rows, columns, b_rows, k, work_groups, data_parallel_tiles,
                               memory_bytes_rate, MAC_ENERGY.get(family, 1))
            if best is None or seconds < best[0]:
                best = seconds, Choice(tiling, work_groups, data_parallel_tiles)
    return best[1]


def plan(family, kind, m, columns, k, device_id, stream_k):
    choice = choose(family, kind, m, columns, k, runtime.num_sm(device_id), memory_rate(device_id), stream_k)
    kernel, info = runtime.load_for(device_id, gemm_name(family, kind, choice.tiling))
    tiles_m = -(-m // info.tile_m)
    tiles_n = -(-columns // COLUMNS_PER_TILE[kind](info.tile_n))
    panel, raster_order = raster(info, tiles_m, tiles_n, k, choice.work_groups, device_id)
    shared = choice.data_parallel_tiles < tiles_m * tiles_n
    part_bytes = 2 * choice.work_groups * info.tile_m * info.tile_n * SCALE_BYTES if shared else 0
    counters = tiles_m * tiles_n - choice.data_parallel_tiles
    return Plan(kernel, info, choice.work_groups, panel, raster_order, choice.data_parallel_tiles, part_bytes, counters)


NO_BUFFERS = (0, 0)


class Launch:
    """The planned kernel launch of a product. A call takes the operand and workspace addresses and a stream handle."""

    def __init__(self, family, kind, m, columns, k, sizes, device_id, stream_k):
        product_plan = plan(family, kind, m, columns, k, device_id, stream_k)
        self.run = runtime.kernel_call(product_plan.kernel, SIGNATURES[kind])
        self.grid = (product_plan.work_groups, 1, 1)
        self.threads = product_plan.info.threads
        self.shared_bytes = product_plan.info.shared_bytes
        self.part_bytes = product_plan.part_bytes
        self.counters = product_plan.counters
        self.sizes = (*sizes, product_plan.panel, RASTER_ORDERS.index(product_plan.raster_order),
                      product_plan.data_parallel_tiles)

    def __call__(self, pointers, buffers, stream):
        self.run(self.grid, self.threads, self.shared_bytes, (*pointers, *buffers, *self.sizes), stream)


@functools.lru_cache(maxsize=4096)
def product_launch(family, kind, m, columns, k, sizes, device_id, stream_k):
    return Launch(family, kind, m, columns, k, sizes, device_id, stream_k)


def run(launch, pointers, workspace, stream):
    launch(pointers, workspace(launch.part_bytes, launch.counters) if launch.part_bytes else NO_BUFFERS, stream)


def vector_view(tensor, name, length, stream):
    if tensor is None:
        return None
    view = view_of(tensor, stream, name)
    require_dtype(view, name, "bfloat16")
    require(view.shape == (length,), f"{name} must have the shape {(length,)}")
    require(length <= 1 or view.strides == (1,), f"{name} must be contiguous")
    require(view.pointer % WORD_BYTES == 0, f"{name} must start at a multiple of 4 bytes")
    return view


def residual_view(residual, rows, width, stream, dtype="bfloat16"):
    if residual is None:
        return None, 0
    view = view_of(residual, stream, "residual")
    residual_rows, residual_width, stride = matrix(view, "residual", dtype)
    require((residual_rows, residual_width) == (rows, width), f"residual must have {rows} rows of {width} values")
    return view, stride


def pointer_of(view):
    return 0 if view is None else view.pointer


def gemm(a, b, scale_a, scale_b, out, *, bias=None, residual=None, activation="none", use_fast_accum=False,
         workspace=None, stream=None):
    """Writes the product of `zig_cu_kernels.torch.fp8_gemm` to `out` on the stream `runtime.stream_handle(stream)`.
    Without `workspace`, each work group computes whole tiles. With the callable `workspace(part_bytes, counters)`,
    work groups may share tiles, and the callable returns the device addresses of `part_bytes` bytes and of `counters`
    `u32` counters that hold zero. The kernel leaves the counters at zero."""
    device_id = runtime.current_device().device_id
    require_devices(device_id, a=a, b=b, scale_a=scale_a, scale_b=scale_b, out=out, bias=bias, residual=residual)
    a_view, b_view, out_view = (view_of(tensor, stream, name) for tensor, name in ((a, "a"), (b, "b"), (out, "out")))
    m, k, a_stride = matrix(a_view, "a", "float8_e4m3fn")
    require(len(b_view.shape) == 2, "b must be an [n, k] matrix")
    n, b_k, b_stride = matrix(b_view, "b", "float8_e4m3fn")
    require(b_k == k, "a and b need the same k")
    require(activation in ACTIVATIONS, f"activation must be one of {list(ACTIVATIONS)}")
    scales = scale_operands(scale_a, {"scale_b": scale_b}, a_view.shape[:-1], n, k, use_fast_accum, stream)
    out_stride = output_matrix(out_view, "out", m, n)
    bias_view = vector_view(bias, "bias", n, stream)
    residual, residual_stride = residual_view(residual, m, n, stream)
    require_separate("out", out_view, ("residual",), a=a_view, b=b_view, scale_a=scales.views[0],
                     scale_b=scales.views[1], bias=bias_view, residual=residual)
    if m == 0 or n == 0:
        return
    (a_rows, a_blocks), (b_rows, b_blocks) = scales.strides
    sizes = (a_stride, b_stride, a_rows, a_blocks, b_rows, b_blocks, scales.e8m0, out_stride, residual_stride, m, n, k,
             ACTIVATIONS[activation])
    launch = product_launch(family_of(scales.scaling_type, use_fast_accum), "store", m, n, k, sizes, device_id,
                            workspace is not None)
    pointers = (a_view.pointer, b_view.pointer, *(view.pointer for view in scales.views), out_view.pointer,
                pointer_of(bias_view), pointer_of(residual))
    run(launch, pointers, workspace, runtime.stream_handle(stream))


def fp8_b128_gemm(a, b, scale_a, scale_b, out, *, bias=None, workspace=None, stream=None):
    """Writes `bf16(A Bᵀ + bias)` to `out` like `gemm` with block128 scales, where `scale_a` holds the activation
    scale atoms `[ceil(m / 128), ceil(k / 512), 32, 16]` of `packs.fp8_b128_activation_scale_shape` and `scale_b` the
    `[ceil(n / 128), k / 128]` weight scales, both FP32 or both E8M0."""
    device_id = runtime.current_device().device_id
    require_devices(device_id, a=a, b=b, scale_a=scale_a, scale_b=scale_b, out=out, bias=bias)
    views = {name: view_of(tensor, stream, name) for name, tensor in
             (("a", a), ("b", b), ("scale_a", scale_a), ("scale_b", scale_b), ("out", out))}
    m, k, a_stride = matrix(views["a"], "a", "float8_e4m3fn")
    n, b_k, b_stride = matrix(views["b"], "b", "float8_e4m3fn")
    require(b_k == k, "a and b need the same k")
    check_product("block128", n, k, False, False)
    dtypes = {name: require_dtype(views[name], name, *SCALE_SUFFIXES) for name in ("scale_a", "scale_b")}
    require(dtypes["scale_a"] == dtypes["scale_b"], "scale_a and scale_b need the same dtype")
    spans = -(-k // (4 * SCALE_GROUP))
    atoms = views["scale_a"]
    require(len(atoms.shape) == 4 and atoms.shape[0] * SCALE_GROUP >= m and atoms.shape[1:] == (spans, 32, 16) and
            is_contiguous(atoms) and atoms.pointer % WORD_BYTES == 0,
            f"scale_a must hold the contiguous scale atoms of {m} rows")
    weights = views["scale_b"]
    require(weights.shape == (-(-n // SCALE_GROUP), k // SCALE_GROUP) and weights.pointer % WORD_BYTES == 0,
            f"scale_b must hold {(-(-n // SCALE_GROUP), k // SCALE_GROUP)} scales")
    b_rows, b_blocks = scale_strides(weights, "scale_b", (-(-n // SCALE_GROUP),), k // SCALE_GROUP)
    out_stride = output_matrix(views["out"], "out", m, n)
    bias_view = vector_view(bias, "bias", n, stream)
    require_separate("out", views["out"], a=views["a"], b=views["b"], scale_a=atoms, scale_b=weights, bias=bias_view)
    if m == 0 or n == 0:
        return
    e8m0 = (3 if dtypes["scale_a"] == "float8_e8m0fnu" else 0) | 16
    sizes = (a_stride, b_stride, 16, spans, b_rows, b_blocks, e8m0, out_stride, 0, m, n, k, ACTIVATIONS["none"])
    launch = product_launch("block128", "store", m, n, k, sizes, device_id, workspace is not None)
    pointers = (views["a"].pointer, views["b"].pointer, atoms.pointer, weights.pointer, views["out"].pointer,
                pointer_of(bias_view), 0)
    run(launch, pointers, workspace, runtime.stream_handle(stream))


def mxfp8_gemm(a, b, scale_a, scale_b, out, *, bias=None, residual=None, low=None, up=None, workspace=None,
               stream=None):
    """Writes `bf16(A Bᵀ + low upᵀ + residual + bias)` to `out` for the E4M3 matrices `a` `[m, k]` and `b` `[n, k]` with
    the E8M0 scale atoms `scale_a` `[ceil(m / 128), k / 128, 32, 16]` and `scale_b` `[ceil(n / 128), k / 128, 32, 16]`,
    on the stream `runtime.stream_handle(stream)`. The optional contiguous BF16 matrices `low` `[m, rank]` and `up`
    `[n, rank]` need `rank % 16 == 0`; the kernel adds `low upᵀ` to the FP32 sums. `workspace` works as in `gemm`."""
    device_id = runtime.current_device().device_id
    require_devices(device_id, a=a, b=b, scale_a=scale_a, scale_b=scale_b, out=out, bias=bias, residual=residual,
                    low=low, up=up)
    views = {name: view_of(tensor, stream, name) for name, tensor in
             (("a", a), ("b", b), ("scale_a", scale_a), ("scale_b", scale_b), ("out", out))}
    m, k, a_stride = matrix(views["a"], "a", "float8_e4m3fn")
    n, b_k, b_stride = matrix(views["b"], "b", "float8_e4m3fn")
    require(b_k == k and k > 0 and k % SCALE_GROUP == 0, f"a and b need the same positive k % {SCALE_GROUP} == 0")
    require(n % N_MULTIPLE == 0 and max(n, k) <= MAX_U32, f"b needs a row count that is a multiple of {N_MULTIPLE}")
    for name, rows in (("scale_a", m), ("scale_b", n)):
        view = views[name]
        require_dtype(view, name, "float8_e8m0fnu")
        require(len(view.shape) == 4 and view.shape[0] * SCALE_GROUP >= rows and
                view.shape[1:] == (k // SCALE_GROUP, 32, 16) and is_contiguous(view) and
                view.pointer % WORD_BYTES == 0, f"{name} must hold the contiguous scale atoms of {rows} rows")
    out_stride = output_matrix(views["out"], "out", m, n)
    bias_view = vector_view(bias, "bias", n, stream)
    residual_f32 = residual is not None and view_of(residual, stream, "residual").dtype == "float32"
    residual, residual_stride = residual_view(residual, m, n, stream, "float32" if residual_f32 else "bfloat16")
    require((low is None) == (up is None), "low and up go together")
    low_view, up_view, rank = low_rank_views(low, up, m, n, stream) if low is not None else (None, None, 0)
    require_separate("out", views["out"], ("residual",), a=views["a"], b=views["b"], scale_a=views["scale_a"],
                     scale_b=views["scale_b"], bias=bias_view, residual=residual, low=low_view, up=up_view)
    if m == 0:
        return
    sizes = (a_stride, b_stride, 0, 0, 0, 0, 3 | 8 * residual_f32, out_stride, residual_stride, m, n, k, ACTIVATIONS["none"])
    pointers = (views["a"].pointer, views["b"].pointer, views["scale_a"].pointer, views["scale_b"].pointer,
                views["out"].pointer, pointer_of(bias_view), pointer_of(residual))
    if low_view is None:
        launch = product_launch("mxfp8", "store", m, n, k, sizes, device_id, workspace is not None)
    else:
        launch = product_launch("mxfp8", "low_rank", m, n, k, (*sizes, rank), device_id, workspace is not None)
        pointers = (*pointers, low_view.pointer, up_view.pointer)
    run(launch, pointers, workspace, runtime.stream_handle(stream))


def low_rank_views(low, up, m, n, stream):
    low_view, up_view = view_of(low, stream, "low"), view_of(up, stream, "up")
    low_rows, rank, low_stride = matrix(low_view, "low", "bfloat16")
    up_rows, up_rank, up_stride = matrix(up_view, "up", "bfloat16")
    require((low_rows, up_rows, up_rank) == (m, n, rank), f"low must be [{m}, rank] and up [{n}, rank]")
    require(rank > 0 and rank % 16 == 0 and rank <= MAX_U32, "the rank must be a positive multiple of 16")
    require(low_rows <= 1 or low_stride == rank, "low must be contiguous")
    require(up_rows <= 1 or up_stride == rank, "up must be contiguous")
    return low_view, up_view, rank


def gated_gemm(a, w_gate, w_up, scale_a, scale_gate, scale_up, out, *, bias_gate=None, bias_up=None, residual=None,
               activation="silu", use_fast_accum=False, workspace=None, stream=None):
    """Writes the product of `zig_cu_kernels.torch.fp8_gated_gemm` to `out`; `workspace` and `stream` follow `gemm`."""
    device_id = runtime.current_device().device_id
    require_devices(device_id, a=a, w_gate=w_gate, w_up=w_up, scale_a=scale_a, scale_gate=scale_gate,
                    scale_up=scale_up, out=out, bias_gate=bias_gate, bias_up=bias_up, residual=residual)
    a_view, gate_view, up_view, out_view = (view_of(tensor, stream, name) for tensor, name in
                                            ((a, "a"), (w_gate, "w_gate"), (w_up, "w_up"), (out, "out")))
    m, k, a_stride = matrix(a_view, "a", "float8_e4m3fn")
    require(len(gate_view.shape) == 2 and gate_view.shape == up_view.shape, "w_gate and w_up must be [f, k] matrices")
    f, gate_k, gate_stride = matrix(gate_view, "w_gate", "float8_e4m3fn")
    _, _, up_stride = matrix(up_view, "w_up", "float8_e4m3fn")
    require(gate_k == k, "a and the weights need the same k")
    require(gate_stride == up_stride or f <= 1, "w_gate and w_up must share a row stride")
    require(activation in ACTIVATIONS, f"activation must be one of {list(ACTIVATIONS)}")
    weight_scales = {"scale_gate": scale_gate, "scale_up": scale_up}
    scales = scale_operands(scale_a, weight_scales, a_view.shape[:-1], f, k, use_fast_accum, stream)
    out_stride = output_matrix(out_view, "out", m, f)
    gate_bias, up_bias = (vector_view(bias, name, f, stream) for bias, name in ((bias_gate, "bias_gate"),
                                                                                  (bias_up, "bias_up")))
    residual, residual_stride = residual_view(residual, m, f, stream)
    require_separate("out", out_view, ("residual",), a=a_view, w_gate=gate_view, w_up=up_view,
                     scale_a=scales.views[0], scale_gate=scales.views[1], scale_up=scales.views[2],
                     bias_gate=gate_bias, bias_up=up_bias, residual=residual)
    if m == 0 or f == 0:
        return
    (a_rows, a_blocks), (gate_rows, gate_blocks), (up_rows, up_blocks) = scales.strides
    sizes = (a_stride, gate_stride, a_rows, a_blocks, gate_rows, gate_blocks, up_rows, up_blocks, scales.e8m0,
             out_stride, residual_stride, m, f, k, ACTIVATIONS[activation])
    launch = product_launch(family_of(scales.scaling_type, use_fast_accum), "gated", m, f, k, sizes, device_id,
                            workspace is not None)
    pointers = (a_view.pointer, gate_view.pointer, up_view.pointer, *(view.pointer for view in scales.views),
                out_view.pointer, pointer_of(gate_bias), pointer_of(up_bias), pointer_of(residual))
    run(launch, pointers, workspace, runtime.stream_handle(stream))


def store_choice(family, m, columns, k, stream_k=False):
    """The `Choice` that `plan` makes for the `store` product of `m` rows, `columns` columns and depth `k`."""
    device_id = runtime.current_device().device_id
    return choose(family, "store", m, columns, k, runtime.num_sm(device_id), memory_rate(device_id), stream_k)


def store_tiling(family, m, columns, k, stream_k=False):
    """The tiling that `plan` picks for the `store` product of `m` rows, `columns` columns and depth `k`."""
    return store_choice(family, m, columns, k, stream_k).tiling


def fp8_b128_glu_gemm(a, w_gate, w_up, scale_a, scale_gate, scale_up, codes, scales, *, activation, z_gate=None,
                      z_up=None, stream=None):
    """Writes the block128 activation pack `(codes, scales)` of `h = glu(up, gate)` with `up = bf16(A w_upᵀ)` and
    `gate = bf16(A w_gateᵀ)`, and `up` and `gate` to `z_up` and `z_gate` when given. `scale_a` holds activation scale
    atoms and `scale_gate`, `scale_up` the `[f / 128, k / 128]` weight scales, all FP32 or all E8M0. `scales` holds the
    `[ceil(m / 128), ceil(f / 512), 32, 16]` scale atoms of `h`, FP32 or E8M0; the kernel writes the blocks of `f`
    columns, so scale blocks past `f` keep their value."""
    device_id = runtime.current_device().device_id
    tensors = {"a": a, "w_gate": w_gate, "w_up": w_up, "scale_a": scale_a, "scale_gate": scale_gate,
               "scale_up": scale_up, "codes": codes, "scales": scales, "z_gate": z_gate, "z_up": z_up}
    require_devices(device_id, **tensors)
    views = {name: view_of(tensor, stream, name) for name, tensor in tensors.items() if tensor is not None}
    m, k, a_stride = matrix(views["a"], "a", "float8_e4m3fn")
    f, gate_k, b_stride = matrix(views["w_gate"], "w_gate", "float8_e4m3fn")
    require(views["w_up"].shape == (f, k) and gate_k == k, "w_gate and w_up must be [f, k] matrices")
    require(matrix(views["w_up"], "w_up", "float8_e4m3fn")[2] == b_stride, "w_gate and w_up must share a row stride")
    require(f % SCALE_GROUP == 0, f"f must be a multiple of {SCALE_GROUP}")
    require(activation in GLU_ACTIVATIONS, f"activation must be one of {GLU_ACTIVATIONS}")
    check_product("block128", f, k, False, False)
    dtype = require_dtype(views["scale_a"], "scale_a", *SCALE_SUFFIXES)
    for name in ("scale_gate", "scale_up"):
        require(require_dtype(views[name], name, *SCALE_SUFFIXES) == dtype, f"{name} needs the dtype of scale_a")
        require(views[name].shape == (f // SCALE_GROUP, k // SCALE_GROUP) and views[name].pointer % WORD_BYTES == 0,
                f"{name} must hold {(f // SCALE_GROUP, k // SCALE_GROUP)} scales")
    spans = -(-k // (4 * SCALE_GROUP))
    atoms = views["scale_a"]
    require(len(atoms.shape) == 4 and atoms.shape[0] * SCALE_GROUP >= m and atoms.shape[1:] == (spans, 32, 16) and
            is_contiguous(atoms) and atoms.pointer % WORD_BYTES == 0,
            f"scale_a must hold the contiguous scale atoms of {m} rows")
    h_spans = -(-f // (4 * SCALE_GROUP))
    h_dtype = require_dtype(views["scales"], "scales", *SCALE_SUFFIXES)
    require(views["scales"].shape == (-(-m // SCALE_GROUP), h_spans, 32, 16) and is_contiguous(views["scales"]),
            "scales must hold the contiguous scale atoms of h")
    code_rows, code_width, code_stride = matrix(views["codes"], "codes", "float8_e4m3fn")
    require((code_rows, code_width) == (m, f) and code_stride % ROW_ALIGNMENT_BYTES == 0, f"codes must be [{m}, {f}]")
    z_stride = 0
    for name in ("z_gate", "z_up"):
        if name in views:
            stride = output_matrix(views[name], name, m, f)
            require(z_stride in (0, stride), "z_gate and z_up must share a row stride")
            z_stride = stride
    if m == 0:
        return
    gate_rows, gate_blocks = scale_strides(views["scale_gate"], "scale_gate", (f // SCALE_GROUP,), k // SCALE_GROUP)
    up_rows, up_blocks = scale_strides(views["scale_up"], "scale_up", (f // SCALE_GROUP,), k // SCALE_GROUP)
    e8m0 = (7 if dtype == "float8_e8m0fnu" else 0) | 16 | (32 if h_dtype == "float8_e8m0fnu" else 0)
    sizes = (a_stride, b_stride, 16, spans, gate_rows, gate_blocks, up_rows, up_blocks, e8m0, z_stride, code_stride,
             h_spans, m, f, f, k, GLU_ACTIVATIONS.index(activation), math.inf)
    launch = product_launch("block128", "glu", m, f, k, sizes, device_id, False)
    pointers = (views["a"].pointer, views["w_gate"].pointer, views["w_up"].pointer, atoms.pointer,
                views["scale_gate"].pointer, views["scale_up"].pointer, *(pointer_of(views.get(name)) for name in
                ("z_gate", "z_up")), views["codes"].pointer, views["scales"].pointer)
    run(launch, pointers, None, runtime.stream_handle(stream))


def mxfp8_glu_gemm(a, w_gate, w_up, scale_a, scale_gate, scale_up, codes, scales, *, activation, hidden,
                   clamp_limit=math.inf, z_gate=None, z_up=None, stream=None):
    """Writes the MXFP8 pack `(codes, scales)` of `h = glu(up, gate)` with `up = bf16(A w_upᵀ)` and
    `gate = bf16(A w_gateᵀ)`, and `up` and `gate` to `z_up` and `z_gate` when given. `w_gate` and `w_up` are `[f, k]`
    with `f % 128 == 0`, and their E8M0 scale atoms are `[f / 128, k / 128, 32, 16]`. `codes` is `[m, f]`, `scales`
    holds the `[ceil(m / 128), f / 128, 32, 16]` scale atoms of `h`, and the columns from `hidden` on hold zeros."""
    device_id = runtime.current_device().device_id
    tensors = {"a": a, "w_gate": w_gate, "w_up": w_up, "scale_a": scale_a, "scale_gate": scale_gate,
               "scale_up": scale_up, "codes": codes, "scales": scales, "z_gate": z_gate, "z_up": z_up}
    require_devices(device_id, **tensors)
    views = {name: view_of(tensor, stream, name) for name, tensor in tensors.items() if tensor is not None}
    m, k, a_stride = matrix(views["a"], "a", "float8_e4m3fn")
    f, gate_k, b_stride = matrix(views["w_gate"], "w_gate", "float8_e4m3fn")
    require(views["w_up"].shape == (f, k) and gate_k == k, "w_gate and w_up must be [f, k] matrices")
    require(matrix(views["w_up"], "w_up", "float8_e4m3fn")[2] == b_stride, "w_gate and w_up must share a row stride")
    require(f % SCALE_GROUP == 0 and k % SCALE_GROUP == 0 and 0 < hidden <= f,
            f"f and k must be multiples of {SCALE_GROUP} and 0 < hidden <= f")
    require(activation in GLU_ACTIVATIONS, f"activation must be one of {GLU_ACTIVATIONS}")
    for name, rows in (("scale_a", m), ("scale_gate", f), ("scale_up", f), ("scales", m)):
        view = views[name]
        groups = (f if name == "scales" else k) // SCALE_GROUP
        require_dtype(view, name, "float8_e8m0fnu")
        require(len(view.shape) == 4 and view.shape[0] * SCALE_GROUP >= rows and
                view.shape[1:] == (groups, 32, 16) and is_contiguous(view) and view.pointer % WORD_BYTES == 0,
                f"{name} must hold the contiguous scale atoms of {rows} rows")
    code_rows, code_width, code_stride = matrix(views["codes"], "codes", "float8_e4m3fn")
    require((code_rows, code_width) == (m, f) and code_stride % ROW_ALIGNMENT_BYTES == 0, f"codes must be [{m}, {f}]")
    z_stride = 0
    for name in ("z_gate", "z_up"):
        if name in views:
            stride = output_matrix(views[name], name, m, f)
            require(z_stride in (0, stride), "z_gate and z_up must share a row stride")
            z_stride = stride
    if m == 0:
        return
    sizes = (a_stride, b_stride, 0, 0, 0, 0, 0, 0, 3, z_stride, code_stride, f // SCALE_GROUP, m, f, hidden, k,
             GLU_ACTIVATIONS.index(activation), clamp_limit)
    launch = product_launch("mxfp8", "glu", m, f, k, sizes, device_id, False)
    pointers = (views["a"].pointer, views["w_gate"].pointer, views["w_up"].pointer, views["scale_a"].pointer,
                views["scale_gate"].pointer, views["scale_up"].pointer, *(pointer_of(views.get(name)) for name in
                ("z_gate", "z_up")), views["codes"].pointer, views["scales"].pointer)
    run(launch, pointers, None, runtime.stream_handle(stream))


def fp8_rowwise_glu_gemm(a, w_gate, w_up, scale_a, scale_gate, scale_up, h, *, activation, z_gate=None, z_up=None,
                         stream=None):
    """Writes `h = glu(up, gate)` in BF16 with `up = bf16(A w_upᵀ)` and `gate = bf16(A w_gateᵀ)` for the FP32 row
    scales `scale_a`, `scale_gate` and `scale_up`, and `up` and `gate` to `z_up` and `z_gate` when given. `w_gate` and
    `w_up` are `[f, k]` with `f % 128 == 0`, and `h` is `[m, f]`."""
    device_id = runtime.current_device().device_id
    tensors = {"a": a, "w_gate": w_gate, "w_up": w_up, "h": h, "z_gate": z_gate, "z_up": z_up}
    require_devices(device_id, scale_a=scale_a, scale_gate=scale_gate, scale_up=scale_up, **tensors)
    views = {name: view_of(tensor, stream, name) for name, tensor in tensors.items() if tensor is not None}
    m, k, a_stride = matrix(views["a"], "a", "float8_e4m3fn")
    f, gate_k, b_stride = matrix(views["w_gate"], "w_gate", "float8_e4m3fn")
    require(views["w_up"].shape == (f, k) and gate_k == k, "w_gate and w_up must be [f, k] matrices")
    require(matrix(views["w_up"], "w_up", "float8_e4m3fn")[2] == b_stride, "w_gate and w_up must share a row stride")
    require(f % SCALE_GROUP == 0, f"f must be a multiple of {SCALE_GROUP}")
    require(activation in GLU_ACTIVATIONS, f"activation must be one of {GLU_ACTIVATIONS}")
    scales = scale_operands(scale_a, {"scale_gate": scale_gate, "scale_up": scale_up}, views["a"].shape[:-1], f, k,
                            False, stream)
    require(scales.scaling_type == "rowwise" and scales.e8m0 == 0, "the scales must be FP32 row scales")
    h_stride = output_matrix(views["h"], "h", m, f)
    z_stride = 0
    for name in ("z_gate", "z_up"):
        if name in views:
            stride = output_matrix(views[name], name, m, f)
            require(z_stride in (0, stride), "z_gate and z_up must share a row stride")
            z_stride = stride
    if m == 0:
        return
    (a_rows, a_blocks), (gate_rows, gate_blocks), (up_rows, up_blocks) = scales.strides
    sizes = (a_stride, b_stride, a_rows, a_blocks, gate_rows, gate_blocks, up_rows, up_blocks, 0, z_stride, h_stride,
             0, m, f, f, k, GLU_ACTIVATIONS.index(activation), math.inf)
    launch = product_launch("rowwise", "glu", m, f, k, sizes, device_id, False)
    pointers = (views["a"].pointer, views["w_gate"].pointer, views["w_up"].pointer,
                *(view.pointer for view in scales.views), *(pointer_of(views.get(name)) for name in ("z_gate", "z_up")),
                views["h"].pointer, 0)
    run(launch, pointers, None, runtime.stream_handle(stream))


def quantizer_input(view, name):
    element = ELEMENTS[require_dtype(view, name, *ELEMENTS)]
    rows, width, stride = matrix(view, name, "bfloat16", "float32")
    require(width % 8 == 0 and rows * (width // 8) <= MAX_U32, f"{name} needs rows of a multiple of 8 values and "
            "fewer than 2^35 values")
    return element, rows, width, stride


def codes_view(codes, name, count, stream):
    view = view_of(codes, stream, name)
    require_dtype(view, name, "float8_e4m3fn")
    require(view.pointer % CODE_ALIGNMENT_BYTES == 0, f"{name} must start at a multiple of 8 bytes")
    require(math.prod(view.shape) == count and is_contiguous(view), f"{name} must be contiguous with {count} values")
    return view


def scales_view(scales, name, shape, stream):
    view = view_of(scales, stream, name)
    suffix = SCALE_SUFFIXES[require_dtype(view, name, *SCALE_SUFFIXES)]
    require(view.shape == shape, f"{name} must have the shape {shape}")
    require(suffix or view.pointer % WORD_BYTES == 0, f"{name} must start at a multiple of 4 bytes")
    return view, suffix


def launch_rows(kernel_name, input_pointer, input_stride, codes, scales, scale_stride, m, k, stream):
    kernel, info = runtime.load(kernel_name)
    work_groups = -(-m * (k // 8) // info.threads)
    values = (input_pointer, input_stride, codes, scales, scale_stride, m, k)
    runtime.run(kernel, (work_groups, 1, 1), info.threads, 0, QUANTIZE_SIGNATURES["rows"], values, stream)


def quantize_rows(x, codes, scales, *, stream=None):
    """Quantizes `x` by the rule of `zig_cu_kernels.torch.quantize_fp8_rows` into `codes` and `scales`."""
    device_id = runtime.current_device().device_id
    require_devices(device_id, x=x, codes=codes, scales=scales)
    x_view = view_of(x, stream, "x")
    element, m, k, stride = quantizer_input(x_view, "x")
    require(k % SCALE_GROUP == 0, "the rows of x must hold a multiple of 128 values")
    q = codes_view(codes, "codes", m * k, stream)
    s, suffix = scales_view(scales, "scales", (m, k // SCALE_GROUP), stream)
    require(m <= 1 or k == 0 or s.strides[0] == 1, "the scales of one scale group must lie next to each other")
    scale_stride = s.strides[1] if k > SCALE_GROUP else m
    require(k <= SCALE_GROUP or m <= 1 or scale_stride >= m, "the elements of scales share memory")
    for name, output in (("codes", q), ("scales", s)):
        require_separate(name, output, x=x_view)
    require_separate("codes", q, scales=s)
    if m * k == 0:
        return
    launch_rows(f"fp8_quantize_rows_{element}{suffix}", x_view.pointer, stride, q.pointer, s.pointer, scale_stride,
                m, k, stream)


def launch_tiles(kernel_name, input_pointer, input_stride, codes, scales, scale_stride, n, k, stream):
    kernel, info = runtime.load(kernel_name)
    tiles = -(-n // SCALE_GROUP) * (k // SCALE_GROUP)
    require(tiles <= MAX_WORK_GROUPS, "w must have at most 2^31 - 1 tiles of 128 x 128 values")
    values = (input_pointer, input_stride, codes, scales, scale_stride, n, k)
    runtime.run(kernel, (tiles, 1, 1), info.threads, 0, QUANTIZE_SIGNATURES["tiles"], values, stream)


def quantize_tiles(w, codes, scales, *, stream=None):
    """Quantizes `w` by the rule of `zig_cu_kernels.torch.quantize_fp8_tiles` into `codes` and `scales`."""
    device_id = runtime.current_device().device_id
    require_devices(device_id, w=w, codes=codes, scales=scales)
    w_view = view_of(w, stream, "w")
    require(len(w_view.shape) == 2, "w must be an [n, k] matrix")
    element, n, k, stride = quantizer_input(w_view, "w")
    require(k % SCALE_GROUP == 0, "the rows of w must hold a multiple of 128 values")
    q = codes_view(codes, "codes", n * k, stream)
    tile_rows = -(-n // SCALE_GROUP)
    s, suffix = scales_view(scales, "scales", (tile_rows, k // SCALE_GROUP), stream)
    require(k <= SCALE_GROUP or s.strides[1] == 1, "the scales of one tile row must be contiguous")
    scale_stride = s.strides[0] if tile_rows > 1 else k // SCALE_GROUP
    require(tile_rows <= 1 or scale_stride >= k // SCALE_GROUP, "the elements of scales share memory")
    for name, output in (("codes", q), ("scales", s)):
        require_separate(name, output, w=w_view)
    require_separate("codes", q, scales=s)
    if n * k == 0:
        return
    launch_tiles(f"fp8_quantize_tiles_{element}{suffix}", w_view.pointer, stride, q.pointer, s.pointer, scale_stride,
                 n, k, stream)


def launch_tensorwise(element, input_pointer, input_stride, codes, rows, width, scale, amax, stream):
    chunks = rows * (width // 8)
    if amax:
        kernel, info = runtime.load(f"fp8_amax_{element}")
        device_id = runtime.current_device().device_id
        work_groups = max(1, min(-(-chunks // info.threads), AMAX_GROUPS_PER_SM * runtime.num_sm(device_id)))
        values = (input_pointer, input_stride, rows, width, amax)
        runtime.run(kernel, (work_groups, 1, 1), info.threads, 0, QUANTIZE_SIGNATURES["amax"], values, stream)
    kernel, info = runtime.load(f"fp8_quantize_tensorwise_{element}")
    work_groups = max(1, -(-chunks // info.threads))
    values = (input_pointer, input_stride, codes, rows, width, scale, amax)
    runtime.run(kernel, (work_groups, 1, 1), info.threads, 0, QUANTIZE_SIGNATURES["tensorwise"], values, stream)


def quantize_tensorwise(x, codes, scale, *, amax=None, stream=None):
    """Quantizes `x` into `codes` with the FP32 `scale`. With `amax`, a one-element 32-bit tensor of zero bits, a
    first kernel writes the amax of `x` to `amax`, and the quantize kernel writes the scale of that amax to `scale`."""
    device_id = runtime.current_device().device_id
    require_devices(device_id, x=x, codes=codes, scale=scale, amax=amax)
    x_view = view_of(x, stream, "x")
    element, rows, width, stride = quantizer_input(x_view, "x")
    q = codes_view(codes, "codes", rows * width, stream)
    scale_view = view_of(scale, stream, "scale")
    require_dtype(scale_view, "scale", "float32")
    require(math.prod(scale_view.shape) == 1 and scale_view.pointer % WORD_BYTES == 0,
            "scale must be one FP32 value at a multiple of 4 bytes")
    require_separate("codes", q, x=x_view, scale=scale_view)
    amax_pointer = 0
    if amax is not None:
        amax_view = view_of(amax, stream, "amax")
        require(math.prod(amax_view.shape) == 1 and amax_view.itemsize == WORD_BYTES, "amax must be one 32-bit word")
        require_separate("amax", amax_view, x=x_view, codes=q)
        require_separate("scale", scale_view, x=x_view, codes=q, amax=amax_view)
        amax_pointer = amax_view.pointer
    elif rows * width == 0:
        return
    launch_tensorwise(element, x_view.pointer, stride, q.pointer, rows, width, scale_view.pointer, amax_pointer,
                      stream)
