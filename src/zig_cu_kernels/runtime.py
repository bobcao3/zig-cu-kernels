import ctypes
import functools
import hashlib
import math
import numbers
import os
import struct
import threading
from collections import namedtuple
from pathlib import Path

import numpy as np
from cuda.bindings import driver
from cuda.core import Device, ObjectCode, Stream, system
from cuda.core._utils.cuda_utils import CUDAError

SCALAR_TYPES = {"p": np.uint64, "q": np.uint64, "u": np.uint32, "i": np.int32, "f": np.float32}
FLOAT_CODES = {code for code, scalar in SCALAR_TYPES.items() if np.issubdtype(scalar, np.floating)}
DEFAULT_DYNAMIC_SHARED_BYTES = 48 * 1024
# A streaming multiprocessor (SM) allocates shared memory to a work group in units of this many bytes.
SHARED_ALLOCATION_BYTES = 128
# The code "m" passes a `CUtensorMap`, the 128 bytes of a tensor map, by value.
TENSOR_MAP_CODE = "m"
TENSOR_MAP_BYTES = 128
TENSOR_MAP_ALIGNMENT = 64
SLOT_FORMATS = {"p": "Q", "q": "Q", "u": "I4x", "i": "i4x", "f": "f4x", TENSOR_MAP_CODE: f"{TENSOR_MAP_BYTES}s"}
SLOT_BYTES = 8
MAX_LAUNCH_CONFIGS = 256
FP32 = struct.Struct("=f")
DOUBLE_SIGNIFICAND_BITS = 53
MAX_STREAM_WRAPPERS = 256
# For a manifest without `# targets` lines, the runtime assumes the targets of a build without `-Dsm` for each `arch`.
UNRECORDED_TARGETS = {80: ("sm_80", "sm_86", "sm_89"), 89: ("sm_89",)}

Manifest = namedtuple("Manifest", "directory entries source_digest targets options")


def manifest_value(text):
    return int(text) if text.isdigit() else text


def parse_manifest(directory, path):
    header, *lines = path.read_text().splitlines()
    marker, *columns = header.split()
    if marker != "#":
        raise RuntimeError(f"{path} has no header line; rebuild the package")
    info_type = namedtuple("KernelInfo", columns)
    entries, targets, source_digest, options = {}, {}, None, ""
    for line in lines:
        if line.startswith("#"):
            _, key, *rest = line.split(maxsplit=2)
            values = rest[0] if rest else ""
            if key == "sources":
                source_digest = values.split()[1]
            elif key == "options":
                options = values
            elif key == "targets":
                module, *module_targets = values.split()
                targets[module] = tuple(module_targets)
        else:
            values = line.split()
            entries[values[0]] = info_type(values[0], *map(manifest_value, values[1:]))
    return Manifest(directory, entries, source_digest, targets, options)


@functools.cache
def read_manifest():
    package = Path(__file__).parent
    override = os.environ.get("ZIG_CU_KERNELS_DIR")
    directories = (Path(override),) if override else (package / "cubin", package.parents[1] / "zig-out")
    for directory in directories:
        path = directory / "kernels.manifest"
        if path.exists():
            return parse_manifest(directory, path)
    searched = " or ".join(str(directory) for directory in directories)
    raise FileNotFoundError(f"kernels.manifest not found in {searched}; build the package first. A build that fails "
                            f"leaves no manifest.")


def manifest():
    contents = read_manifest()
    return contents.directory, contents.entries


def source_digest(root):
    """Returns the digest of the sources under `root` that `build.zig` writes into the manifest."""
    with os.scandir(os.fsencode(root / "zig")) as entries:
        names = [entry.name for entry in entries
                 if entry.name.endswith(b".zig") and entry.is_file(follow_symlinks=False)]
    paths = sorted([b"build.zig", *(b"zig/" + name for name in names)])
    digest = hashlib.sha256()
    for path in paths:
        contents = (root / os.fsdecode(path)).read_bytes()
        digest.update(path + b"\0" + len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


@functools.cache
def check_sources():
    contents = read_manifest()
    directory = Path(os.path.abspath(contents.directory))
    root = directory.parent
    if not ((root / "build.zig").is_file() and (root / "zig").is_dir()):
        return
    if contents.source_digest != source_digest(root):
        command = " ".join(filter(None, ("zig build -p", str(directory), contents.options)))
        raise RuntimeError(f"the kernels in {directory} were built from other sources than those in {root}; "
                           f"rebuild them with `{command}` in {root}")


@functools.cache
def declared():
    lines = (manifest()[0] / "kernels.declared").read_text().split()
    partial = bool(lines) and lines[0] == "partial"
    return partial, tuple(lines[1:] if partial else lines)


def current_device():
    device = Device()
    device.set_current()
    return device


@functools.cache
def primary_context(device_id):
    Device(device_id).set_current()
    status, context = driver.cuCtxGetCurrent()
    if status:
        raise RuntimeError(f"cuCtxGetCurrent failed: {status}")
    return int(context)


def bind_device(device_id):
    """Makes the primary context of the device current on the calling thread, because `run` launches into the
    current context."""
    status, context = driver.cuCtxGetCurrent()
    if status or int(context) != primary_context(device_id):
        Device(device_id).set_current()


@functools.cache
def num_sm(device_id):
    return Device(device_id).properties.multiprocessor_count


@functools.cache
def l2_bytes(device_id):
    return Device(device_id).properties.l2_cache_size


@functools.cache
def max_shared_bytes(device_id):
    return Device(device_id).properties.max_shared_memory_per_block_optin


@functools.cache
def shared_bytes_per_sm(device_id):
    return Device(device_id).properties.max_shared_memory_per_multiprocessor


@functools.cache
def reserved_shared_bytes(device_id):
    return Device(device_id).properties.reserved_shared_memory_per_block


@functools.cache
def compute_capability(device_id):
    capability = Device(device_id).compute_capability
    return capability.major * 10 + capability.minor


def info(kernel_name):
    _, entries = manifest()
    if kernel_name not in entries:
        raise RuntimeError(f"the kernel {kernel_name} is not built")
    return entries[kernel_name]


def cubin_path(entry):
    directory, _ = manifest()
    for cubin_directory in (directory / "cubin", directory):
        path = cubin_directory / f"{entry.module}.cubin"
        if path.exists():
            return path
    return None


def targets_of(entry):
    targets = read_manifest().targets.get(entry.module)
    return UNRECORDED_TARGETS.get(entry.arch, ()) if targets is None else targets


def target_capability(target):
    """Returns the compute capability (major, minor) of a target such as `sm_89`, `sm_90a` or `sm_100f`."""
    return divmod(int(target.removeprefix("sm_").removesuffix("a").removesuffix("f")), 10)


def target_runs_on(target, major, minor):
    """CUDA runs machine code for compute capability X.y on a device of compute capability X.z when z >= y (CUDA C++
    Programming Guide, "Binary Compatibility"). Code for a target with the suffix "a" runs on its own compute capability
    only, and code for a target with the suffix "f" runs on the later devices of its family (PTX ISA 11.1.2, ".target").
    """
    target_major, target_minor = target_capability(target)
    if target.endswith("a"):
        return (target_major, target_minor) == (major, minor)
    return target_major == major and target_minor <= minor


def runs_on(entry, device_id):
    major, minor = divmod(compute_capability(device_id), 10)
    return any(target_runs_on(target, major, minor) for target in targets_of(entry))


def incompatibility(entry, device_id):
    major, minor = divmod(compute_capability(device_id), 10)
    targets = targets_of(entry)
    held = f"machine code for {', '.join(targets)}" if targets else "no machine code"
    return (f"{entry.kernel} cannot run on device {device_id} of compute capability {major}.{minor}, because the build "
            f"holds {held} in the module {entry.module}")


def unavailable_reason(kernel_name, device_id=None):
    """Returns why the device cannot run the kernel, or None when it can."""
    check_sources()
    _, entries = manifest()
    if kernel_name not in entries:
        return f"the kernel {kernel_name} is not built"
    device_id = current_device().device_id if device_id is None else device_id
    entry = entries[kernel_name]
    if not runs_on(entry, device_id):
        return incompatibility(entry, device_id)
    if entry.shared_bytes > max_shared_bytes(device_id):
        return (f"{kernel_name} needs {entry.shared_bytes} bytes of shared memory per work group, and device "
                f"{device_id} offers {max_shared_bytes(device_id)}")
    if cubin_path(entry) is None:
        return f"the build holds no cubin of the module {entry.module}"
    return None


def is_available(kernel_name, device_id=None):
    return unavailable_reason(kernel_name, device_id) is None


def driver_version():
    cuda = "CUDA " + ".".join(map(str, system.get_user_mode_driver_version()))
    try:
        major, *rest = system.get_kernel_mode_driver_version()
    except (RuntimeError, system.NvmlError):
        return cuda
    return f"{'.'.join([str(major), *(f'{part:02d}' for part in rest)])} ({cuda})"


def load(kernel_name):
    return load_for(current_device().device_id, kernel_name)


@functools.cache
def load_for(device_id, kernel_name):
    check_sources()
    entry = info(kernel_name)
    if not runs_on(entry, device_id):
        raise RuntimeError(incompatibility(entry, device_id))
    path = cubin_path(entry)
    if path is not None:
        cubin = path.read_bytes()
        try:
            kernel = ObjectCode.from_cubin(cubin).get_kernel(kernel_name)
        except CUDAError as error:
            raise RuntimeError(f"the CUDA driver {driver_version()} cannot load the kernel {kernel_name} from {path}: "
                               f"{error}") from error
        if entry.shared_bytes > DEFAULT_DYNAMIC_SHARED_BYTES:
            attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            status, = driver.cuKernelSetAttribute(attribute, entry.shared_bytes, kernel.handle, device_id)
            if status != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuKernelSetAttribute failed: {status}")
        return kernel, entry
    raise RuntimeError(f"{entry.module} has no cubin")


@functools.cache
def parameter_sizes(kernel):
    sizes = []
    while True:
        status, _, size = driver.cuKernelGetParamInfo(kernel.handle, len(sizes))
        if status != driver.CUresult.CUDA_SUCCESS:
            return tuple(sizes)
        sizes.append(size)


@functools.cache
def verify_parameters(kernel, signature):
    sizes = list(parameter_sizes(kernel))
    expected = [TENSOR_MAP_BYTES if code == TENSOR_MAP_CODE else np.dtype(SCALAR_TYPES[code]).itemsize
                for code in signature]
    if sizes != expected:
        raise RuntimeError(f"kernel parameter sizes {sizes} do not match launch signature {signature!r}")


@functools.lru_cache(maxsize=MAX_STREAM_WRAPPERS)
def stream_wrapper(handle, context, device_id):
    # Each call of `Stream.from_handle` builds a new class, and only the cycle collector frees such a class.
    return Stream.from_handle(handle)


def stream_from_handle(handle):
    """The driver may give the handle of a destroyed stream to a new stream of another context or device, so the cache
    of cuda.core streams keys on the handle, the context and the device."""
    status, context = driver.cuStreamGetCtx(handle)
    if status:
        raise RuntimeError(f"cuStreamGetCtx failed: {status}")
    status, device = driver.cuStreamGetDevice(handle)
    if status:
        raise RuntimeError(f"cuStreamGetDevice failed: {status}")
    return stream_wrapper(handle, int(context), int(device))


def stream_of(stream):
    if stream is None:
        return current_device().default_stream
    if isinstance(stream, int):
        return stream_from_handle(stream)
    return current_device().create_stream(stream)


def stream_handle(stream):
    """Returns the driver handle of a stream in one of the forms of README section "Launchers"."""
    return stream if isinstance(stream, int) else int(stream_of(stream).handle)


def fp32(name, value):
    """Returns the Python number `value` rounded to FP32, to nearest with ties to even (README section
    "Scalar arguments")."""
    if type(value) is not float and (isinstance(value, bool) or not isinstance(value, numbers.Real)):
        raise TypeError(f"{name} must be a Python number, got {type(value).__name__}")
    try:
        number = value if type(value) is float else odd_rounded(value) if isinstance(value, int) else float(value)
        return FP32.unpack(FP32.pack(number))[0]
    except OverflowError:
        raise OverflowError(f"{name} = {value} rounds to infinity in FP32") from None


def odd_rounded(value):
    """Rounds the integer `value` to 53 significant bits by round to odd, so that rounding the result to FP32 gives
    the FP32 rounding of `value`."""
    magnitude = abs(value)
    dropped = max(0, magnitude.bit_length() - DOUBLE_SIGNIFICAND_BITS)
    kept = magnitude >> dropped | (magnitude & ((1 << dropped) - 1) != 0)
    return math.copysign(float(kept << dropped), value)


def argument_layout(signature):
    """Returns the `struct` layout of the arguments of `signature` and the offset of each argument. A scalar takes
    an 8-byte slot, and a tensor map takes 128 bytes at a multiple of 64 bytes."""
    layout, offsets = "=", []
    for code in signature:
        alignment = TENSOR_MAP_ALIGNMENT if code == TENSOR_MAP_CODE else SLOT_BYTES
        padding = -struct.calcsize(layout) % alignment
        layout += f"{padding}x" * (padding > 0)
        offsets.append(struct.calcsize(layout))
        layout += SLOT_FORMATS[code]
    return struct.Struct(layout), tuple(offsets)


@functools.lru_cache(maxsize=MAX_LAUNCH_CONFIGS)
def launch_config(grid, block, shared_bytes, stream, cluster, pdl):
    """Returns the `cuLaunchKernelEx` configuration of a launch. `grid` counts work groups, and the driver rejects a
    `grid` whose dimensions are not multiples of the `cluster` dimensions. With `pdl`, the launch lets the kernel start
    before the previous kernel in the stream completes."""
    attributes = []
    if cluster is not None:
        attribute = driver.CUlaunchAttribute()
        attribute.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attribute.value.clusterDim.x, attribute.value.clusterDim.y, attribute.value.clusterDim.z = cluster
        attributes.append(attribute)
    if pdl:
        attribute = driver.CUlaunchAttribute()
        attribute.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        attribute.value.programmaticStreamSerializationAllowed = 1
        attributes.append(attribute)
    config = driver.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = grid
    config.blockDimX, config.blockDimY, config.blockDimZ = block
    config.sharedMemBytes = shared_bytes
    config.hStream = driver.CUstream(stream)
    config.attrs = attributes
    config.numAttrs = len(attributes)
    return config


class KernelCall:
    """`cuLaunchKernel` and `cuLaunchKernelEx` copy the arguments before they return, so each thread reuses one
    argument buffer."""

    def __init__(self, kernel, signature):
        verify_parameters(kernel, signature)
        self.handle = kernel.handle
        self.function = driver.CUfunction(int(kernel.handle))
        self.signature = signature
        self.layout, self.offsets = argument_layout(signature)
        self.float_slots = tuple(slot for slot, code in enumerate(signature) if code in FLOAT_CODES)
        self.tensor_map_slots = tuple(slot for slot, code in enumerate(signature) if code == TENSOR_MAP_CODE)
        self.buffers = threading.local()

    def argument_buffer(self):
        storage = ctypes.create_string_buffer(self.layout.size + TENSOR_MAP_ALIGNMENT)
        start = -ctypes.addressof(storage) % TENSOR_MAP_ALIGNMENT
        base = ctypes.addressof(storage) + start
        pointers = (ctypes.c_void_p * len(self.signature))(*(base + offset for offset in self.offsets))
        self.buffers.value = storage, start, ctypes.addressof(pointers), pointers
        return self.buffers.value

    def __call__(self, grid, threads, shared_bytes, values, stream, cluster=None, pdl=False):
        """Launches the kernel with `cuLaunchKernel`, or with `cuLaunchKernelEx` when the launch names `cluster`
        dimensions or asks for programmatic dependent launch (`pdl`)."""
        storage, start, address, _ = getattr(self.buffers, "value", None) or self.argument_buffer()
        for slot in self.float_slots:
            if not isinstance(values[slot], (float, np.floating)):
                raise TypeError(f"launch signature {self.signature!r} gives the value {values[slot]!r} the code 'f'")
        for slot in self.tensor_map_slots:
            if not isinstance(values[slot], bytes) or len(values[slot]) != TENSOR_MAP_BYTES:
                raise TypeError(f"launch signature {self.signature!r} needs {TENSOR_MAP_BYTES} bytes for the code "
                                f"{TENSOR_MAP_CODE!r}, got {values[slot]!r}")
        try:
            self.layout.pack_into(storage, start, *values)
        except struct.error as error:
            raise TypeError(f"the values {values!r} do not fit the launch signature {self.signature!r}") from error
        block = (threads, 1, 1) if isinstance(threads, int) else threads
        if cluster is None and not pdl:
            status, = driver.cuLaunchKernel(self.handle, *grid, *block, shared_bytes, stream, address, 0)
            if status:
                raise RuntimeError(f"cuLaunchKernel failed: {status}")
            return
        config = launch_config(tuple(grid), tuple(block), shared_bytes, int(stream),
                               None if cluster is None else tuple(cluster), bool(pdl))
        status, = driver.cuLaunchKernelEx(config, self.function, address, 0)
        if status:
            raise RuntimeError(f"cuLaunchKernelEx failed: {status}")


@functools.cache
def kernel_call(kernel, signature):
    return KernelCall(kernel, signature)


def run(kernel, grid, threads, shared_bytes, signature, values, stream, cluster=None, pdl=False):
    call = kernel_call(kernel, signature)
    call(grid, threads, shared_bytes, values, stream_handle(stream), cluster=cluster, pdl=pdl)


@functools.cache
def static_shared_bytes(kernel, device_id):
    attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
    status, value = driver.cuKernelGetAttribute(attribute, kernel.handle, device_id)
    if status:
        raise RuntimeError(f"cuKernelGetAttribute failed: {status}")
    return value


@functools.cache
def allow_dynamic_shared(kernel, device_id):
    """Sets the dynamic shared memory limit of `kernel` to the most that a work group can hold and returns the limit."""
    most = max_shared_bytes(device_id) - static_shared_bytes(kernel, device_id)
    attribute = driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    status, = driver.cuKernelSetAttribute(attribute, most, kernel.handle, device_id)
    if status:
        raise RuntimeError(f"cuKernelSetAttribute failed: {status}")
    return most


@functools.cache
def resident_work_groups(kernel, threads, shared_bytes):
    return kernel.occupancy.max_active_blocks_per_multiprocessor(threads, shared_bytes)


def shared_bytes_for(kernel, device_id, work_groups):
    """Returns the dynamic shared memory per work group that limits an SM to `work_groups` work groups of `kernel`."""
    per_work_group = shared_bytes_per_sm(device_id) // work_groups // SHARED_ALLOCATION_BYTES * SHARED_ALLOCATION_BYTES
    return min(allow_dynamic_shared(kernel, device_id),
               per_work_group - reserved_shared_bytes(device_id) - static_shared_bytes(kernel, device_id))
