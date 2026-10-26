import functools

import numpy as np
import pytest
import torch
from cuda.bindings import driver
from cuda.core import system

from checks import queue_equal_check
from conftest import cuda_seed, declare_gpu_work
from zig_cu_kernels import axpby as axpby_launcher
from zig_cu_kernels import fp8_gemm as fp8_launcher
from zig_cu_kernels import gemm as gemm_launcher
from zig_cu_kernels import rms_norm as rms_norm_launcher
from zig_cu_kernels import runtime, tensormap
from zig_cu_kernels.torch import axpby, gemm, quantize_fp8_rows, rms_norm

# No other test calls axpby on this many elements, so the launch plan of this count takes the kernel that the test
# loads.
UNSHARED_COUNT = 2053
STREAM_ATTEMPTS = 64
PARAMETER_COPY = "launch_test_copy_parameters"
LOCAL_BYTES_LIMITS = {"gemm_store_bf16_192x192x32": 24, "gemm_gated_192x192x32": 24,
                      "rms_norm_bf16_weight_bf16_to_fp8_rowwise_c4_max32768": 8,
                      "rms_norm_bf16_gated_scale_shift_to_fp8_rowwise_c4_max32768": 8}
# The kernels of the bulk engine, which a caller selects with `ZIG_CU_KERNELS_RMS_NORM_ENGINE`, spill registers on
# compute capability 9.0 and up.
OPT_IN_SUFFIXES = ("_bulk",)


@pytest.mark.skipif("-Dkernel-optimize=safe" in runtime.read_manifest().options.split(),
                    reason="the safety checks of a safe build spill registers to local memory")
def test_kernels_use_no_local_memory():
    _, entries = runtime.manifest()
    names = [name for name in entries if runtime.is_available(name) and not name.endswith(OPT_IN_SUFFIXES)]
    assert names
    for name in names:
        kernel, _ = runtime.load(name)
        assert kernel.attributes.local_size_bytes <= LOCAL_BYTES_LIMITS.get(name, 0), f"{name} uses local memory"


def test_launcher_coefficients_round_to_fp32():
    assert runtime.fp32("alpha", 0.1) == float(np.float32(0.1))
    assert runtime.fp32("alpha", 2**24 + 1) == 2.0**24
    with pytest.raises(OverflowError):
        runtime.fp32("alpha", 3.5e38)
    for value in (True, "1", None, torch.tensor(1.0)):
        with pytest.raises(TypeError):
            runtime.fp32("alpha", value)


def test_a_cubin_that_the_driver_cannot_load_raises_until_it_is_replaced(monkeypatch, tmp_path):
    cuda_seed(1)
    declare_gpu_work(3)
    name = "axpby_f32"
    intact = runtime.cubin_path(runtime.info(name))
    broken = tmp_path / intact.name
    broken.write_bytes(b"not a cubin")
    cubins = {"path": broken}
    monkeypatch.setattr(runtime, "cubin_path", lambda entry: cubins["path"])
    monkeypatch.setattr(runtime, "load_for", functools.cache(runtime.load_for.__wrapped__))
    x, y = (torch.randint(-8, 9, (UNSHARED_COUNT,), device="cuda").float() for _ in range(2))
    with pytest.raises(RuntimeError) as raised:
        axpby(2.0, x, -1.0, y)
    cuda_version = "CUDA " + ".".join(map(str, system.get_user_mode_driver_version()))
    for part in (name, str(broken), cuda_version, "CUDA_ERROR_INVALID_IMAGE"):
        assert part in str(raised.value), (part, str(raised.value))
    cubins["path"] = intact
    queue_equal_check(axpby(2.0, x, -1.0, y), 2 * x - y, "result once the cubin loads")


def test_stream_wrappers_follow_a_stream_handle_into_another_context():
    """`runtime.stream_from_handle` returns a stream of the context in which the driver reused a stream handle."""
    declare_gpu_work(0)
    checked = gemm_launcher.checked
    device = runtime.current_device()
    primary = int(checked(driver.cuCtxGetCurrent()))
    created = checked(driver.cuCtxCreate(driver.CUctxCreateParams(), 0, checked(driver.cuDeviceGet(device.device_id))))
    streams = []
    try:
        handle = int(checked(driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING)))
        assert int(runtime.stream_from_handle(handle).context.handle) == int(created)
        checked(driver.cuStreamDestroy(handle))
        checked(driver.cuCtxSetCurrent(primary))
        while len(streams) < STREAM_ATTEMPTS and handle not in streams:
            streams.append(int(checked(driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING))))
        if handle not in streams:
            pytest.skip(f"no new stream of {STREAM_ATTEMPTS} got the handle of the destroyed stream")
        assert int(runtime.stream_from_handle(handle).context.handle) == primary
    finally:
        for stream in streams:
            checked(driver.cuStreamDestroy(stream))
        checked(driver.cuCtxDestroy(created))
        device.set_current()


class ArrayInterface:
    """Exports the memory at `address` through the CUDA Array Interface only, as contiguous elements of `typestr`."""

    def __init__(self, address, shape, typestr):
        self.__cuda_array_interface__ = {"shape": shape, "typestr": typestr, "data": (address, False), "version": 3}


def exported(tensor, typestr):
    return ArrayInterface(tensor.data_ptr(), tuple(tensor.shape), typestr)


class HostMemory:
    """Exports pageable host memory through the CUDA Array Interface and keeps the arrays alive."""

    def __init__(self):
        self.arrays = []

    def __call__(self, shape, typestr):
        array = np.zeros(shape, np.dtype(typestr.replace("V", "u")))
        self.arrays.append(array)
        return ArrayInterface(array.ctypes.data, shape, typestr)


def elsewhere(name):
    return f"{name} must be on device {torch.cuda.current_device()}, the current device, not in host memory"


def in_pageable_memory(name):
    return f"{name} must be in device memory, not in pageable host memory"


def test_launchers_read_exported_device_memory_and_name_operands_in_host_memory():
    """The BF16 GEMM, axpby and RMS norm launchers read exported device memory with the bits of the torch ops and
    raise ValueError naming an operand in host memory or in a CPU tensor."""
    cuda_seed(2)
    declare_gpu_work(6, dram_bytes=1 << 20)
    stream = torch.cuda.current_stream().cuda_stream
    host = HostMemory()
    a, b = torch.randn(8, 64, device="cuda").bfloat16(), torch.randn(64, 64, device="cuda").bfloat16()
    out = torch.empty(8, 64, device="cuda", dtype=torch.bfloat16)
    gemm_launcher.gemm(exported(a, "<V2"), b, out, stream=stream)
    queue_equal_check(out, gemm(a, b), "gemm with an exported a")
    with pytest.raises(ValueError, match=in_pageable_memory("a")):
        gemm_launcher.gemm(host((8, 64), "<V2"), b, out, stream=stream)
    with pytest.raises(ValueError, match=elsewhere("b")):
        gemm_launcher.gemm(a, b.cpu(), out, stream=stream)
    x, y = torch.randn(4096, device="cuda"), torch.randn(4096, device="cuda")
    total = torch.empty_like(x)
    axpby_launcher.axpby(2.0, exported(x, "<f4"), -1.0, y, total, element="f32", stream=stream)
    queue_equal_check(total, axpby(2.0, x, -1.0, y), "axpby with an exported x")
    with pytest.raises(ValueError, match=in_pageable_memory("y")):
        axpby_launcher.axpby(2.0, x, -1.0, host((4096,), "<f4"), total, element="f32", stream=stream)
    pinned = x.cpu().pin_memory()
    with pytest.raises(ValueError, match=elsewhere("x")):
        axpby_launcher.axpby(2.0, pinned, -1.0, y, total, element="f32", stream=stream)
    with pytest.raises(ValueError, match=elsewhere("x")):
        axpby_launcher.axpby(2.0, exported(pinned, "<f4"), -1.0, y, total, element="f32", stream=stream)
    h, weight = torch.randn(4, 64, device="cuda").bfloat16(), torch.randn(64, device="cuda").bfloat16()
    normed = torch.empty_like(h)
    options = rms_norm_launcher.Options(weight="bf16")
    rms_norm_launcher.rms_norm(options, h, normed, epsilon=1e-6, weight=exported(weight, "<V2"), stream=stream)
    queue_equal_check(normed, rms_norm(h, weight, 1e-6), "rms_norm with an exported weight")
    with pytest.raises(ValueError, match=in_pageable_memory("weight")):
        rms_norm_launcher.rms_norm(options, h, normed, epsilon=1e-6, weight=host((64,), "<V2"), stream=stream)
    with pytest.raises(ValueError, match=elsewhere("y")):
        rms_norm_launcher.rms_norm(options, h, normed.cpu(), epsilon=1e-6, weight=weight, stream=stream)


@pytest.mark.skipif(runtime.unavailable_reason("fp8_quantize_rows_f32") is not None,
                    reason=runtime.unavailable_reason("fp8_quantize_rows_f32") or "")
def test_quantizer_launcher_reads_exported_memory_and_names_operands_in_host_memory():
    """The FP8 row quantizer launcher reads exported memory with the bits of the torch op, takes an empty input at
    address 0 and raises ValueError naming an operand in host memory, in a CPU tensor or in a NumPy array."""
    cuda_seed(3)
    declare_gpu_work(3, dram_bytes=1 << 16)
    stream = torch.cuda.current_stream().cuda_stream
    x = torch.randn(4, 128, device="cuda")
    codes = torch.empty(4, 128, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, 4, device="cuda").t()
    fp8_launcher.quantize_rows(exported(x, "<f4"), codes, scales, stream=stream)
    expected_codes, expected_scales = quantize_fp8_rows(x)
    queue_equal_check(codes, expected_codes, "codes of an exported x")
    queue_equal_check(scales, expected_scales, "scales of an exported x")
    fp8_launcher.quantize_rows(ArrayInterface(0, (0, 128), "<f4"), codes[:0], scales[:0], stream=stream)
    with pytest.raises(ValueError, match=in_pageable_memory("x")):
        fp8_launcher.quantize_rows(HostMemory()((4, 128), "<f4"), codes, scales, stream=stream)
    with pytest.raises(ValueError, match=elsewhere("codes")):
        fp8_launcher.quantize_rows(x, codes.cpu(), scales, stream=stream)
    with pytest.raises(ValueError, match=elsewhere("scales")):
        fp8_launcher.quantize_rows(x, codes, np.zeros((4, 1), np.float32), stream=stream)


def test_targets_with_suffixes_run_on_their_own_devices_or_family():
    assert [runtime.target_capability(target) for target in ("sm_89", "sm_90a", "sm_100f", "sm_120f")] == [
        (8, 9), (9, 0), (10, 0), (12, 0)]
    devices = ((8, 9), (9, 0), (10, 0), (10, 3), (11, 0), (12, 0), (12, 1))
    runs = {target: [device for device in devices if runtime.target_runs_on(target, *device)]
            for target in ("sm_89", "sm_90", "sm_90a", "sm_100a", "sm_100f", "sm_120f")}
    assert runs == {"sm_89": [(8, 9)], "sm_90": [(9, 0)], "sm_90a": [(9, 0)], "sm_100a": [(10, 0)],
                    "sm_100f": [(10, 0), (10, 3)], "sm_120f": [(12, 0), (12, 1)]}


def test_argument_buffers_align_tensor_maps_to_64_bytes():
    layout, offsets = runtime.argument_layout("umpu")
    assert offsets == (0, 64, 192, 200)
    assert layout.size == 208
    assert runtime.argument_layout("pqu")[1] == (0, 8, 16)


class EncodedMap:
    def __init__(self, words):
        self.opaque = words


def test_tensor_maps_list_dimensions_innermost_first_and_cache_encodings(monkeypatch):
    calls = []

    def encode(*arguments):
        calls.append(arguments)
        return driver.CUresult.CUDA_SUCCESS, EncodedMap(list(range(len(calls), len(calls) + 16)))

    monkeypatch.setattr(driver, "cuTensorMapEncodeTiled", encode)
    tensormap.encoded.cache_clear()
    values = torch.empty(4, 384, 256, dtype=torch.float8_e4m3fn)[:, :320]
    encoding = tensormap.tensor_map(values, (1, 128, 64), swizzle=64)
    assert len(encoding) == runtime.TENSOR_MAP_BYTES
    assert np.frombuffer(encoding, "<u8").tolist() == list(range(1, 17))
    assert tensormap.tensor_map(values, (1, 128, 64), swizzle=64) is encoding
    data_type, rank, pointer, shape, strides, box, element_strides, _, swizzle, _, _ = calls[0]
    assert (data_type, rank, pointer) == (tensormap.DATA_TYPES[torch.uint8], 3, values.data_ptr())
    assert [int(extent) for extent in shape] == [256, 320, 4]
    assert [int(stride) for stride in strides] == [256, 384 * 256]
    assert [int(extent) for extent in box] == [64, 128, 1]
    assert [int(stride) for stride in element_strides] == [1, 1, 1]
    assert swizzle == tensormap.SWIZZLES[64]
    scales = torch.empty(512, 64, dtype=torch.float32)
    tensormap.tensor_map(scales, (32, 32), swizzle=128)
    assert len(calls) == 2
    assert [int(stride) for stride in calls[1][4]] == [64 * 4]
    for tensor, box in ((scales.t(), (32, 32)), (scales[0], (32,)), (scales.double(), (32, 32))):
        with pytest.raises(ValueError):
            tensormap.tensor_map(tensor, box)
    assert len(calls) == 2
    tensormap.encoded.cache_clear()


@pytest.mark.skipif(runtime.compute_capability(torch.cuda.current_device()) < 90,
                    reason="the driver encodes tensor maps on compute capability 9.0 and up")
def test_the_driver_encodes_tensor_maps():
    tensormap.encoded.cache_clear()
    values = torch.empty(256, 512, dtype=torch.bfloat16, device="cuda")
    encoding = tensormap.tensor_map(values, (64, 64), swizzle=128)
    assert len(encoding) == runtime.TENSOR_MAP_BYTES
    assert tensormap.tensor_map(values, (64, 64), swizzle=128) is encoding
    with pytest.raises(ValueError):
        tensormap.tensor_map(values, (512, 64))
    tensormap.encoded.cache_clear()


@pytest.mark.skipif(runtime.unavailable_reason(PARAMETER_COPY) is not None,
                    reason=runtime.unavailable_reason(PARAMETER_COPY) or "")
def test_kernels_receive_tensor_maps_by_value_with_each_launch_path():
    """The parameter copy kernel writes its first `u32`, the 128 bytes of its tensor map and its last `u32` to memory,
    after a launch with `cuLaunchKernel` and after launches with `cuLaunchKernelEx`."""
    declare_gpu_work(3)
    kernel, _ = runtime.load(PARAMETER_COPY)
    stream = torch.cuda.current_stream().cuda_stream
    words = torch.empty(2 + runtime.TENSOR_MAP_BYTES // 4, dtype=torch.int32, device="cuda")
    capability = runtime.compute_capability(torch.cuda.current_device())
    launches = [{}, {"pdl": True}] + [{"cluster": (1, 1, 1)}] * (capability >= 90)
    for index, options in enumerate(launches):
        map_bytes = bytes((index * 37 + byte) % 256 for byte in range(runtime.TENSOR_MAP_BYTES))
        words.fill_(-1)
        runtime.run(kernel, (1, 1, 1), 32, 0, "umpu", (0x89ABCDEF, map_bytes, words.data_ptr(), index), stream,
                    **options)
        expected = np.concatenate([[0x89ABCDEF], np.frombuffer(map_bytes, "<u4"), [index]]).astype("<u4")
        assert words.cpu().numpy().view("<u4").tolist() == expected.tolist(), options
    if capability < 90:
        with pytest.raises(RuntimeError, match="CUDA_ERROR_INVALID_CLUSTER_SIZE"):
            runtime.run(kernel, (1, 1, 1), 32, 0, "umpu", (0, bytes(128), words.data_ptr(), 0), stream,
                        cluster=(1, 1, 1))
    with pytest.raises(TypeError):
        runtime.run(kernel, (1, 1, 1), 32, 0, "umpu", (0, bytes(64), words.data_ptr(), 0), stream)
