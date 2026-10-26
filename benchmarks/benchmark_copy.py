"""Measures the copy rate of the device, in bytes read plus bytes written per second, for the torch copy and the
kernels of `zig/copy_rate.zig`. The best rate is the copy ceiling of the RMS norms (README, section Benchmarks). The
`halve` kernels read 16 bytes for each 8 bytes they write, the mix of a norm with an FP8 output; the `read` kernels only
read."""

import argparse
import itertools

import torch

import dse
import timing
from zig_cu_kernels import runtime

KINDS = ("torch", "registers", "halve", "read", "cast", "cast_rounded", "bulk")
UNROLLS = (1, 2, 4, 8)
WORK_GROUPS_PER_SM = (1, 2, 4, 8)
TILE_KIB = (4, 8, 16, 32)
STAGES = (2, 4, 8)
REGISTER_THREADS = 256
CHUNK_BYTES = 16
NEVER = 0x9E3779B9
# The bytes that a kernel moves per byte of `source`.
MOVED_PER_SOURCE_BYTE = {"torch": 2, "registers": 2, "bulk": 2, "halve": 1.5, "cast": 1.5, "cast_rounded": 1.5, "read": 1}
KIB = 1024


def configurations(kind, arguments):
    if kind == "torch":
        return [()]
    if kind in ("registers", "halve", "read", "cast", "cast_rounded"):
        return list(itertools.product(arguments.unrolls, arguments.work_groups_per_sm))
    return list(itertools.product(arguments.tile_kib, arguments.stages, arguments.work_groups_per_sm))


def call_of(kind, configuration, source, target, device_id):
    if kind == "torch":
        return lambda: target.copy_(source), "torch"
    sms = runtime.num_sm(device_id)
    if kind in ("registers", "halve", "read", "cast", "cast_rounded"):
        unroll, per_sm = configuration
        kernel, _ = runtime.load_for(device_id, f"copy_rate_{kind}_unroll{unroll}")
        chunks = source.numel() // CHUNK_BYTES
        if kind == "read":
            call = runtime.kernel_call(kernel, "ppuu")
            values = (source.data_ptr(), target.data_ptr(), chunks, NEVER)
        elif kind in ("cast", "cast_rounded"):
            call = runtime.kernel_call(kernel, "ppuf")
            values = (source.data_ptr(), target.data_ptr(), chunks, 0.5)
        else:
            call = runtime.kernel_call(kernel, "ppu")
            values = (source.data_ptr(), target.data_ptr(), chunks)
        stream = torch.cuda.current_stream().cuda_stream
        return (lambda: call((sms * per_sm, 1, 1), REGISTER_THREADS, 0, values, stream),
                f"{kind} unroll={unroll} work_groups_per_sm={per_sm}")
    tile_kib, stages, per_sm = configuration
    tile_bytes = tile_kib * KIB
    kernel, _ = runtime.load_for(device_id, "copy_rate_bulk")
    shared_bytes = stages * tile_bytes
    if shared_bytes > runtime.allow_dynamic_shared(kernel, device_id):
        return None, None
    if runtime.resident_work_groups(kernel, 32, shared_bytes) < per_sm:
        return None, None
    call = runtime.kernel_call(kernel, "ppuuu")
    values = (source.data_ptr(), target.data_ptr(), source.numel() // tile_bytes, tile_bytes, stages)
    stream = torch.cuda.current_stream().cuda_stream
    return (lambda: call((sms * per_sm, 1, 1), 32, shared_bytes, values, stream),
            f"bulk tile={tile_kib}KiB stages={stages} work_groups_per_sm={per_sm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    parser.add_argument("--unrolls", nargs="+", type=int, default=UNROLLS)
    parser.add_argument("--work-groups-per-sm", nargs="+", type=int, default=WORK_GROUPS_PER_SM)
    parser.add_argument("--tile-kib", nargs="+", type=int, default=TILE_KIB)
    parser.add_argument("--stages", nargs="+", type=int, default=STAGES)
    parser.add_argument("--mib", type=int, default=timing.CEILING_BYTES // (KIB * KIB), help="bytes copied, in MiB")
    parser.add_argument("--check", action="store_true", help="compare the copy with its source once per kernel")
    arguments = parser.parse_args()
    device_id = runtime.current_device().device_id
    size = arguments.mib * KIB * KIB
    source = torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda")
    target = torch.empty_like(source)
    print(f"{torch.cuda.get_device_name()}: copying {arguments.mib} MiB, L2 {runtime.l2_bytes(device_id) >> 20} MiB",
          flush=True)
    for kind in arguments.kinds:
        for configuration in configurations(kind, arguments):
            call, name = call_of(kind, configuration, source, target, device_id)
            if call is None:
                continue
            if arguments.check and kind in ("torch", "registers", "bulk"):
                target.zero_()
                call()
                torch.cuda.synchronize()
                if not torch.equal(source, target):
                    raise SystemExit(f"{name} copied wrong bytes")
            moved = MOVED_PER_SOURCE_BYTE[kind] * size
            seconds = timing.sustained_seconds(call, sol_seconds=moved / 1e12)
            rate = moved / seconds
            print(f"{name:48} {seconds * 1e6:8.1f} us {rate * 1e-9:7.0f} GB/s {timing.clock_note()}", flush=True)
            dse.log("copy", f"{rate * 1e-9:.1f}", "GB/s", based_on=kind, axis="configuration", value=name)


if __name__ == "__main__":
    timing.run(main)
