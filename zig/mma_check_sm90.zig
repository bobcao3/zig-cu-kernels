//! Small FP8 `wgmma` kernels for comparison with a host product. A producer warpgroup copies `A`
//! and `B` tiles with 128-byte swizzle into a ring of stages. Each consumer warpgroup multiplies
//! its 64 rows of `A` by all of `B` and writes its accumulator fragment to global memory.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("mma_check_sm90_configurations.zig");
const cluster = @import("cluster.zig");
const pipeline = @import("pipeline.zig");
const tma = @import("tma.zig");
const wgmma = @import("wgmma.zig");

const Check = configurations.Check;
const k_block = configurations.k_block;
const stages = configurations.stages;

extern var shared_storage: [0]u8 align(1024) addrspace(.shared);

/// Traps when the barrier phase does not complete within `wait_limit_ns`, so that a protocol error
/// ends the launch instead of hanging it.
pub inline fn waitPhase(
    barrier: *addrspace(.shared) nvptx.Mbarrier,
    phase: u1,
    comptime order: ?nvptx.mbarrier.WaitOrder,
) void {
    if (barrier.tryWaitPhase(phase, order, null)) return;
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitPhase(phase, order, null)) {
        if (nvptx.globaltimer() -% start > configurations.wait_limit_ns) asm volatile ("trap;");
    }
}

fn Kernel(comptime check: Check) type {
    return struct {
        const n = check.n;
        const ctas = check.cluster_ctas;
        const scope: pipeline.Scope = if (ctas == 1) .cta else .cluster;
        const order: ?nvptx.mbarrier.WaitOrder = switch (scope) {
            .cta => null,
            .cluster => .{ .semantics = .acquire, .scope = .cluster },
        };
        const Ring = pipeline.Pipeline(stages, scope);
        const a_rows = 64 * check.consumers;
        const a_bytes = a_rows * k_block;
        const b_rows_per_cta = n / ctas;
        const stage_bytes = check.stageBytes();
        const options: std.lang.CallingConvention.NvptxKernelOptions = .{
            .required_work_group_size = .{ .x = check.threads() },
            .cluster_size = .{ .x = ctas },
        };

        var ring: Ring addrspace(.shared) = undefined;

        fn kernel(
            a_map: tma.TensorMap,
            b_map: tma.TensorMap,
            out: [*]addrspace(.global) f32,
            k_blocks: u32,
        ) callconv(.{ .nvptx_kernel = options }) void {
            const base = tma.sharedAddress(&shared_storage);
            const tiles = std.mem.alignForward(u32, base, 1024);
            const tid = @workItemId(0);
            if (tid == 0) ring.init(1, 4 * check.consumers * ctas);
            if (ctas == 1) nvptx.barrierSyncAligned(.work_group, null) else cluster.sync();
            if (tid < 128) {
                if (check.registers) |split| cluster.setMaxRegisters(.dec, split.producer);
                if (tid == 0) produce(tiles, &a_map, &b_map, k_blocks);
            } else {
                if (check.registers) |split| cluster.setMaxRegisters(.inc, split.consumer);
                consume(tiles, tid / 128 - 1, tid % 128, out, k_blocks);
            }
            if (ctas > 1) cluster.sync();
        }

        fn produce(
            tiles: u32,
            a_map: *const tma.TensorMap,
            b_map: *const tma.TensorMap,
            k_blocks: u32,
        ) void {
            const rank: u32 = if (ctas == 1) 0 else cluster.rank();
            const a_row: i32 = @intCast(@workGroupId(0) * a_rows);
            const b_row = rank * b_rows_per_cta;
            var slot: Ring.Slot = .{};
            var block: u32 = 0;
            while (block < k_blocks) : (block += 1) {
                waitPhase(ring.emptyBarrier(slot), ~slot.phase, order);
                ring.producerExpect(slot, stage_bytes);
                const k: i32 = @intCast(block * k_block);
                const tile = tiles + slot.index * stage_bytes;
                const full = ring.fullBarrier(slot);
                tma.load(2, tile, a_map, .{ k, a_row }, full, null);
                const b_tile = tile + a_bytes + b_row * k_block;
                if (ctas == 1) {
                    tma.load(2, b_tile, b_map, .{ k, 0 }, full, null);
                } else {
                    const everyone: u16 = (1 << ctas) - 1;
                    const coordinates: [2]i32 = .{ k, @intCast(b_row) };
                    tma.loadMulticast(2, b_tile, b_map, coordinates, full, everyone, null);
                }
                slot = slot.next();
            }
        }

        fn consume(
            tiles: u32,
            consumer: u32,
            thread: u32,
            out: [*]addrspace(.global) f32,
            k_blocks: u32,
        ) void {
            var accumulator: wgmma.Accumulator(n) = @splat(0);
            var slot: Ring.Slot = .{};
            var block: u32 = 0;
            while (block < k_blocks) : (block += 1) {
                waitPhase(ring.fullBarrier(slot), slot.phase, order);
                const tile = tiles + slot.index * stage_bytes;
                const a: wgmma.Descriptor = .kMajor(tile + consumer * 64 * k_block, .bytes_128);
                const b: wgmma.Descriptor = .kMajor(tile + a_bytes, .bytes_128);
                wgmma.fence();
                wgmma.mmaKSteps(n, 4, .bytes_128, 64, n, &accumulator, a, b, block != 0);
                wgmma.commit();
                wgmma.wait(0);
                wgmma.fenceOperand(&accumulator);
                if (thread % 32 == 0) {
                    if (ctas == 1) {
                        ring.consumerRelease(slot);
                    } else {
                        for (0..ctas) |target| ring.consumerReleaseRemote(slot, @intCast(target));
                    }
                }
                slot = slot.next();
            }
            // The accumulator fragment layout is that of PTX ISA 9.7.17.5.1.1.3.
            const warp = thread / 32;
            const lane = thread % 32;
            const row = @workGroupId(0) * a_rows + consumer * 64 + warp * 16 + lane / 4;
            inline for (0..n / 2) |index| {
                const fragment_row = row + 8 * (index / 2 % 2);
                const column = 8 * (index / 4) + 2 * (lane % 4) + index % 2;
                out[fragment_row * n + column] = accumulator[index];
            }
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (configurations.checks) |check| {
            @export(&Kernel(check).kernel, .{ .name = configurations.kernelName(check) });
        }
    }
}
