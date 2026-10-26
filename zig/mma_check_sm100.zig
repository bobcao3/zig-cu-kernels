//! Small `tcgen05.mma` FP8 kernels whose results the tests compare with PyTorch.

const builtin = @import("builtin");
const std = @import("std");
const nvptx = std.nvptx;
const cluster = @import("cluster.zig");
const configurations = @import("mma_check_sm100_configurations.zig");
const tcgen05 = @import("tcgen05.zig");
const tma = @import("tma.zig");

const Mbarrier = nvptx.Mbarrier;
const TensorMap = tma.TensorMap;
const threads = configurations.threads;
const k = configurations.k;
const n = configurations.n;
const swizzle = 128;
const steps = k / tcgen05.k_step_bytes;
const columns = 512;
const wait_limit_ns = 1_000_000_000;

const kernel: std.builtin.CallingConvention = .{
    .nvptx_kernel = .{ .max_work_group_size = .{ .x = threads } },
};

const Shared = struct {
    var barriers: [2]Mbarrier addrspace(.shared) = undefined;
    var tensor_base: u32 addrspace(.shared) = undefined;
    extern var storage: [0]u8 align(16) addrspace(.shared);
};

/// Traps when the phase does not complete within one second.
pub fn waitPhase(barrier: *addrspace(.shared) Mbarrier, phase: u32) void {
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitPhase(phase, null, null)) {
        if (nvptx.globaltimer() - start > wait_limit_ns) @trap();
    }
}

pub fn tileBase(storage: *addrspace(.shared) [0]u8) u32 {
    const alignment = configurations.tile_alignment;
    return std.mem.alignForward(u32, tma.sharedAddress(storage), alignment);
}

/// Each warp reads its 32 tensor memory lanes, one accumulator row per thread.
fn storeRows(tensor: u32, d: [*]addrspace(.global) f32, row_base: u32) void {
    const warp = @workItemId(0) / 32;
    const row = row_base + @workItemId(0);
    inline for (0..n / 32) |chunk| {
        const lane_address = tcgen05.address(warp * 32, chunk * 32);
        const values = tcgen05.load(f32, .@"32x32b", 32, .none, tensor + lane_address);
        tcgen05.waitSyncAligned(.ld);
        const array: [32]f32 = values;
        for (array, 0..) |value, column| d[row * n + chunk * 32 + column] = value;
    }
}

fn initBarriers() void {
    if (@workItemId(0) == 0) {
        Shared.barriers[0].init(1);
        Shared.barriers[1].init(1);
        nvptx.fenceMbarrierInit();
    }
}

fn checkOne(
    comptime scaled: bool,
    a_map: *const TensorMap,
    b_map: *const TensorMap,
    scales: [*]addrspace(.global) const u32,
    d: [*]addrspace(.global) f32,
) void {
    const thread = @workItemId(0);
    const warp = thread / 32;
    const a_tile = tileBase(&Shared.storage);
    const b_tile = a_tile + 128 * k;
    const scale_tile = b_tile + n * k;
    const loaded = &Shared.barriers[0];
    const done = &Shared.barriers[1];
    initBarriers();
    if (scaled) {
        const words: [*]addrspace(.shared) u32 = @ptrFromInt(scale_tile);
        for (0..3) |part| words[thread + part * threads] = scales[thread + part * threads];
        nvptx.fenceProxyAsync(.shared);
    }
    if (warp == 0) {
        tcgen05.allocate(.one, &Shared.tensor_base, columns);
        tcgen05.relinquishAllocPermit(.one);
    }
    tcgen05.fence(.before_thread_sync);
    nvptx.barrierSyncAligned(.work_group, null);
    tcgen05.fence(.after_thread_sync);
    const tensor = Shared.tensor_base;
    if (thread == 0) {
        _ = loaded.arriveExpectTx((128 + n) * k);
        tma.load(2, a_tile, a_map, .{ 0, 0 }, loaded, null);
        tma.load(2, b_tile, b_map, .{ 0, 0 }, loaded, null);
        waitPhase(loaded, 0);
        const a = tcgen05.kMajorDescriptor(a_tile, swizzle);
        const b = tcgen05.kMajorDescriptor(b_tile, swizzle);
        const shape: tcgen05.Shape = .{ .cta_group = .one, .m = 128, .n = n };
        const a_offsets = comptime tcgen05.kMajorOffsets(steps, 128, swizzle);
        const b_offsets = comptime tcgen05.kMajorOffsets(steps, n, swizzle);
        if (scaled) {
            const scale_a = tensor + n;
            const scale_b = scale_a + 4;
            const atom = configurations.scale_atom_bytes;
            inline for (.{ scale_a, scale_b, scale_b + 4 }, 0..) |target, index| {
                const source =
                    tcgen05.sharedDescriptor(scale_tile + @as(u32, index) * atom, .scale_atom);
                tcgen05.copyWarpx4(.one, target, source);
            }
            const instructions = comptime tcgen05.mxf8f6f4StepDescriptors(shape, steps);
            tcgen05.mmaBlockScaledSteps(
                .one,
                &a_offsets,
                &b_offsets,
                &instructions,
                tensor,
                a,
                b,
                scale_a,
                scale_b,
                false,
            );
        } else {
            const instructions: [steps]u32 = comptime @splat(tcgen05.f8f6f4Descriptor(shape));
            tcgen05.mmaSteps(.one, &a_offsets, &b_offsets, &instructions, tensor, a, b, false);
        }
        tcgen05.commit(.one, done);
    }
    waitPhase(done, 0);
    tcgen05.fence(.after_thread_sync);
    storeRows(tensor, d, 0);
    tcgen05.fence(.before_thread_sync);
    nvptx.barrierSyncAligned(.work_group, null);
    if (warp == 0) {
        tcgen05.fence(.after_thread_sync);
        tcgen05.deallocate(.one, tensor, columns);
    }
}

fn plain(
    a_map: TensorMap,
    b_map: TensorMap,
    d: [*]addrspace(.global) f32,
) callconv(kernel) void {
    checkOne(false, &a_map, &b_map, undefined, d);
}

fn blockScaled(
    a_map: TensorMap,
    b_map: TensorMap,
    scales: [*]addrspace(.global) const u32,
    d: [*]addrspace(.global) f32,
) callconv(kernel) void {
    checkOne(true, &a_map, &b_map, scales, d);
}

/// CTA 0 issues the M256 MMAs of the pair. Each CTA loads 128 rows of A and of B and reads its
/// 128 accumulator rows.
fn pair(
    a_map: TensorMap,
    b_map: TensorMap,
    d: [*]addrspace(.global) f32,
) callconv(kernel) void {
    const thread = @workItemId(0);
    const warp = thread / 32;
    const rank = cluster.rank();
    const a_tile = tileBase(&Shared.storage);
    const b_tile = a_tile + 128 * k;
    const loaded = &Shared.barriers[0];
    const done = &Shared.barriers[1];
    initBarriers();
    if (warp == 0) {
        tcgen05.allocate(.two, &Shared.tensor_base, columns);
        tcgen05.relinquishAllocPermit(.two);
    }
    tcgen05.fence(.before_thread_sync);
    cluster.sync();
    tcgen05.fence(.after_thread_sync);
    const tensor = Shared.tensor_base;
    if (thread == 0) {
        const leader_loaded = cluster.leaderAddress(loaded);
        if (rank == 0) _ = loaded.arriveExpectTx(2 * 2 * 128 * k);
        const row: i32 = @intCast(rank * 128);
        tma.loadPair(2, a_tile, &a_map, .{ 0, row }, leader_loaded, null);
        tma.loadPair(2, b_tile, &b_map, .{ 0, row }, leader_loaded, null);
        if (rank == 0) {
            waitPhase(loaded, 0);
            const a = tcgen05.kMajorDescriptor(a_tile, swizzle);
            const b = tcgen05.kMajorDescriptor(b_tile, swizzle);
            const shape: tcgen05.Shape = .{ .cta_group = .two, .m = 256, .n = n };
            const offsets = comptime tcgen05.kMajorOffsets(steps, 128, swizzle);
            const instructions: [steps]u32 = comptime @splat(tcgen05.f8f6f4Descriptor(shape));
            tcgen05.mmaSteps(.two, &offsets, &offsets, &instructions, tensor, a, b, false);
            tcgen05.commitMulticast(.two, done, 0b11);
        }
    }
    waitPhase(done, 0);
    tcgen05.fence(.after_thread_sync);
    storeRows(tensor, d, rank * 128);
    tcgen05.fence(.before_thread_sync);
    cluster.sync();
    if (warp == 0) {
        tcgen05.fence(.after_thread_sync);
        tcgen05.deallocate(.two, tensor, columns);
    }
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        @export(&plain, .{ .name = configurations.dense_name });
        @export(&blockScaled, .{ .name = configurations.scaled_name });
        @export(&pair, .{ .name = configurations.pair_name });
    }
}
