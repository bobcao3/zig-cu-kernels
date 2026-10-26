//! Peak-rate kernels for `tcgen05.mma.cta_group::2` M256 N256 K32 on FP8 operands that stay in
//! shared memory. Thread 0 of CTA 0 issues `batches` batches of MMAs and waits for a batch to
//! complete before it issues the batch after the next one.

const builtin = @import("builtin");
const std = @import("std");
const nvptx = std.nvptx;
const cluster = @import("cluster.zig");
const configurations = @import("mma_rate_sm100_configurations.zig");
const tcgen05 = @import("tcgen05.zig");
const tma = @import("tma.zig");

const Mbarrier = nvptx.Mbarrier;
const threads = configurations.threads;
const stages = configurations.stages;
const k = configurations.k;
const stage_bytes = configurations.stage_bytes;
const n = 256;
const swizzle = 128;
const columns = 512;
const scale_columns = 16;
const wait_limit_ns = 1_000_000_000;

const kernel: std.builtin.CallingConvention = .{
    .nvptx_kernel = .{ .max_work_group_size = .{ .x = threads } },
};

const Shared = struct {
    var barriers: [2]Mbarrier addrspace(.shared) = undefined;
    var tensor_base: u32 addrspace(.shared) = undefined;
    extern var storage: [0]u8 align(16) addrspace(.shared);
};

fn waitPhase(barrier: *addrspace(.shared) Mbarrier, phase: u32) void {
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitPhase(phase, null, null)) {
        if (nvptx.globaltimer() - start > wait_limit_ns) @trap();
    }
}

fn waitBatch(batch: u32) void {
    waitPhase(&Shared.barriers[batch & 1], batch >> 1 & 1);
}

fn fill(
    target: u32,
    source: [*]addrspace(.global) const @Vector(4, u32),
    comptime vectors: u32,
) void {
    const words: [*]addrspace(.shared) @Vector(4, u32) = @ptrFromInt(target);
    var index = @workItemId(0);
    while (index < vectors) : (index += threads) words[index] = source[index];
}

fn issue(comptime scaled: bool, tile_base: u32, tensor: u32, batches: u32) void {
    const shape: tcgen05.Shape = .{ .cta_group = .two, .m = 256, .n = n };
    const offsets = comptime tcgen05.kMajorOffsets(configurations.steps, 128, swizzle);
    const dense_instructions: [configurations.steps]u32 =
        comptime @splat(tcgen05.f8f6f4Descriptor(shape));
    const scaled_instructions =
        comptime tcgen05.mxf8f6f4StepDescriptors(shape, configurations.steps);
    var batch: u32 = 0;
    while (batch < batches) : (batch += 1) {
        if (batch >= 2) waitBatch(batch - 2);
        inline for (0..stages) |stage| {
            const a_tile = tile_base + @as(u32, stage) * stage_bytes;
            const a = tcgen05.kMajorDescriptor(a_tile, swizzle);
            const b = tcgen05.kMajorDescriptor(a_tile + 128 * k, swizzle);
            if (scaled) {
                const scale_a = tensor + n + @as(u32, stage) * scale_columns;
                tcgen05.mmaBlockScaledSteps(
                    .two,
                    &offsets,
                    &offsets,
                    &scaled_instructions,
                    tensor,
                    a,
                    b,
                    scale_a,
                    scale_a + 4,
                    true,
                );
            } else {
                tcgen05.mmaSteps(.two, &offsets, &offsets, &dense_instructions, tensor, a, b, true);
            }
        }
        tcgen05.commit(.two, &Shared.barriers[batch & 1]);
    }
    if (batches >= 2) waitBatch(batches - 2);
    if (batches >= 1) waitBatch(batches - 1);
}

fn rate(
    comptime scaled: bool,
    values: [*]addrspace(.global) const @Vector(4, u32),
    scales: [*]addrspace(.global) const @Vector(4, u32),
    batches: u32,
) void {
    const thread = @workItemId(0);
    const warp = thread / 32;
    const rank = cluster.rank();
    const address = tma.sharedAddress(&Shared.storage);
    const tile_base = std.mem.alignForward(u32, address, 1024);
    const scale_base = tile_base + stages * stage_bytes;
    const value_vectors = stages * stage_bytes / 16;
    const scale_vectors = stages * configurations.scale_stage_bytes / 16;
    fill(tile_base, values + rank * value_vectors, value_vectors);
    if (scaled) fill(scale_base, scales + rank * scale_vectors, scale_vectors);
    nvptx.fenceProxyAsync(.shared);
    if (thread == 0) {
        Shared.barriers[0].init(1);
        Shared.barriers[1].init(1);
        nvptx.fenceMbarrierInit();
    }
    if (warp == 0) {
        tcgen05.allocate(.two, &Shared.tensor_base, columns);
        tcgen05.relinquishAllocPermit(.two);
    }
    tcgen05.fence(.before_thread_sync);
    cluster.sync();
    tcgen05.fence(.after_thread_sync);
    const tensor = Shared.tensor_base;
    if (rank == 0 and thread == 0) {
        if (scaled) {
            for (0..stages * 3) |atom_index| {
                const atom: u32 = @intCast(atom_index);
                const stage = atom / 3;
                const target = tensor + n + stage * scale_columns + atom % 3 * 4;
                const source = scale_base + atom * 512;
                tcgen05.copyWarpx4(.two, target, tcgen05.sharedDescriptor(source, .scale_atom));
            }
        }
        issue(scaled, tile_base, tensor, batches);
    }
    tcgen05.fence(.before_thread_sync);
    cluster.sync();
    if (warp == 0) {
        tcgen05.fence(.after_thread_sync);
        tcgen05.deallocate(.two, tensor, columns);
    }
}

fn plain(
    values: [*]addrspace(.global) const @Vector(4, u32),
    batches: u32,
) callconv(kernel) void {
    rate(false, values, undefined, batches);
}

fn blockScaled(
    values: [*]addrspace(.global) const @Vector(4, u32),
    scales: [*]addrspace(.global) const @Vector(4, u32),
    batches: u32,
) callconv(kernel) void {
    rate(true, values, scales, batches);
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        @export(&plain, .{ .name = configurations.dense_name });
        @export(&blockScaled, .{ .name = configurations.scaled_name });
    }
}
