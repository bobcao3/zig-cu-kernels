//! One-tile kernels that check the FP8 `mma.sync` forms and the tensor map copies of compute
//! capability 12.x against host references.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("mma_check_sm120_configurations.zig");
const mma = @import("mma_sm120.zig");
const tma = @import("tma.zig");
const pipeline = @import("pipeline.zig");

const Global = [*]addrspace(.global) u32;
const ConstGlobal = [*]addrspace(.global) const u32;
const tile_bytes = configurations.tile_rows * configurations.tile_columns;
const tile_words = tile_bytes / 4;
const copy_threads = configurations.copy_threads;
const Ring = pipeline.Pipeline(configurations.stages, .cta);
const Mbarrier = nvptx.Mbarrier;

const Fragments = struct { a: mma.A, b: mma.B };

/// `a` holds 16 rows of 32 bytes and `b` holds 8 rows of 32 bytes (PTX ISA 9.7.16.5.10).
fn loadFragments(a: ConstGlobal, b: ConstGlobal, lane: u32) Fragments {
    const group = lane >> 2;
    const thread = lane & 3;
    return .{
        .a = .{
            a[group * 8 + thread],
            a[(group + 8) * 8 + thread],
            a[group * 8 + 4 + thread],
            a[(group + 8) * 8 + 4 + thread],
        },
        .b = .{ b[group * 8 + thread], b[group * 8 + 4 + thread] },
    };
}

fn storeTile(d: [*]addrspace(.global) f32, lane: u32, tile: mma.Accumulator) void {
    const row = lane >> 2;
    const column = (lane & 3) * 2;
    d[row * 8 + column] = tile[0];
    d[row * 8 + column + 1] = tile[1];
    d[(row + 8) * 8 + column] = tile[2];
    d[(row + 8) * 8 + column + 1] = tile[3];
}

const warp_kernel: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = 32 },
} };

fn checkF8f6f4(
    d: [*]addrspace(.global) f32,
    a: ConstGlobal,
    b: ConstGlobal,
) callconv(warp_kernel) void {
    const lane = @workItemId(0);
    const fragments = loadFragments(a, b, lane);
    storeTile(d, lane, mma.mma(fragments.a, fragments.b, @splat(0)));
}

fn CheckBlockScale(comptime index: usize) type {
    const s = configurations.selectors[index];
    return struct {
        fn kernel(
            d: [*]addrspace(.global) f32,
            a: ConstGlobal,
            b: ConstGlobal,
            scale_a: ConstGlobal,
            scale_b: ConstGlobal,
        ) callconv(warp_kernel) void {
            const lane = @workItemId(0);
            const fragments = loadFragments(a, b, lane);
            const options: mma.BlockScaleOptions = .{
                .a_byte = s.a_byte,
                .a_thread = s.a_thread,
                .b_byte = s.b_byte,
                .b_thread = s.b_thread,
            };
            const tile = mma.mmaBlockScaled(
                options,
                fragments.a,
                fragments.b,
                @splat(0),
                scale_a[lane],
                scale_b[lane],
            );
            storeTile(d, lane, tile);
        }
    };
}

fn waitBounded(barrier: *addrspace(.shared) Mbarrier, phase: u1) void {
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitParity(phase)) {
        if (nvptx.globaltimer() -% start > configurations.wait_limit_ns) @trap();
    }
}

fn copyOut(buffer: *align(1024) addrspace(.shared) [tile_bytes]u8, raw: Global) void {
    const words: *addrspace(.shared) const [tile_words]u32 = @ptrCast(buffer);
    var index: u32 = @workItemId(0);
    while (index < tile_words) : (index += copy_threads) raw[index] = words[index];
}

var ring: Ring addrspace(.shared) = undefined;
var tiles: [configurations.stages][tile_bytes]u8 align(1024) addrspace(.shared) = undefined;

const copy_kernel: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = copy_threads },
} };

/// Loads each tile of the source through the ring, writes the shared bytes of the tile to `raw` and
/// stores the tile back through `store_map`.
fn tmaCopy(
    load_map: tma.TensorMap,
    store_map: tma.TensorMap,
    raw: Global,
    row_tiles: u32,
    column_tiles: u32,
) callconv(copy_kernel) void {
    const leader = @workItemId(0) == 0;
    if (leader) {
        tma.prefetch(&load_map);
        ring.init(1, 1);
    }
    nvptx.barrierSyncAligned(.work_group, null);
    var slot: Ring.Slot = .{};
    var tile: u32 = 0;
    while (tile < row_tiles * column_tiles) : (tile += 1) {
        const coordinates: [2]i32 = .{
            @intCast(tile % column_tiles * configurations.tile_columns),
            @intCast(tile / column_tiles * configurations.tile_rows),
        };
        const buffer = &tiles[slot.index];
        if (leader) {
            ring.producerExpect(slot, tile_bytes);
            tma.load(2, buffer, &load_map, coordinates, ring.fullBarrier(slot), null);
        }
        waitBounded(ring.fullBarrier(slot), slot.phase);
        copyOut(buffer, raw + tile * tile_words);
        if (leader) {
            tma.store(2, &store_map, coordinates, buffer, null);
            tma.commitGroup();
            tma.waitGroup(0, true);
        }
        nvptx.barrierSyncAligned(.work_group, null);
        slot = slot.next();
    }
    if (leader) tma.waitGroup(0, false);
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        @export(&checkF8f6f4, .{ .name = configurations.f8f6f4_kernel_name });
        for (0..configurations.selectors.len) |index| {
            @export(&CheckBlockScale(index).kernel, .{
                .name = configurations.blockScaleKernelName(index),
            });
        }
        @export(&tmaCopy, .{ .name = configurations.tma_copy_kernel_name });
    }
}
