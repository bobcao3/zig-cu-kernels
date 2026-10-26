//! `Y = A Bᵀ` over FP8 packs on SM100, and `D = Aᵀ B` over the columnwise MXFP8 packs `[K, M]` and
//! `[K, N]`. A cluster of two CTAs computes 256x256 output tiles with `tcgen05.mma.cta_group::2`
//! M256 N256 K32. One warp loads the stages by TMA, one warp of CTA 0 issues the MMAs, and four
//! warps per CTA write the epilogue of one tile while the MMAs of the next tile run.

const builtin = @import("builtin");
const std = @import("std");
const nvptx = std.nvptx;
const bf16 = @import("bf16.zig");
const cluster = @import("cluster.zig");
const configurations = @import("mm_sm100_configurations.zig");
const fp8 = @import("fp8.zig");
const fp8_pack = @import("fp8_pack.zig");
const tcgen05 = @import("tcgen05.zig");
const tma = @import("tma.zig");

const math = nvptx.math;
const Mbarrier = nvptx.Mbarrier;
const TensorMap = tma.TensorMap;
const Recipe = configurations.Recipe;
const Operation = configurations.Operation;
const Chunk = bf16.Chunk;
const stages = configurations.stages;
const tile_size = configurations.tile;
const half = configurations.half;
const k_block = configurations.k_block;
const rank_block = configurations.rank_block;
const chunk_columns = configurations.chunk_columns;
const swizzle = 128;
const steps = k_block / tcgen05.k_step_bytes;
const columns = 512;
const scale_columns = 4;
const wait_limit_ns = 1_000_000_000;
const producer_warp = configurations.epilogue_warps;
const chunk_bytes = half * chunk_columns * 2;
const warp_chunk_bytes = chunk_bytes / configurations.epilogue_warps;
const atom = configurations.scale_atom_bytes;
const mn_major_step = tcgen05.k_step_bytes * swizzle / 16;

const kernel: std.builtin.CallingConvention = .{
    .nvptx_kernel = .{ .max_work_group_size = .{ .x = configurations.threads } },
};

const Shared = struct {
    var full: [stages]Mbarrier addrspace(.shared) = undefined;
    var empty: [stages]Mbarrier addrspace(.shared) = undefined;
    var accumulator_full: [2]Mbarrier addrspace(.shared) = undefined;
    var accumulator_empty: [2]Mbarrier addrspace(.shared) = undefined;
    var weight_scales_ready: Mbarrier addrspace(.shared) = undefined;
    var tensor_base: u32 addrspace(.shared) = undefined;
    extern var storage: [0]u8 align(16) addrspace(.shared);
};

const WaitOrder = nvptx.mbarrier.WaitOrder;
const cluster_acquire: WaitOrder = .{ .semantics = .acquire, .scope = .cluster };
const cluster_release: nvptx.mbarrier.ArriveOrder = .{ .semantics = .release, .scope = .cluster };

/// Traps when the phase does not complete within one second.
fn waitPhase(barrier: *addrspace(.shared) Mbarrier, phase: u32, comptime order: ?WaitOrder) void {
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitPhase(phase, order, null)) {
        if (nvptx.globaltimer() - start > wait_limit_ns) @trap();
    }
}

const Records = [*]addrspace(.global) u64;
const ProducerPart = enum { empty_wait, issue };
const MmaPart = enum { full_wait, copy, mma, commit, accumulator_wait };
const EpiloguePart = enum { accumulator_wait, drain };
const producer_parts = @typeInfo(ProducerPart).@"enum".field_names.len;
const mma_parts = @typeInfo(MmaPart).@"enum".field_names.len;

/// A probe kernel adds the clocks between two marks of a role to the sum of the part that ends at
/// the second mark, and writes the sums of each CTA into the buffer of the `bias` argument.
fn Clocks(comptime probe: bool, comptime Part: type, comptime offset: u32) type {
    return struct {
        sums: [@typeInfo(Part).@"enum".field_names.len]u64 = @splat(0),
        last: u64,

        const Self = @This();

        inline fn start() Self {
            return .{ .last = if (probe) nvptx.clock64() else 0 };
        }

        inline fn mark(clocks: *Self, comptime part: Part) void {
            if (!probe) return;
            const now = nvptx.clock64();
            clocks.sums[@backingInt(part)] += now - clocks.last;
            clocks.last = now;
        }

        inline fn write(clocks: Self, records: Records) void {
            if (!probe) return;
            const record = records + @workGroupId(0) * configurations.probe_words + offset;
            inline for (clocks.sums, 0..) |sum, index| record[index] = sum;
        }
    };
}

const Tile = struct { m: u32, n: u32 };

/// A cluster computes the tiles `first`, `first + step`, ... below `count`.
const Walk = struct {
    first: u32,
    step: u32,
    count: u32,
    tiles_m: u32,
    tiles_n: u32,
};

/// The rowwise recipe keeps two accumulators of 256 columns. The block-scaled recipes keep one slot
/// of scales in the last 12 columns, so their second accumulator ends below the scales and shares
/// its first columns with the last columns of the first accumulator. The GLU kernels start the
/// second accumulator at column 128. The epilogue drains the chunks with the shared columns first,
/// so that the MMAs of the next tile can start while it drains the other chunks.
fn Accumulator(comptime recipe: Recipe, comptime operation: Operation) type {
    return struct {
        const shares_columns = recipe != .fp8_rowwise;
        const scale_column: u32 = columns - 3 * scale_columns;
        const stride: u32 = if (!shares_columns)
            tile_size
        else if (isGlu(operation))
            half
        else
            scale_column - tile_size;
        const lag: u32 = if (shares_columns) 1 else 2;
        const shared_columns: u32 = if (shares_columns) tile_size - stride else tile_size;
        const release_chunks: u32 = std.math.divCeil(u32, shared_columns, chunk_columns) catch
            unreachable;
        const first_chunk: u32 = if (shares_columns) stride / chunk_columns else 0;
    };
}

/// Walks bands of `band_rows` tile rows, row first within a band, so that resident clusters share
/// the tiles of B in L2.
fn tileOf(index: u32, tiles_m: u32, tiles_n: u32) Tile {
    const band_tiles = configurations.band_rows * tiles_n;
    const band = index / band_tiles;
    const first = band * configurations.band_rows;
    const rows_in_band = @min(configurations.band_rows, tiles_m - first);
    const within = index - band * band_tiles;
    return .{ .m = first + within % rows_in_band, .n = within / rows_in_band };
}

const Layout = struct {
    base: u32,

    fn aTile(layout: Layout, slot: u32) u32 {
        return layout.base + slot * configurations.value_stage_bytes;
    }

    fn bTile(layout: Layout, slot: u32) u32 {
        return layout.aTile(slot) + half * k_block;
    }

    fn aScales(layout: Layout, slot: u32) u32 {
        const values = stages * configurations.value_stage_bytes;
        return layout.base + values + slot * configurations.scale_stage_bytes;
    }

    fn bScales(layout: Layout, slot: u32) u32 {
        return layout.aScales(slot) + atom;
    }

    fn weightScales(layout: Layout, span: u32) u32 {
        return layout.aScales(stages) + span * configurations.weight_span_bytes;
    }

    fn output(layout: Layout, comptime recipe: Recipe) u32 {
        return if (recipe == .fp8_b128) layout.base else layout.aScales(stages);
    }
};

/// A columnwise tile holds 128 K rows of 128 bytes in 1024-byte swizzle atoms of 8 rows. The
/// stride byte offset steps over 8 K rows; the leading byte offset steps between 128-byte chunks
/// along M or N, and a tile of one CTA has one chunk (PTX ISA 9.7.18.3.1 and 9.7.18.3.2).
const mn_major_layout: tcgen05.SharedLayout = .{
    .leading_byte_offset = k_block * swizzle,
    .stride_byte_offset = 8 * swizzle,
    .swizzle = .bytes_128,
};

fn operandDescriptor(comptime operation: Operation, start: u32) u64 {
    return switch (operation) {
        .mm_abt,
        .mm_abt_low_rank,
        .mm_glu_swiglu,
        .mm_glu_geglu_tanh,
        => tcgen05.kMajorDescriptor(start, swizzle),
        .mm_atb => tcgen05.sharedDescriptor(start, mn_major_layout),
    };
}

fn stepOffsets(comptime operation: Operation) [steps]u32 {
    return switch (operation) {
        .mm_abt,
        .mm_abt_low_rank,
        .mm_glu_swiglu,
        .mm_glu_geglu_tanh,
        => tcgen05.kMajorOffsets(steps, half, swizzle),
        .mm_atb => std.simd.iota(u32, steps) * @as(@Vector(steps, u32), @splat(mn_major_step)),
    };
}

/// The scale atoms of a row pack run along K within a block of 128 rows; the atoms of a
/// columnwise pack run along M or N within a block of 128 K rows.
fn scaleAtom(comptime operation: Operation, block: u32, g: u32, blocks: u32) i32 {
    return @intCast(switch (operation) {
        .mm_abt, .mm_abt_low_rank, .mm_glu_swiglu, .mm_glu_geglu_tanh => block * blocks + g,
        .mm_atb => g * blocks + block,
    });
}

fn stageBytes(comptime recipe: Recipe) u32 {
    const scale_bytes: u32 = switch (recipe) {
        .mxfp8 => 3 * atom,
        .fp8_b128 => atom,
        .fp8_rowwise => 0,
    };
    return 2 * (configurations.value_stage_bytes + scale_bytes);
}

const Maps = struct {
    a: *const TensorMap,
    b: *const TensorMap,
    a_scales: *const TensorMap,
    b_scales: *const TensorMap,
    low: *const TensorMap,
    up: *const TensorMap,
};

const Sizes = struct {
    rows: u32,
    width: u32,
    k_blocks: u32,
    scale_blocks: u32,
    b_blocks: u32,
    residual_stride: u32,
    rank_steps: u32,
};

/// The weight of `fp8_b128` has one scale byte per 128x128 tile. Each 32-bit word holds the bytes
/// of four K blocks. Lane `l` of the producer warp holds the words of span `32 * chunk + l` for the
/// two N blocks of the tile.
fn loadWeightWords(
    weight_scales: [*]addrspace(.global) const u32,
    first_block: u32,
    chunk: u32,
    sizes: Sizes,
    lane: u32,
) [2]u32 {
    const span = chunk * 32 + lane;
    var words: [2]u32 = undefined;
    inline for (&words, 0..) |*word, part| {
        const block = first_block + part;
        const in_range = block < sizes.b_blocks and span < sizes.scale_blocks;
        word.* = if (in_range) weight_scales[block * sizes.scale_blocks + span] else 0;
    }
    return words;
}

/// The MMA copies the first 8 rows of 16 bytes to all rows with `repeated_scale_rows`, so 32 copies
/// of a word fill the 128 rows of one N block. The producer warp writes the words of every span of
/// the tile once, before the first stage.
fn writeWeightScales(
    layout: Layout,
    weight_scales: [*]addrspace(.global) const u32,
    first_block: u32,
    sizes: Sizes,
    lane: u32,
) void {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    const chunks = std.math.divCeil(u32, sizes.scale_blocks, 32) catch unreachable;
    for (0..chunks) |chunk_index| {
        const chunk: u32 = @intCast(chunk_index);
        const words = loadWeightWords(weight_scales, first_block, chunk, sizes, lane);
        const spans = @min(32, sizes.scale_blocks - chunk * 32);
        for (0..spans) |source_index| {
            const source: u32 = @intCast(source_index);
            const target: [*]addrspace(.shared) u32 =
                @ptrFromInt(layout.weightScales(chunk * 32 + source));
            inline for (words, 0..) |word, part| {
                target[part * 32 + lane] = nvptx.shflSync(.idx, u32, word, source, clamp, .all);
            }
        }
    }
    nvptx.fenceProxyAsync(.shared);
    nvptx.barWarpSync(.all);
    if (lane == 0) {
        const ready = cluster.leaderAddress(&Shared.weight_scales_ready);
        nvptx.mbarrier.arriveCluster(ready, cluster_release);
    }
}

fn produce(
    comptime recipe: Recipe,
    comptime operation: Operation,
    comptime probe: bool,
    layout: Layout,
    rank: u32,
    walk: Walk,
    maps: Maps,
    weight_scales: [*]addrspace(.global) const u32,
    sizes: Sizes,
    records: Records,
) void {
    var clocks: Clocks(probe, ProducerPart, 0) = .start();
    var slot: u32 = 0;
    var phase: u32 = 0;
    var tile_index = walk.first;
    while (tile_index < walk.count) : (tile_index += walk.step) {
        const tile = tileOf(tile_index, walk.tiles_m, walk.tiles_n);
        produceTile(
            recipe,
            operation,
            layout,
            rank,
            tile,
            maps,
            weight_scales,
            sizes,
            &slot,
            &phase,
            &clocks,
        );
    }
    if (@workItemId(0) % 32 == 0) clocks.write(records);
}

fn produceTile(
    comptime recipe: Recipe,
    comptime operation: Operation,
    layout: Layout,
    rank: u32,
    tile: Tile,
    maps: Maps,
    weight_scales: [*]addrspace(.global) const u32,
    sizes: Sizes,
    slot_pointer: *u32,
    phase_pointer: *u32,
    clocks: anytype,
) void {
    const lane = @workItemId(0) % 32;
    const a_row: i32 = @intCast(tile.m * tile_size + rank * half);
    const glu = comptime isGlu(operation);
    const b_block = if (glu) tile.n else tile.n * 2;
    const b_next = if (glu) tile.n + sizes.b_blocks else b_block + 1;
    const b_row: i32 = if (glu)
        @intCast((if (rank == 0) b_block else b_next) * half)
    else
        @intCast(tile.n * tile_size + rank * half);
    const a_block = tile.m * 2 + rank;
    if (recipe == .fp8_b128) writeWeightScales(layout, weight_scales, b_block, sizes, lane);
    var slot = slot_pointer.*;
    var phase = phase_pointer.*;
    defer {
        slot_pointer.* = slot;
        phase_pointer.* = phase;
    }
    for (0..sizes.k_blocks) |index| {
        const g: u32 = @intCast(index);
        waitPhase(&Shared.empty[slot], phase ^ 1, null);
        clocks.mark(.empty_wait);
        if (nvptx.electSync(.all)) {
            const leader_full = cluster.leaderAddress(&Shared.full[slot]);
            if (rank == 0) _ = Shared.full[slot].arriveExpectTx(stageBytes(recipe));
            const k: i32 = @intCast(g * k_block);
            const a_at: [2]i32 = if (operation == .mm_atb) .{ a_row, k } else .{ k, a_row };
            const b_at: [2]i32 = if (operation == .mm_atb) .{ b_row, k } else .{ k, b_row };
            tma.loadPair(2, layout.aTile(slot), maps.a, a_at, leader_full, null);
            tma.loadPair(2, layout.bTile(slot), maps.b, b_at, leader_full, null);
            const blocks = sizes.scale_blocks;
            const b_blocks = if (operation == .mm_atb) sizes.b_blocks else blocks;
            switch (recipe) {
                .mxfp8 => {
                    const a_atom = scaleAtom(operation, a_block, g, blocks);
                    const b_atom = scaleAtom(operation, b_block, g, b_blocks);
                    const b_target = layout.bScales(slot);
                    const b_map = maps.b_scales;
                    tma.loadPair(
                        2,
                        layout.aScales(slot),
                        maps.a_scales,
                        .{ 0, a_atom },
                        leader_full,
                        null,
                    );
                    tma.loadPair(2, b_target, b_map, .{ 0, b_atom }, leader_full, null);
                    const next = scaleAtom(operation, b_next, g, b_blocks);
                    tma.loadPair(2, b_target + atom, b_map, .{ 0, next }, leader_full, null);
                },
                .fp8_b128 => {
                    const a_atom: i32 = @intCast(a_block * blocks + g / 4);
                    tma.loadPair(
                        2,
                        layout.aScales(slot),
                        maps.a_scales,
                        .{ 0, a_atom },
                        leader_full,
                        null,
                    );
                },
                .fp8_rowwise => {},
            }
        }
        clocks.mark(.issue);
        slot += 1;
        if (slot == stages) {
            slot = 0;
            phase ^= 1;
        }
    }
    if (operation == .mm_abt_low_rank) {
        for (0..rankBlocks(sizes)) |index| {
            waitPhase(&Shared.empty[slot], phase ^ 1, null);
            if (nvptx.electSync(.all)) {
                const leader_full = cluster.leaderAddress(&Shared.full[slot]);
                if (rank == 0) _ = Shared.full[slot].arriveExpectTx(stageBytes(.fp8_rowwise));
                const column: i32 = @intCast(index * rank_block);
                const low_at: [2]i32 = .{ column, a_row };
                const up_at: [2]i32 = .{ column, b_row };
                tma.loadPair(2, layout.aTile(slot), maps.low, low_at, leader_full, null);
                tma.loadPair(2, layout.bTile(slot), maps.up, up_at, leader_full, null);
            }
            slot += 1;
            if (slot == stages) {
                slot = 0;
                phase ^= 1;
            }
        }
    }
}

fn isGlu(operation: Operation) bool {
    return operation == .mm_glu_swiglu or operation == .mm_glu_geglu_tanh;
}

fn rankBlocks(sizes: Sizes) u32 {
    return std.math.divCeil(u32, sizes.rank_steps, steps) catch unreachable;
}

/// `.kind::f16` reads the types of A and B from the bits that `.kind::f8f6f4` uses, and BF16 is
/// type 1 (PTX ISA 9.7.18.4.2, Table 51).
fn bf16Descriptor(comptime shape: tcgen05.Shape) u32 {
    const bf16_types = 1 << 7 | 1 << 10;
    return tcgen05.f8f6f4Descriptor(shape) | bf16_types;
}

/// Adds the BF16 product of `count` K16 steps to the accumulator `d`. A K16 step of BF16 covers
/// the same 32 bytes of a row as a K32 step of FP8.
fn mmaBf16Steps(comptime count: u32, d: u32, a: u64, b: u64, comptime instruction: u32) void {
    const offsets = comptime tcgen05.kMajorOffsets(count, half, swizzle);
    comptime var text: []const u8 = "{\n.reg .b32 a_low, a_high, b_low, b_high, a_step, b_step;\n" ++
        ".reg .b64 a, b;\nmov.b64 {a_low, a_high}, %[a];\nmov.b64 {b_low, b_high}, %[b];\n";
    inline for (offsets) |offset| {
        text = text ++ std.fmt.comptimePrint(
            "add.u32 a_step, a_low, {d};\nmov.b64 a, {{a_step, a_high}};\n" ++
                "add.u32 b_step, b_low, {d};\nmov.b64 b, {{b_step, b_high}};\n" ++
                "tcgen05.mma.cta_group::2.kind::f16 [%[d]], a, b, {d}, 1;\n",
            .{ offset, offset, instruction },
        );
    }
    asm volatile (text ++ "}"
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
        : .{ .memory = true });
}

fn issue(
    comptime recipe: Recipe,
    comptime operation: Operation,
    comptime probe: bool,
    layout: Layout,
    tensor: u32,
    walk: Walk,
    sizes: Sizes,
    records: Records,
) void {
    const accumulator = Accumulator(recipe, operation);
    var clocks: Clocks(probe, MmaPart, producer_parts) = .start();
    if (recipe == .fp8_b128) waitPhase(&Shared.weight_scales_ready, 0, cluster_acquire);
    var slot: u32 = 0;
    var phase: u32 = 0;
    var local: u32 = 0;
    var tile_index = walk.first;
    while (tile_index < walk.count) : ({
        tile_index += walk.step;
        local += 1;
    }) {
        const stage = local % 2;
        if (local >= accumulator.lag) {
            const previous = local - accumulator.lag;
            const barrier = &Shared.accumulator_empty[previous % 2];
            waitPhase(barrier, (previous / 2) % 2, cluster_acquire);
            tcgen05.fence(.after_thread_sync);
        }
        clocks.mark(.accumulator_wait);
        const target = tensor + stage * accumulator.stride;
        issueTile(recipe, operation, layout, tensor, target, sizes, &slot, &phase, &clocks);
        if (nvptx.electSync(.all)) {
            tcgen05.commitMulticast(.two, &Shared.accumulator_full[stage], 0b11);
        }
    }
    if (@workItemId(0) % 32 == 0) clocks.write(records);
}

/// The PTX loop that one thread runs over the K blocks of an mxfp8 tile. It tests the full barrier
/// of the next stage before it issues the copies and MMAs of the current stage, and it branches on
/// that test only at the next loop head, so the thread does not wait for the test result before it
/// issues.
fn mxfp8BlocksText(comptime offsets: []const u32, comptime instructions: []const u32) []const u8 {
    const copy = "tcgen05.cp.cta_group::2.32x128b.warpx4 ";
    const commit = "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster" ++
        ".multicast::cluster.b64 ";
    const mma = "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.block32 [%[d]], a, b, " ++
        "instruction, [%[scale_a]], [%[scale_b]], ";
    var steps_text: []const u8 = "";
    for (offsets, instructions, 0..) |offset, instruction, step| {
        steps_text = steps_text ++ std.fmt.comptimePrint(
            "add.u32 low, a_slot, {d};\nmov.b64 a, {{low, a_high}};\n" ++
                "add.u32 low, b_slot, {d};\nmov.b64 b, {{low, b_high}};\n" ++
                "mov.b32 instruction, {d};\n{s}{s};\n",
            .{ offset, offset, instruction, mma, if (step == 0) "accumulate" else "1" },
        );
    }
    return std.fmt.comptimePrint(
        \\{{
        \\.reg .pred ready, more, wrap, late, accumulate;
        \\.reg .b32 slot, phase, next, next_phase, g, full, low, instruction;
        \\.reg .b32 a_low, a_high, b_low, b_high, s_low, s_high, a_slot, b_slot;
        \\.reg .b64 a, b, s, start, now;
        \\.reg .b16 mask;
        \\mov.b32 slot, %[slot];
        \\mov.b32 phase, %[phase];
        \\mov.b32 g, 0;
        \\mov.b64 {{a_low, a_high}}, %[a];
        \\mov.b64 {{b_low, b_high}}, %[b];
        \\mov.b64 {{s_low, s_high}}, %[s];
        \\mov.b16 mask, 3;
        \\mad.lo.u32 full, slot, 8, %[full];
        \\mbarrier.try_wait.parity.shared.b64 ready, [full], phase;
        \\issue_loop:
        \\@!ready bra wait_start;
        \\issue_ready:
        \\tcgen05.fence::after_thread_sync;
        \\setp.ne.u32 accumulate, g, 0;
        \\add.u32 g, g, 1;
        \\setp.lt.u32 more, g, %[k_blocks];
        \\add.u32 next, slot, 1;
        \\setp.eq.u32 wrap, next, {d};
        \\selp.b32 next, 0, next, wrap;
        \\xor.b32 next_phase, phase, 1;
        \\selp.b32 next_phase, next_phase, phase, wrap;
        \\mad.lo.u32 full, next, 8, %[full];
        \\@more mbarrier.try_wait.parity.shared.b64 ready, [full], next_phase;
        \\mad.lo.u32 low, slot, {d}, s_low;
        \\mov.b64 s, {{low, s_high}};
        \\{s}[%[scale_a]], s;
        \\add.u32 low, low, {d};
        \\mov.b64 s, {{low, s_high}};
        \\{s}[%[scale_b]], s;
        \\add.u32 low, low, {d};
        \\mov.b64 s, {{low, s_high}};
        \\{s}[%[scale_upper]], s;
        \\mad.lo.u32 a_slot, slot, {d}, a_low;
        \\mad.lo.u32 b_slot, slot, {d}, b_low;
        \\{s}mad.lo.u32 low, slot, 8, %[empty];
        \\{s}[low], mask;
        \\mov.b32 slot, next;
        \\mov.b32 phase, next_phase;
        \\@more bra issue_loop;
        \\bra.uni issue_done;
        \\wait_start:
        \\mov.u64 start, %globaltimer;
        \\wait_spin:
        \\mbarrier.try_wait.parity.shared.b64 ready, [full], phase;
        \\@ready bra issue_ready;
        \\mov.u64 now, %globaltimer;
        \\sub.u64 now, now, start;
        \\setp.gt.u64 late, now, {d};
        \\@late trap;
        \\bra.uni wait_spin;
        \\issue_done:
        \\}}
    , .{
        stages,
        configurations.scale_stage_bytes >> 4,
        copy,
        atom >> 4,
        copy,
        atom >> 4,
        copy,
        configurations.value_stage_bytes >> 4,
        configurations.value_stage_bytes >> 4,
        steps_text,
        commit,
        wait_limit_ns,
    });
}

fn issueTile(
    comptime recipe: Recipe,
    comptime operation: Operation,
    layout: Layout,
    tensor: u32,
    target: u32,
    sizes: Sizes,
    slot_pointer: *u32,
    phase_pointer: *u32,
    clocks: anytype,
) void {
    const major: tcgen05.Major = if (operation == .mm_atb) .mn else .k;
    const shape: tcgen05.Shape = .{
        .cta_group = .two,
        .m = tile_size,
        .n = tile_size,
        .a_major = major,
        .b_major = major,
    };
    const offsets = comptime stepOffsets(operation);
    var slot = slot_pointer.*;
    var phase = phase_pointer.*;
    defer {
        slot_pointer.* = slot;
        phase_pointer.* = phase;
    }
    if (recipe == .mxfp8 and major == .k) {
        if (sizes.k_blocks != 0) if (nvptx.electSync(.all)) {
            const instructions = comptime tcgen05.mxf8f6f4StepDescriptors(shape, steps);
            const text = comptime mxfp8BlocksText(&offsets, &instructions);
            const scale_a = tensor + Accumulator(recipe, operation).scale_column;
            asm volatile (text
                :
                : [slot] "r" (slot),
                  [phase] "r" (phase),
                  [k_blocks] "r" (sizes.k_blocks),
                  [full] "r" (&Shared.full[0]),
                  [empty] "r" (&Shared.empty[0]),
                  [a] "l" (operandDescriptor(operation, layout.aTile(0))),
                  [b] "l" (operandDescriptor(operation, layout.bTile(0))),
                  [s] "l" (tcgen05.sharedDescriptor(layout.aScales(0), .scale_atom)),
                  [d] "r" (target),
                  [scale_a] "r" (scale_a),
                  [scale_b] "r" (scale_a + scale_columns),
                  [scale_upper] "r" (scale_a + 2 * scale_columns),
                : .{ .memory = true });
        };
        clocks.mark(.mma);
        const total = slot + sizes.k_blocks;
        slot = total % stages;
        phase ^= (total / stages) % 2;
    } else for (0..sizes.k_blocks) |index| {
        const g: u32 = @intCast(index);
        waitPhase(&Shared.full[slot], phase, null);
        tcgen05.fence(.after_thread_sync);
        clocks.mark(.full_wait);
        const a = operandDescriptor(operation, layout.aTile(slot));
        const b = operandDescriptor(operation, layout.bTile(slot));
        const accumulate = g != 0;
        // Every K block copies its scales into the same columns. This relies on the issue order
        // of the copies and the MMAs of one warp (PTX ISA 9.7.18.6.2, pairs 2 and 7).
        const scale_a = tensor + Accumulator(recipe, operation).scale_column;
        const scale_b = scale_a + scale_columns;
        const a_source = tcgen05.sharedDescriptor(layout.aScales(slot), .scale_atom);
        if (nvptx.electSync(.all)) switch (recipe) {
            .fp8_rowwise => {
                const instructions: [steps]u32 = comptime @splat(tcgen05.f8f6f4Descriptor(shape));
                clocks.mark(.copy);
                tcgen05.mmaSteps(.two, &offsets, &offsets, &instructions, target, a, b, accumulate);
            },
            .mxfp8 => {
                const b_scales = layout.bScales(slot);
                tcgen05.copyWarpx4(.two, scale_a, a_source);
                tcgen05.copyWarpx4(.two, scale_b, tcgen05.sharedDescriptor(b_scales, .scale_atom));
                const upper = tcgen05.sharedDescriptor(b_scales + atom, .scale_atom);
                tcgen05.copyWarpx4(.two, scale_b + scale_columns, upper);
                clocks.mark(.copy);
                const instructions = comptime tcgen05.mxf8f6f4StepDescriptors(shape, steps);
                tcgen05.mmaBlockScaledSteps(
                    .two,
                    &offsets,
                    &offsets,
                    &instructions,
                    target,
                    a,
                    b,
                    scale_a,
                    scale_b,
                    accumulate,
                );
            },
            .fp8_b128 => {
                const b_scales = layout.weightScales(g / 4);
                const repeated = tcgen05.SharedLayout.repeated_scale_rows;
                tcgen05.copyWarpx4(.two, scale_a, a_source);
                tcgen05.copyWarpx4(.two, scale_b, tcgen05.sharedDescriptor(b_scales, repeated));
                const upper = tcgen05.sharedDescriptor(b_scales + 128, repeated);
                tcgen05.copyWarpx4(.two, scale_b + scale_columns, upper);
                clocks.mark(.copy);
                switch (g % 4) {
                    inline 0...3 => |id| {
                        const byte: u2 = @intCast(id);
                        const descriptor = comptime tcgen05.mxf8f6f4Descriptor(shape, byte, byte);
                        const instructions: [steps]u32 = comptime @splat(descriptor);
                        tcgen05.mmaBlockScaledSteps(
                            .two,
                            &offsets,
                            &offsets,
                            &instructions,
                            target,
                            a,
                            b,
                            scale_a,
                            scale_b,
                            accumulate,
                        );
                    },
                    else => unreachable,
                }
            },
        };
        clocks.mark(.mma);
        if (nvptx.electSync(.all)) tcgen05.commitMulticast(.two, &Shared.empty[slot], 0b11);
        clocks.mark(.commit);
        slot += 1;
        if (slot == stages) {
            slot = 0;
            phase ^= 1;
        }
    }
    if (operation == .mm_abt_low_rank) {
        const instruction = comptime bf16Descriptor(shape);
        for (0..rankBlocks(sizes)) |index| {
            const first_step: u32 = @intCast(index * steps);
            waitPhase(&Shared.full[slot], phase, null);
            tcgen05.fence(.after_thread_sync);
            const a = tcgen05.kMajorDescriptor(layout.aTile(slot), swizzle);
            const b = tcgen05.kMajorDescriptor(layout.bTile(slot), swizzle);
            if (nvptx.electSync(.all)) {
                if (sizes.rank_steps - first_step >= steps) {
                    mmaBf16Steps(steps, target, a, b, instruction);
                } else {
                    mmaBf16Steps(steps / 2, target, a, b, instruction);
                }
            }
            if (nvptx.electSync(.all)) tcgen05.commitMulticast(.two, &Shared.empty[slot], 0b11);
            slot += 1;
            if (slot == stages) {
                slot = 0;
                phase ^= 1;
            }
        }
    }
}

const Terms = struct {
    row_scales: [*]addrspace(.global) const f32,
    column_scales: [*]addrspace(.global) const @Vector(4, f32),
    bias: [*]addrspace(.global) const Chunk,
    residual: [*]addrspace(.global) const Chunk,
    has_bias: bool,
    glu: Glu = undefined,
};

/// The GLU kernels write the FP8 codes and E8M0 scales of `h`, and the BF16 `Z` when `z` is not 0.
const Glu = struct {
    codes: [*]addrspace(.global) Chunk,
    scales: [*]addrspace(.global) u16,
    z: [*]addrspace(.global) Chunk,
    hidden: u32,
    gate_first: bool,
    limit: f32,
};

/// Each thread owns one accumulator row. A warp stores its 32 rows of each 64-column chunk through
/// a 128-byte swizzled tile, whose 16-byte unit `u` of row `r` sits at unit `u ^ (r % 8)`.
fn drain(
    comptime recipe: Recipe,
    comptime operation: Operation,
    comptime probe: bool,
    layout: Layout,
    tensor: u32,
    rank: u32,
    walk: Walk,
    y_map: *const TensorMap,
    terms: Terms,
    sizes: Sizes,
    records: Records,
) void {
    var clocks: Clocks(probe, EpiloguePart, producer_parts + mma_parts) = .start();
    var local: u32 = 0;
    var tile_index = walk.first;
    while (tile_index < walk.count) : ({
        tile_index += walk.step;
        local += 1;
    }) {
        const tile = tileOf(tile_index, walk.tiles_m, walk.tiles_n);
        const stage = local % 2;
        waitPhase(&Shared.accumulator_full[stage], (local / 2) % 2, null);
        tcgen05.fence(.after_thread_sync);
        clocks.mark(.accumulator_wait);
        if (comptime isGlu(operation)) {
            drainGlu(operation, layout, tensor, rank, tile, stage, y_map, terms.glu, sizes);
        } else {
            drainTile(recipe, operation, layout, tensor, rank, tile, stage, y_map, terms, sizes);
        }
        clocks.mark(.drain);
    }
    if (@workItemId(0) % 32 == 0) tma.waitGroup(0, true);
    if (@workItemId(0) == 0) clocks.write(records);
}

fn drainTile(
    comptime recipe: Recipe,
    comptime operation: Operation,
    layout: Layout,
    tensor: u32,
    rank: u32,
    tile: Tile,
    stage: u32,
    y_map: *const TensorMap,
    terms: Terms,
    sizes: Sizes,
) void {
    const accumulator = Accumulator(recipe, operation);
    const thread = @workItemId(0);
    const warp = thread / 32;
    const lane = thread % 32;
    const row_base = tile.m * tile_size + rank * half + warp * 32;
    const row = row_base + lane;
    const row_in_range = row < sizes.rows;
    const has_bias = terms.has_bias;
    const has_residual = @intFromPtr(terms.residual) != 0 and row_in_range;
    const row_scale: f32 = if (recipe != .fp8_rowwise)
        1
    else if (row_in_range)
        terms.row_scales[row]
    else
        0;
    const chunks = tile_size / chunk_columns;
    const rotation: u32 = if (stage == 0) accumulator.first_chunk else 0;
    const separate_chunks = recipe == .fp8_b128;
    inline for (0..chunks) |order| {
        const chunk = (@as(u32, order) + rotation) % chunks;
        const chunk_offset = if (separate_chunks) chunk * chunk_bytes else 0;
        const output = layout.output(recipe) + chunk_offset + warp * warp_chunk_bytes;
        const column_base = tile.n * tile_size + chunk * chunk_columns;
        const tensor_column = stage * accumulator.stride + chunk * chunk_columns;
        const lane_address = tcgen05.address(warp * 32, tensor_column);
        const loaded = tcgen05.load(f32, .@"32x32b", chunk_columns, .none, tensor + lane_address);
        tcgen05.waitSyncAligned(.ld);
        if (order + 1 == accumulator.release_chunks) {
            tcgen05.fence(.before_thread_sync);
            nvptx.barWarpSync(.all);
            if (lane == 0) {
                const empty = cluster.leaderAddress(&Shared.accumulator_empty[stage]);
                nvptx.mbarrier.arriveCluster(empty, cluster_release);
            }
        }
        const values: [chunk_columns]f32 = loaded;
        if (!separate_chunks and lane == 0) tma.waitGroup(0, true);
        nvptx.barWarpSync(.all);
        const target = output + lane * 128;
        inline for (0..chunk_columns / 8) |unit_index| {
            const unit: u32 = unit_index;
            const column = column_base + unit * 8;
            var sums: [8]f32 = values[unit * 8 ..][0..8].*;
            if (column < sizes.width) {
                if (recipe == .fp8_rowwise) {
                    const low = terms.column_scales[column / 4];
                    const high = terms.column_scales[column / 4 + 1];
                    inline for (0..4) |i| {
                        sums[i] *= row_scale * low[i];
                        sums[i + 4] *= row_scale * high[i];
                    }
                }
                if (has_residual) {
                    const residual = bf16.unpackChunk(
                        terms.residual[(row * sizes.residual_stride + column) / 8],
                    );
                    inline for (&sums, residual) |*sum, term| sum.* += term;
                }
                if (has_bias) {
                    const bias = bf16.unpackChunk(terms.bias[column / 8]);
                    inline for (&sums, bias) |*sum, term| sum.* += term;
                }
            }
            const unit_address = target + ((unit ^ (lane % 8)) * 16);
            const pointer: *addrspace(.shared) Chunk = @ptrFromInt(unit_address);
            pointer.* = bf16.packChunk(sums);
        }
        nvptx.fenceProxyAsync(.shared);
        nvptx.barWarpSync(.all);
        if (lane == 0 and row_base < sizes.rows and column_base < sizes.width) {
            const coordinates: [2]i32 = .{ @intCast(column_base), @intCast(row_base) };
            tma.store(2, y_map, coordinates, output, null);
            tma.commitGroup();
        }
    }
}

const negative_log2e: f32 = @bitCast(@as(u32, 0xbfb8_aa3b));
const gelu_linear: f32 = @bitCast(@as(u32, 0x4013_5761));
const gelu_cubic: f32 = @bitCast(@as(u32, 0x3dd2_d3e8));

/// The GLU functions repeat the instructions of `glu_quant.zig`, so that the fused epilogue and
/// the composed chain give the same bits.
inline fn swiglu(up: f32, gate: f32, limit: f32) f32 {
    const g = @min(gate, limit);
    const u = @max(@min(up, limit), -limit);
    const exponent = math.mulF32(.nearest_even, .{ .ftz = true }, g, negative_log2e);
    const power = math.ex2ApproxF32(true, exponent);
    const sigmoid = math.rcpApproxF32(true, math.addF32(.nearest_even, .{ .ftz = true }, power, 1));
    const silu = bf16.round(math.mulF32(.nearest_even, .{}, g, sigmoid));
    return bf16.round(math.mulF32(.nearest_even, .{}, u, silu));
}

inline fn gegluTanh(up: f32, gate: f32) f32 {
    const square = math.mulF32(.nearest_even, .{}, gate, gate);
    const inner = math.fmaF32(.nearest_even, .{}, gelu_cubic, square, gelu_linear);
    const power = math.ex2ApproxF32(true, math.mulF32(.nearest_even, .{}, @abs(gate), inner));
    const reciprocal = math.rcpApproxF32(true, math.addF32(.nearest_even, .{}, power, 1));
    const magnitude = math.fmaF32(.nearest_even, .{}, reciprocal, -2, 1);
    const sign: f32 = if (gate < 0) -0.5 else 0.5;
    const one_half = math.fmaF32(.nearest_even, .{}, sign, magnitude, 0.5);
    const gelu = bf16.round(math.mulF32(.nearest_even, .{}, gate, one_half));
    return bf16.round(math.mulF32(.nearest_even, .{}, gelu, up));
}

/// CTA 0 multiplies the `lo` rows of W1 and CTA 1 the matching `hi` rows, so accumulator columns 0
/// to 127 hold `lo` and columns 128 to 255 hold `hi` of the same 128 columns of `h`. Each thread
/// keeps the BF16 sums of the half that shares its columns with the other accumulator, releases
/// those columns, and then combines them with the other half.
fn drainGlu(
    comptime operation: Operation,
    layout: Layout,
    tensor: u32,
    rank: u32,
    tile: Tile,
    stage: u32,
    z_map: *const TensorMap,
    glu: Glu,
    sizes: Sizes,
) void {
    const accumulator = Accumulator(.mxfp8, operation);
    const warp = @workItemId(0) / 32;
    const lane = @workItemId(0) % 32;
    const row = tile.m * tile_size + rank * half + warp * 32 + lane;
    const chunks = tile_size / chunk_columns;
    const rotation: u32 = if (stage == 0) accumulator.first_chunk else 0;
    const kept_is_lo = stage != 0;
    var kept: [2][chunk_columns / 2]u32 = undefined;
    inline for (0..chunks) |order| {
        const chunk = (@as(u32, order) + rotation) % chunks;
        const tensor_column = stage * accumulator.stride + chunk * chunk_columns;
        const lane_address = tcgen05.address(warp * 32, tensor_column);
        const loaded = tcgen05.load(f32, .@"32x32b", chunk_columns, .none, tensor + lane_address);
        tcgen05.waitSyncAligned(.ld);
        if (order + 1 == accumulator.release_chunks) {
            tcgen05.fence(.before_thread_sync);
            nvptx.barWarpSync(.all);
            if (lane == 0) {
                const empty = cluster.leaderAddress(&Shared.accumulator_empty[stage]);
                nvptx.mbarrier.arriveCluster(empty, cluster_release);
            }
        }
        const values: [chunk_columns]f32 = loaded;
        var words: [chunk_columns / 2]u32 = undefined;
        inline for (&words, 0..) |*word, i| word.* = bf16.pack(values[2 * i], values[2 * i + 1]);
        if (@intFromPtr(glu.z) != 0) {
            storeZ(layout, z_map, words, tile, chunk, row - lane, glu, sizes);
        }
        if (order < 2) {
            kept[order] = words;
        } else {
            const column0 = tile.n * half + @as(u32, order - 2) * chunk_columns;
            const lo = if (kept_is_lo) kept[order - 2] else words;
            const hi = if (kept_is_lo) words else kept[order - 2];
            storeGlu(operation, glu, sizes, row, column0, lo, hi);
        }
    }
}

/// Writes the BF16 sums of a chunk through the swizzled tile of the store epilogue into the map of
/// `Z` viewed as `[rows, 2, hidden]`, which clips the columns of each half at `hidden`.
fn storeZ(
    layout: Layout,
    z_map: *const TensorMap,
    words: [chunk_columns / 2]u32,
    tile: Tile,
    chunk: u32,
    row_base: u32,
    glu: Glu,
    sizes: Sizes,
) void {
    const warp = @workItemId(0) / 32;
    const lane = @workItemId(0) % 32;
    const output = layout.output(.mxfp8) + warp * warp_chunk_bytes;
    if (lane == 0) tma.waitGroup(0, true);
    nvptx.barWarpSync(.all);
    inline for (0..chunk_columns / 8) |unit| {
        const address = output + lane * 128 + (unit ^ (lane % 8)) * 16;
        const pointer: *addrspace(.shared) Chunk = @ptrFromInt(address);
        pointer.* = words[unit * 4 ..][0..4].*;
    }
    nvptx.fenceProxyAsync(.shared);
    nvptx.barWarpSync(.all);
    const column = tile.n * half + chunk % 2 * chunk_columns;
    if (lane == 0 and row_base < sizes.rows and column < glu.hidden) {
        const coordinates: [3]i32 = .{ @intCast(column), @intCast(chunk / 2), @intCast(row_base) };
        tma.store(3, z_map, coordinates, output, null);
        tma.commitGroup();
    }
}

/// Writes `h` for the 64 columns from `column0` of one row: two blocks of 32 codes with one scale
/// byte each.
fn storeGlu(
    comptime operation: Operation,
    glu: Glu,
    sizes: Sizes,
    row: u32,
    column0: u32,
    lo: [chunk_columns / 2]u32,
    hi: [chunk_columns / 2]u32,
) void {
    @setEvalBranchQuota(20_000);
    const row_in_range = row < sizes.rows;
    const width = sizes.b_blocks * half;
    var scales: [2]u8 = undefined;
    inline for (0..2) |block| {
        var h: [32]f32 = undefined;
        inline for (0..16) |pair| {
            const lo_pair = bf16.unpack(lo[block * 16 + pair]);
            const hi_pair = bf16.unpack(hi[block * 16 + pair]);
            inline for (0..2) |e| {
                const up = if (glu.gate_first) hi_pair[e] else lo_pair[e];
                const gate = if (glu.gate_first) lo_pair[e] else hi_pair[e];
                const product = switch (operation) {
                    .mm_glu_swiglu => swiglu(up, gate, glu.limit),
                    .mm_glu_geglu_tanh => gegluTanh(up, gate),
                    else => comptime unreachable,
                };
                const column = column0 + block * 32 + pair * 2 + e;
                h[pair * 2 + e] = if (row_in_range and column < glu.hidden) product else 0;
            }
        }
        const scaling: fp8_pack.Scaling(.e8m0) = .init(fp8.amaxOf(h));
        scales[block] = scaling.scale;
        if (row_in_range) {
            inline for (0..2) |unit| {
                var codes: Chunk = undefined;
                inline for (0..4) |word| {
                    const q = h[unit * 16 + word * 4 ..][0..4].*;
                    codes[word] = @backingInt(nvptx.E4m3x4.fromF32(
                        math.mulF32(.nearest_even, .{}, q[0], scaling.factor),
                        math.mulF32(.nearest_even, .{}, q[1], scaling.factor),
                        math.mulF32(.nearest_even, .{}, q[2], scaling.factor),
                        math.mulF32(.nearest_even, .{}, q[3], scaling.factor),
                    ));
                }
                const column = column0 + block * 32 + unit * 16;
                glu.codes[(@as(u64, row) * width + column) / 16] = codes;
            }
        }
    }
    if (row < std.mem.alignForward(u32, sizes.rows, half)) {
        const scale_atom = row / half * sizes.b_blocks + column0 / half;
        const byte = scale_atom * atom + row % 32 * 16 + row % half / 32 * 4 + column0 / 32 % 4;
        glu.scales[byte / 2] = @as(u16, scales[1]) << 8 | scales[0];
    }
}

fn multiply(
    comptime recipe: Recipe,
    comptime operation: Operation,
    comptime probe: bool,
    maps: Maps,
    y_map: *const TensorMap,
    weight_scales: [*]addrspace(.global) const u32,
    terms: Terms,
    sizes: Sizes,
    tiles: [2]u32,
    records: Records,
) void {
    const thread = @workItemId(0);
    const warp = thread / 32;
    const rank = cluster.rank();
    const walk: Walk = .{
        .first = cluster.clusterid(0),
        .step = cluster.nclusterid(0),
        .count = tiles[0] * tiles[1],
        .tiles_m = tiles[0],
        .tiles_n = tiles[1],
    };
    const storage = tma.sharedAddress(&Shared.storage);
    const layout: Layout = .{
        .base = std.mem.alignForward(u32, storage, configurations.tile_alignment),
    };
    if (thread == 0) {
        for (&Shared.full) |*barrier| barrier.init(1);
        for (&Shared.empty) |*barrier| barrier.init(1);
        for (&Shared.accumulator_full) |*barrier| barrier.init(1);
        for (&Shared.accumulator_empty) |*barrier| barrier.init(2 * configurations.epilogue_warps);
        Shared.weight_scales_ready.init(2);
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
    const start_clock = if (probe) nvptx.clock64() else 0;
    const start_time = if (probe) nvptx.globaltimer() else 0;
    if (warp < producer_warp) {
        drain(recipe, operation, probe, layout, tensor, rank, walk, y_map, terms, sizes, records);
    } else if (warp == producer_warp) {
        produce(recipe, operation, probe, layout, rank, walk, maps, weight_scales, sizes, records);
    } else if (rank == 0) {
        issue(recipe, operation, probe, layout, tensor, walk, sizes, records);
    }
    if (probe and thread == 0) {
        const record = records + @workGroupId(0) * configurations.probe_words;
        record[configurations.probe_words - 2] = nvptx.clock64() - start_clock;
        record[configurations.probe_words - 1] = nvptx.globaltimer() - start_time;
    }
    tcgen05.fence(.before_thread_sync);
    cluster.sync();
    if (warp == 0) {
        tcgen05.fence(.after_thread_sync);
        tcgen05.deallocate(.two, tensor, columns);
    }
}

fn Entry(comptime recipe: Recipe, comptime operation: Operation, comptime probe: bool) type {
    return struct {
        fn run(
            a_map: TensorMap,
            b_map: TensorMap,
            a_scale_map: TensorMap,
            b_scale_map: TensorMap,
            y_map: TensorMap,
            weight_scales: [*]addrspace(.global) const u32,
            row_scales: [*]addrspace(.global) const f32,
            column_scales: [*]addrspace(.global) const @Vector(4, f32),
            bias: [*]addrspace(.global) const Chunk,
            residual: [*]addrspace(.global) const Chunk,
            rows: u32,
            width: u32,
            k_blocks: u32,
            tiles_m: u32,
            tiles_n: u32,
            scale_blocks: u32,
            b_blocks: u32,
            residual_stride: u32,
        ) callconv(kernel) void {
            const maps: Maps = .{
                .a = &a_map,
                .b = &b_map,
                .a_scales = &a_scale_map,
                .b_scales = &b_scale_map,
                .low = &a_map,
                .up = &b_map,
            };
            const terms: Terms = .{
                .row_scales = row_scales,
                .column_scales = column_scales,
                .bias = bias,
                .residual = residual,
                .has_bias = !probe and @intFromPtr(bias) != 0,
            };
            const sizes: Sizes = .{
                .rows = rows,
                .width = width,
                .k_blocks = k_blocks,
                .scale_blocks = scale_blocks,
                .b_blocks = b_blocks,
                .residual_stride = residual_stride,
                .rank_steps = 0,
            };
            const records: Records = @ptrCast(@constCast(bias));
            const tiles: [2]u32 = .{ tiles_m, tiles_n };
            multiply(
                recipe,
                operation,
                probe,
                maps,
                &y_map,
                weight_scales,
                terms,
                sizes,
                tiles,
                records,
            );
        }
    };
}

/// The `mxfp8` kernel adds `low · upᵀ` with `rank_steps` BF16 MMAs of depth 16, which the MMA warp
/// issues into the same accumulator after the K blocks of each tile.
const LowRankEntry = struct {
    fn run(
        a_map: TensorMap,
        b_map: TensorMap,
        a_scale_map: TensorMap,
        b_scale_map: TensorMap,
        y_map: TensorMap,
        low_map: TensorMap,
        up_map: TensorMap,
        bias: [*]addrspace(.global) const Chunk,
        residual: [*]addrspace(.global) const Chunk,
        rows: u32,
        width: u32,
        k_blocks: u32,
        tiles_m: u32,
        tiles_n: u32,
        scale_blocks: u32,
        rank_steps: u32,
    ) callconv(kernel) void {
        const maps: Maps = .{
            .a = &a_map,
            .b = &b_map,
            .a_scales = &a_scale_map,
            .b_scales = &b_scale_map,
            .low = &low_map,
            .up = &up_map,
        };
        const terms: Terms = .{
            .row_scales = undefined,
            .column_scales = undefined,
            .bias = bias,
            .residual = residual,
            .has_bias = @intFromPtr(bias) != 0,
        };
        const sizes: Sizes = .{
            .rows = rows,
            .width = width,
            .k_blocks = k_blocks,
            .scale_blocks = scale_blocks,
            .b_blocks = 0,
            .residual_stride = 0,
            .rank_steps = rank_steps,
        };
        const tiles: [2]u32 = .{ tiles_m, tiles_n };
        multiply(
            .mxfp8,
            .mm_abt_low_rank,
            false,
            maps,
            &y_map,
            undefined,
            terms,
            sizes,
            tiles,
            @ptrCast(@constCast(bias)),
        );
    }
};

/// The `mxfp8` kernels of `h = act(gate) · up` over `w1 = [lo | hi]` with `b_blocks` blocks of 128
/// rows in each half.
fn GluEntry(comptime operation: Operation) type {
    return struct {
        fn run(
            a_map: TensorMap,
            b_map: TensorMap,
            a_scale_map: TensorMap,
            b_scale_map: TensorMap,
            z_map: TensorMap,
            codes: [*]addrspace(.global) Chunk,
            h_scales: [*]addrspace(.global) u16,
            z: [*]addrspace(.global) Chunk,
            rows: u32,
            hidden: u32,
            k_blocks: u32,
            tiles_m: u32,
            tiles_n: u32,
            b_blocks: u32,
            gate_first: u32,
            limit: f32,
        ) callconv(kernel) void {
            const maps: Maps = .{
                .a = &a_map,
                .b = &b_map,
                .a_scales = &a_scale_map,
                .b_scales = &b_scale_map,
                .low = &a_map,
                .up = &b_map,
            };
            const terms: Terms = .{
                .row_scales = undefined,
                .column_scales = undefined,
                .bias = undefined,
                .residual = undefined,
                .has_bias = false,
                .glu = .{
                    .codes = codes,
                    .scales = h_scales,
                    .z = z,
                    .hidden = hidden,
                    .gate_first = gate_first != 0,
                    .limit = limit,
                },
            };
            const sizes: Sizes = .{
                .rows = rows,
                .width = hidden,
                .k_blocks = k_blocks,
                .scale_blocks = k_blocks,
                .b_blocks = b_blocks,
                .residual_stride = 0,
                .rank_steps = 0,
            };
            const tiles: [2]u32 = .{ tiles_m, tiles_n };
            multiply(
                .mxfp8,
                operation,
                false,
                maps,
                &z_map,
                undefined,
                terms,
                sizes,
                tiles,
                undefined,
            );
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (configurations.glu_operations) |operation| {
            const name = configurations.name(.mxfp8, operation, false);
            @export(&GluEntry(operation).run, .{ .name = name });
        }
        for ([_]Recipe{ .mxfp8, .fp8_b128, .fp8_rowwise }) |recipe| {
            const name = configurations.name(recipe, .mm_abt, false);
            @export(&Entry(recipe, .mm_abt, false).run, .{ .name = name });
        }
        for (configurations.probe_recipes) |recipe| {
            const name = configurations.name(recipe, .mm_abt, true);
            @export(&Entry(recipe, .mm_abt, true).run, .{ .name = name });
        }
        const atb_name = configurations.name(.mxfp8, .mm_atb, false);
        @export(&Entry(.mxfp8, .mm_atb, false).run, .{ .name = atb_name });
        const low_rank_name = configurations.name(.mxfp8, .mm_abt_low_rank, false);
        @export(&LowRankEntry.run, .{ .name = low_rank_name });
    }
}
