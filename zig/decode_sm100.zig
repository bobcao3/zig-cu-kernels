//! Decode `Y = A Bᵀ` for at most 128 rows of `A` on SM100. The kernel computes `Yᵀ = B Aᵀ` with
//! `tcgen05.mma.cta_group::1`: 128 rows of the weight `B` form the MMA's M operand and `T` token
//! rows of `A` form its N operand. A unit is one 128-row weight tile and one 128-deep K block, and
//! the units are numbered with the K block fastest. CTA `c` of `C` takes the units from
//! `c * U / C` to `(c + 1) * U / C`. A CTA that holds every K block of a tile writes BF16 output;
//! for a tile that it shares with other CTAs it writes its FP32 sum to a workspace slot.

const builtin = @import("builtin");
const std = @import("std");
const nvptx = std.nvptx;
const bf16 = @import("bf16.zig");
const configurations = @import("decode_sm100_configurations.zig");
const tcgen05 = @import("tcgen05.zig");
const tma = @import("tma.zig");

const Mbarrier = nvptx.Mbarrier;
const TensorMap = tma.TensorMap;
const Recipe = configurations.Recipe;
const Chunk = bf16.Chunk;
const k_block = configurations.k_block;
const tile_rows = configurations.tile_rows;
const max_stages = configurations.max_stages;
const swizzle = 128;
const steps = k_block / tcgen05.k_step_bytes;
const columns = 512;
const scale_column_base = 256;
const scale_columns = 4;
const wait_limit_ns = 1_000_000_000;
const producer_warp = configurations.epilogue_warps;
const atom = configurations.scale_atom_bytes;
const load_columns = 16;

const kernel: std.builtin.CallingConvention = .{
    .nvptx_kernel = .{ .max_work_group_size = .{ .x = configurations.threads } },
};

const Shared = struct {
    var full: [max_stages]Mbarrier addrspace(.shared) = undefined;
    var empty: [max_stages]Mbarrier addrspace(.shared) = undefined;
    var accumulator_full: [2]Mbarrier addrspace(.shared) = undefined;
    var accumulator_empty: [2]Mbarrier addrspace(.shared) = undefined;
    var tensor_base: u32 addrspace(.shared) = undefined;
    extern var storage: [0]u8 align(16) addrspace(.shared);
};

/// Traps when the phase does not complete within one second.
fn waitPhase(barrier: *addrspace(.shared) Mbarrier, phase: u32) void {
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitPhase(phase, null, null)) {
        if (nvptx.globaltimer() - start > wait_limit_ns) @trap();
    }
}

/// The K blocks `first` to `last` of the weight tile `tile`, which form one accumulator.
const Segment = struct {
    tile: u32,
    first: u32,
    last: u32,
    slot: u32,
};

const Range = struct {
    begin: u32,
    end: u32,
    k_blocks: u32,

    fn of(k_blocks: u32, tiles: u32) Range {
        const units: u64 = @as(u64, tiles) * k_blocks;
        const cta: u64 = @workGroupId(0);
        const ctas: u64 = nvptx.nctaid(0);
        return .{
            .begin = @intCast(cta * units / ctas),
            .end = @intCast((cta + 1) * units / ctas),
            .k_blocks = k_blocks,
        };
    }

    fn firstTile(range: Range) u32 {
        return range.begin / range.k_blocks;
    }

    fn endTile(range: Range) u32 {
        if (range.end <= range.begin) return range.firstTile();
        return (range.end - 1) / range.k_blocks + 1;
    }

    fn segment(range: Range, tile: u32) Segment {
        const start = tile * range.k_blocks;
        return .{
            .tile = tile,
            .first = @max(range.begin, start) - start,
            .last = @min(range.end, start + range.k_blocks) - start,
            .slot = if (tile == range.firstTile()) 0 else 1,
        };
    }
};

fn Layout(comptime tokens: u32) type {
    return struct {
        base: u32,

        const stages = configurations.stages(tokens);
        const value_bytes = configurations.valueStageBytes(tokens);

        fn weight(layout: @This(), slot: u32) u32 {
            return layout.base + slot * value_bytes;
        }

        fn activation(layout: @This(), slot: u32) u32 {
            return layout.weight(slot) + tile_rows * k_block;
        }

        fn weightScales(layout: @This(), slot: u32) u32 {
            return layout.base + stages * value_bytes + slot * configurations.scale_slot_bytes;
        }

        fn activationScales(layout: @This(), slot: u32) u32 {
            return layout.weightScales(slot) + atom;
        }
    };
}

fn stageBytes(comptime recipe: Recipe, comptime tokens: u32) u32 {
    const scale_bytes: u32 = switch (recipe) {
        .mxfp8 => 2 * atom,
        .fp8_b128 => atom,
        .fp8_rowwise => 0,
    };
    return configurations.valueStageBytes(tokens) + scale_bytes;
}

const Maps = struct {
    weight: *const TensorMap,
    activation: *const TensorMap,
    weight_scales: *const TensorMap,
    activation_scales: *const TensorMap,
};

const Sizes = struct {
    m: u32,
    n: u32,
    k_blocks: u32,
    tiles: u32,
    scale_blocks: u32,
    y_stride: u32,
    residual_stride: u32,
};

const Advance = struct {
    slot: u32 = 0,
    phase: u32 = 0,

    fn next(position: *Advance, comptime stages: u32) void {
        position.slot += 1;
        if (position.slot == stages) {
            position.slot = 0;
            position.phase ^= 1;
        }
    }
};

/// The `fp8_b128` weight has one scale byte per unit, four K blocks to a 32-bit word. The warp
/// writes 32 copies of the word, which a `tcgen05.cp` with `repeated_scale_rows` spreads over the
/// 128 lanes.
fn writeWeightWord(target: u32, weight_words: [*]addrspace(.global) const u32, index: u32) void {
    const lane = @workItemId(0) % 32;
    const word = weight_words[index];
    const pointer: [*]addrspace(.shared) u32 = @ptrFromInt(target);
    pointer[lane] = word;
    nvptx.fenceProxyAsync(.shared);
    nvptx.barWarpSync(.all);
}

fn produce(
    comptime recipe: Recipe,
    comptime tokens: u32,
    layout: Layout(tokens),
    range: Range,
    maps: Maps,
    weight_words: [*]addrspace(.global) const u32,
    sizes: Sizes,
) void {
    const stages = Layout(tokens).stages;
    const weight_policy = tma.createPolicy(.evict_first);
    const activation_policy = tma.createPolicy(.evict_last);
    var position: Advance = .{};
    var unit = range.begin;
    while (unit < range.end) : (unit += 1) {
        const tile = unit / sizes.k_blocks;
        const g = unit % sizes.k_blocks;
        const slot = position.slot;
        waitPhase(&Shared.empty[slot], position.phase ^ 1);
        if (recipe == .fp8_b128) {
            const index = tile * sizes.scale_blocks + g / 4;
            writeWeightWord(layout.weightScales(slot), weight_words, index);
        }
        if (nvptx.electSync(.all)) {
            const full = &Shared.full[slot];
            _ = full.arriveExpectTx(stageBytes(recipe, tokens));
            const k: i32 = @intCast(g * k_block);
            const row: i32 = @intCast(tile * tile_rows);
            tma.load(2, layout.weight(slot), maps.weight, .{ k, row }, full, weight_policy);
            tma.load(2, layout.activation(slot), maps.activation, .{ k, 0 }, full, activation_policy);
            const scales = layout.activationScales(slot);
            switch (recipe) {
                .mxfp8 => {
                    const weight_atom: i32 = @intCast(tile * sizes.scale_blocks + g);
                    const target = layout.weightScales(slot);
                    tma.load(2, target, maps.weight_scales, .{ 0, weight_atom }, full, weight_policy);
                    const coordinates: [2]i32 = .{ 0, @intCast(g) };
                    tma.load(2, scales, maps.activation_scales, coordinates, full, activation_policy);
                },
                .fp8_b128 => {
                    const coordinates: [2]i32 = .{ 0, @intCast(g / 4) };
                    tma.load(2, scales, maps.activation_scales, coordinates, full, activation_policy);
                },
                .fp8_rowwise => {},
            }
        }
        position.next(stages);
    }
}

fn issue(comptime recipe: Recipe, comptime tokens: u32, layout: Layout(tokens), tensor: u32, range: Range) void {
    const stages = Layout(tokens).stages;
    const shape: tcgen05.Shape = .{ .cta_group = .one, .m = tile_rows, .n = tokens };
    const offsets = comptime tcgen05.kMajorOffsets(steps, tile_rows, swizzle);
    var position: Advance = .{};
    var local: u32 = 0;
    var tile = range.firstTile();
    while (tile < range.endTile()) : ({
        tile += 1;
        local += 1;
    }) {
        const segment = range.segment(tile);
        const buffer = local % 2;
        if (local >= 2) {
            waitPhase(&Shared.accumulator_empty[buffer], ((local - 2) / 2) % 2);
            tcgen05.fence(.after_thread_sync);
        }
        const target = tensor + buffer * tokens;
        var g = segment.first;
        while (g < segment.last) : (g += 1) {
            const slot = position.slot;
            waitPhase(&Shared.full[slot], position.phase);
            tcgen05.fence(.after_thread_sync);
            const a = tcgen05.kMajorDescriptor(layout.weight(slot), swizzle);
            const b = tcgen05.kMajorDescriptor(layout.activation(slot), swizzle);
            const accumulate = g != segment.first;
            const scale_a = tensor + scale_column_base + slot * 2 * scale_columns;
            const scale_b = scale_a + scale_columns;
            const b_source = tcgen05.sharedDescriptor(layout.activationScales(slot), .scale_atom);
            if (nvptx.electSync(.all)) switch (recipe) {
                .fp8_rowwise => {
                    const instructions: [steps]u32 = comptime @splat(tcgen05.f8f6f4Descriptor(shape));
                    tcgen05.mmaSteps(.one, &offsets, &offsets, &instructions, target, a, b, accumulate);
                },
                .mxfp8 => {
                    const a_source = tcgen05.sharedDescriptor(layout.weightScales(slot), .scale_atom);
                    tcgen05.copyWarpx4(.one, scale_a, a_source);
                    tcgen05.copyWarpx4(.one, scale_b, b_source);
                    const instructions = comptime tcgen05.mxf8f6f4StepDescriptors(shape, steps);
                    tcgen05.mmaBlockScaledSteps(
                        .one,
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
                    const repeated = tcgen05.SharedLayout.repeated_scale_rows;
                    const a_source = tcgen05.sharedDescriptor(layout.weightScales(slot), repeated);
                    tcgen05.copyWarpx4(.one, scale_a, a_source);
                    tcgen05.copyWarpx4(.one, scale_b, b_source);
                    switch (g % 4) {
                        inline 0...3 => |id| {
                            const byte: u2 = @intCast(id);
                            const descriptor = comptime tcgen05.mxf8f6f4Descriptor(shape, byte, byte);
                            const instructions: [steps]u32 = comptime @splat(descriptor);
                            tcgen05.mmaBlockScaledSteps(
                                .one,
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
            if (nvptx.electSync(.all)) tcgen05.commit(.one, &Shared.empty[slot]);
            position.next(stages);
        }
        if (nvptx.electSync(.all)) tcgen05.commit(.one, &Shared.accumulator_full[buffer]);
    }
}

const Terms = struct {
    y: [*]addrspace(.global) u16,
    workspace: [*]addrspace(.global) f32,
    row_scales: [*]addrspace(.global) const f32,
    column_scales: [*]addrspace(.global) const f32,
    bias: [*]addrspace(.global) const u16,
    residual: [*]addrspace(.global) const u16,
};

/// Thread `l` of epilogue warp `w` owns the accumulator lane `32 w + l`, which holds output
/// column `128 tile + 32 w + l` for every token.
fn drain(
    comptime recipe: Recipe,
    comptime tokens: u32,
    tensor: u32,
    range: Range,
    terms: Terms,
    sizes: Sizes,
) void {
    const thread = @workItemId(0);
    const warp = thread / 32;
    const lane = thread % 32;
    var local: u32 = 0;
    var tile = range.firstTile();
    while (tile < range.endTile()) : ({
        tile += 1;
        local += 1;
    }) {
        const segment = range.segment(tile);
        const buffer = local % 2;
        waitPhase(&Shared.accumulator_full[buffer], (local / 2) % 2);
        tcgen05.fence(.after_thread_sync);
        const whole = segment.first == 0 and segment.last == sizes.k_blocks;
        const lane_row = warp * 32 + lane;
        const column = tile * tile_rows + lane_row;
        const column_in_range = column < sizes.n;
        const rowwise = recipe == .fp8_rowwise and column_in_range;
        const column_scale: f32 = if (rowwise) terms.column_scales[column] else 1;
        const has_bias = @intFromPtr(terms.bias) != 0;
        const bias: f32 = if (has_bias and column_in_range) bf16.toF32(terms.bias[column]) else 0;
        const partial_base = ((2 * @workGroupId(0) + segment.slot) * tokens) * tile_rows + lane_row;
        inline for (0..tokens / load_columns) |chunk_index| {
            const chunk: u32 = chunk_index;
            const tensor_column = buffer * tokens + chunk * load_columns;
            const address = tensor + tcgen05.address(warp * 32, tensor_column);
            const loaded = tcgen05.load(f32, .@"32x32b", load_columns, .none, address);
            tcgen05.waitSyncAligned(.ld);
            if (chunk + 1 == tokens / load_columns) {
                tcgen05.fence(.before_thread_sync);
                nvptx.barWarpSync(.all);
                if (lane == 0) _ = Shared.accumulator_empty[buffer].arrive();
            }
            const values: [load_columns]f32 = loaded;
            inline for (values, 0..) |value, index| {
                const token: u32 = chunk * load_columns + @as(u32, index);
                const stored = token < sizes.m;
                if (stored and !whole) {
                    terms.workspace[partial_base + token * tile_rows] = value;
                } else if (stored and column_in_range) {
                    var sum = value;
                    if (recipe == .fp8_rowwise) sum = sum * terms.row_scales[token] * column_scale;
                    if (@intFromPtr(terms.residual) != 0) {
                        sum += bf16.toF32(terms.residual[token * sizes.residual_stride + column]);
                    }
                    if (has_bias) sum += bias;
                    terms.y[token * sizes.y_stride + column] = bf16.fromF32(sum);
                }
            }
        }
    }
}

fn mmAbt(
    comptime recipe: Recipe,
    comptime tokens: u32,
    maps: Maps,
    weight_words: [*]addrspace(.global) const u32,
    terms: Terms,
    sizes: Sizes,
) void {
    const thread = @workItemId(0);
    const warp = thread / 32;
    const range: Range = .of(sizes.k_blocks, sizes.tiles);
    const storage = tma.sharedAddress(&Shared.storage);
    const layout: Layout(tokens) = .{
        .base = std.mem.alignForward(u32, storage, configurations.tile_alignment),
    };
    if (thread == 0) {
        for (&Shared.full) |*barrier| barrier.init(1);
        for (&Shared.empty) |*barrier| barrier.init(1);
        for (&Shared.accumulator_full) |*barrier| barrier.init(1);
        for (&Shared.accumulator_empty) |*barrier| barrier.init(configurations.epilogue_warps);
        nvptx.fenceMbarrierInit();
    }
    if (warp == producer_warp and nvptx.electSync(.all)) {
        tma.prefetch(maps.weight);
        tma.prefetch(maps.activation);
    }
    if (warp == 0) {
        tcgen05.allocate(.one, &Shared.tensor_base, columns);
        tcgen05.relinquishAllocPermit(.one);
    }
    tcgen05.fence(.before_thread_sync);
    nvptx.barrierSyncAligned(.work_group, null);
    tcgen05.fence(.after_thread_sync);
    nvptx.griddepcontrol(.wait);
    nvptx.griddepcontrol(.launch_dependents);
    const tensor = Shared.tensor_base;
    if (warp < producer_warp) {
        drain(recipe, tokens, tensor, range, terms, sizes);
    } else if (warp == producer_warp) {
        produce(recipe, tokens, layout, range, maps, weight_words, sizes);
    } else {
        issue(recipe, tokens, layout, tensor, range);
    }
    tcgen05.fence(.before_thread_sync);
    nvptx.barrierSyncAligned(.work_group, null);
    if (warp == 0) {
        tcgen05.fence(.after_thread_sync);
        tcgen05.deallocate(.one, tensor, columns);
    }
}

fn Entry(comptime recipe: Recipe, comptime tokens: u32) type {
    return struct {
        fn run(
            weight_map: TensorMap,
            activation_map: TensorMap,
            weight_scale_map: TensorMap,
            activation_scale_map: TensorMap,
            weight_words: [*]addrspace(.global) const u32,
            row_scales: [*]addrspace(.global) const f32,
            column_scales: [*]addrspace(.global) const f32,
            bias: [*]addrspace(.global) const u16,
            residual: [*]addrspace(.global) const u16,
            y: [*]addrspace(.global) u16,
            workspace: [*]addrspace(.global) f32,
            m: u32,
            n: u32,
            k_blocks: u32,
            tiles: u32,
            scale_blocks: u32,
            y_stride: u32,
            residual_stride: u32,
        ) callconv(kernel) void {
            const maps: Maps = .{
                .weight = &weight_map,
                .activation = &activation_map,
                .weight_scales = &weight_scale_map,
                .activation_scales = &activation_scale_map,
            };
            const terms: Terms = .{
                .y = y,
                .workspace = workspace,
                .row_scales = row_scales,
                .column_scales = column_scales,
                .bias = bias,
                .residual = residual,
            };
            const sizes: Sizes = .{
                .m = m,
                .n = n,
                .k_blocks = k_blocks,
                .tiles = tiles,
                .scale_blocks = scale_blocks,
                .y_stride = y_stride,
                .residual_stride = residual_stride,
            };
            mmAbt(recipe, tokens, maps, weight_words, terms, sizes);
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (configurations.recipes) |recipe| {
            for (configurations.token_tiles) |tokens| {
                @export(&Entry(recipe, tokens).run, .{ .name = configurations.name(recipe, tokens) });
            }
        }
    }
}
