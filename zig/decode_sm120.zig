//! The product `A Bᵀ` of two FP8 packs on compute capability 12.x for at most 128 rows of `A`.
//! The kernel computes `Yᵀ = B Aᵀ`: 128 rows of the weight `B` form the A operand of
//! `mma.sync.m16n8k32` and a tile of token rows of `A` forms its B operand. A unit is one 128-deep
//! K block of one tile of 128 weight rows. CTA `c` of `C` takes the units
//! `[c U / C, (c + 1) U / C)` with K fastest. A CTA that holds every unit of a tile stores the
//! BF16 result of the tile, and a CTA that holds part of a tile stores FP32 sums in its slot of the
//! partial workspace.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("decode_sm120_configurations.zig");
const mma = @import("mma_sm120.zig");
const tma = @import("tma.zig");
const pipeline = @import("pipeline.zig");
const bf16 = @import("bf16.zig");

const Recipe = configurations.Recipe;
const Mbarrier = nvptx.Mbarrier;
const Chunk = @Vector(4, u32);
const atom_bytes = 512;
const atom_rows = 128;
const weight_rows = configurations.weight_rows;
const block_k = configurations.block_k;
const row_chunks = block_k / 16;
const k_steps = block_k / 32;
const weight_bytes = weight_rows * block_k;
const zero: mma.Accumulator = @splat(0);

extern var storage: [0]u8 align(16) addrspace(.shared);

const Problem = struct {
    out: u64,
    bias: u64,
    residual: u64,
    scale_x: u64,
    scale_w: u64,
    partials: u64,
    m: u32,
    n: u32,
    padded_n: u32,
    k_blocks: u32,
    ctas: u32,
};

const Range = struct {
    first: u32,
    end: u32,
};

fn waitBounded(barrier: *addrspace(.shared) Mbarrier, phase: u1) void {
    if (barrier.tryWaitParity(phase)) return;
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitParity(phase)) {
        if (nvptx.globaltimer() -% start > configurations.wait_limit_ns) @trap();
    }
}

/// The byte of `(row, block)` in a 4-D scale tensor whose rows hold `atoms_per_row` atoms.
inline fn atomIndex(row: u32, block: u32, atoms_per_row: u32) u32 {
    const atom = row / atom_rows * atoms_per_row + block / 4;
    return atom * atom_bytes + row % 32 * 16 + row % atom_rows / 32 * 4 + block % 4;
}

/// TMA writes the 16-byte chunk `c` of row `r` of a 128-byte swizzled tile at chunk `c ^ (r % 8)`
/// of the row (PTX ISA 5.5.7).
inline fn swizzle(row: u32, column: u32) u32 {
    return row * row_chunks + (column ^ row % 8);
}

fn Decode(comptime tokens: u32) type {
    return struct {
        const stages = configurations.stages(tokens);
        const stage_bytes = configurations.stageBytes(tokens);
        const n_tiles = tokens / 8;
        const scale_words = (n_tiles + 3) / 4;
        const Ring = pipeline.Pipeline(stages, .cta);
        const Sums = [n_tiles]mma.Accumulator;

        var ring: Ring addrspace(.shared) = undefined;

        const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = configurations.threads },
        } };

        fn range(p: Problem) Range {
            const units: u64 = @as(u64, p.padded_n / weight_rows) * p.k_blocks;
            const c: u64 = @workGroupId(0);
            return .{
                .first = @intCast(c * units / p.ctas),
                .end = @intCast((c + 1) * units / p.ctas),
            };
        }

        fn produce(
            map_w: *const tma.TensorMap,
            map_x: *const tma.TensorMap,
            base: u32,
            p: Problem,
            r: Range,
        ) void {
            const weights = tma.createPolicy(.evict_first);
            const activations = tma.createPolicy(.evict_last);
            var slot: Ring.Slot = .{};
            var unit = r.first;
            while (unit < r.end) : (unit += 1) {
                waitBounded(ring.emptyBarrier(slot), ~slot.phase);
                ring.producerExpect(slot, stage_bytes);
                const stage = base + slot.index * stage_bytes;
                const k: i32 = @intCast(unit % p.k_blocks * block_k);
                const row: i32 = @intCast(unit / p.k_blocks * weight_rows);
                const full = ring.fullBarrier(slot);
                tma.load(2, stage, map_w, .{ k, row }, full, weights);
                tma.load(2, stage + weight_bytes, map_x, .{ k, 0 }, full, activations);
                slot = slot.next();
            }
        }

        fn Consumer(comptime recipe: Recipe) type {
            return struct {
                const BlockScales = switch (recipe) {
                    .mxfp8, .fp8_b128_e8m0 => struct { w: u32, x: [scale_words]u32 },
                    .fp8_b128_fp32 => struct { w: f32, x: [n_tiles][2]f32 },
                    .fp8_rowwise => void,
                };

                /// With the thread selector 0 for the weight fragment and `j % 4` for token
                /// fragment `j`, the block-scaled MMA takes the byte of weight row `r` from work
                /// item `4 (r % 8) + r / 8` and the byte of token `t` of fragment `j` from work
                /// item `4 t + j % 4` (PTX ISA 9.7.16.3).
                inline fn blockScales(p: Problem, unit: u32, warp: u32, lane: u32) BlockScales {
                    const tile = unit / p.k_blocks;
                    const block = unit % p.k_blocks;
                    const span = (p.k_blocks + 3) / 4;
                    const weight_index = tile * (4 * span) + block;
                    switch (recipe) {
                        .mxfp8 => {
                            const words_w: [*]addrspace(.global) const u32 = @ptrFromInt(p.scale_w);
                            const words_x: [*]addrspace(.global) const u32 = @ptrFromInt(p.scale_x);
                            const row = tile * weight_rows + warp * 16 + lane / 4 + lane % 2 * 8;
                            var scales: BlockScales = undefined;
                            scales.w = words_w[atomIndex(row, block * 4, p.k_blocks) / 4];
                            inline for (&scales.x, 0..) |*word, quad| {
                                const token = @as(u32, quad) * 32 + lane % 4 * 8 + lane / 4;
                                word.* = words_x[atomIndex(token, block * 4, p.k_blocks) / 4];
                            }
                            return scales;
                        },
                        .fp8_b128_e8m0 => {
                            const bytes_w: [*]addrspace(.global) const u8 = @ptrFromInt(p.scale_w);
                            const bytes_x: [*]addrspace(.global) const u8 = @ptrFromInt(p.scale_x);
                            var scales: BlockScales = undefined;
                            scales.w = bytes_w[weight_index];
                            inline for (&scales.x, 0..) |*word, quad| {
                                const token = @as(u32, quad) * 32 + lane % 4 * 8 + lane / 4;
                                word.* = bytes_x[atomIndex(token, block, span)];
                            }
                            return scales;
                        },
                        .fp8_b128_fp32 => {
                            const Factors = [*]addrspace(.global) const f32;
                            const factors_w: Factors = @ptrFromInt(p.scale_w);
                            const factors_x: Factors = @ptrFromInt(p.scale_x);
                            var scales: BlockScales = undefined;
                            scales.w = factors_w[weight_index];
                            inline for (&scales.x, 0..) |*pair, j| {
                                inline for (pair, 0..) |*factor, e| {
                                    const token = @as(u32, j) * 8 + lane % 4 * 2 + @as(u32, e);
                                    factor.* = factors_x[atomIndex(token, block, span)];
                                }
                            }
                            return scales;
                        },
                        .fp8_rowwise => return {},
                    }
                }

                inline fn multiply(
                    stage: [*]addrspace(.shared) Chunk,
                    a_lane: u32,
                    b_lane: u32,
                    scales: BlockScales,
                    sums: *Sums,
                ) void {
                    @setEvalBranchQuota(100_000);
                    const x_stage = stage + weight_bytes / 16;
                    var partial: Sums = @splat(zero);
                    const target = if (recipe == .fp8_b128_fp32) &partial else sums;
                    inline for (0..k_steps) |s| {
                        const a: mma.A = nvptx.ldmatrix(.x4, false, &stage[a_lane ^ s * 2]);
                        var b: [n_tiles]mma.B = undefined;
                        inline for (0..n_tiles / 2) |j| {
                            const pair = nvptx.ldmatrix(.x4, false, &x_stage[
                                (b_lane ^ s * 2) + j * 16 * row_chunks
                            ]);
                            b[j * 2] = .{ pair[0], pair[1] };
                            b[j * 2 + 1] = .{ pair[2], pair[3] };
                        }
                        inline for (0..n_tiles) |j| {
                            target[j] = switch (recipe) {
                                .mxfp8, .fp8_b128_e8m0 => mma.mmaBlockScaled(
                                    .{
                                        .a_byte = if (recipe == .mxfp8) s else 0,
                                        .a_thread = 0,
                                        .b_byte = if (recipe == .mxfp8) s else 0,
                                        .b_thread = j % 4,
                                    },
                                    a,
                                    b[j],
                                    target[j],
                                    scales.w,
                                    scales.x[j / 4],
                                ),
                                .fp8_b128_fp32, .fp8_rowwise => mma.mma(a, b[j], target[j]),
                            };
                        }
                    }
                    if (recipe == .fp8_b128_fp32) {
                        inline for (0..n_tiles) |j| {
                            inline for (0..4) |e| {
                                const factor = scales.x[j][e % 2] * scales.w;
                                sums[j][e] = @mulAdd(f32, partial[j][e], factor, sums[j][e]);
                            }
                        }
                    }
                }

                /// Accumulator element `2 h + e` of fragment `j` holds weight row
                /// `lane / 4 + 8 h` and token `8 j + 2 (lane % 4) + e` (PTX ISA 9.7.16.5.14).
                inline fn store(
                    p: Problem,
                    tile: u32,
                    warp: u32,
                    lane: u32,
                    sums: *const Sums,
                ) void {
                    const out: [*]addrspace(.global) u16 = @ptrFromInt(p.out);
                    const bias: [*]addrspace(.global) const u16 = @ptrFromInt(p.bias);
                    const residual: [*]addrspace(.global) const u16 = @ptrFromInt(p.residual);
                    const token_scales: [*]addrspace(.global) const f32 = @ptrFromInt(p.scale_x);
                    const weight_scales: [*]addrspace(.global) const f32 = @ptrFromInt(p.scale_w);
                    inline for (0..2) |h| {
                        const column = tile * weight_rows + warp * 16 + lane / 4 + h * 8;
                        const inside = column < p.n;
                        const weight_factor = if (recipe == .fp8_rowwise and inside)
                            weight_scales[column]
                        else
                            1;
                        const bias_term = if (p.bias != 0 and inside)
                            bf16.toF32(bias[column])
                        else
                            0;
                        inline for (0..n_tiles) |j| {
                            inline for (0..2) |e| {
                                const token = @as(u32, j) * 8 + lane % 4 * 2 + @as(u32, e);
                                if (token < p.m and inside) {
                                    var x = sums[j][h * 2 + e];
                                    if (recipe == .fp8_rowwise) {
                                        x = x * token_scales[token] * weight_factor;
                                    }
                                    const token64: u64 = token;
                                    if (p.residual != 0) {
                                        x += bf16.toF32(residual[token64 * p.padded_n + column]);
                                    }
                                    if (p.bias != 0) x += bias_term;
                                    out[token64 * p.n + column] = bf16.fromF32(x);
                                }
                            }
                        }
                    }
                }

                inline fn storePartial(
                    p: Problem,
                    slot: u32,
                    warp: u32,
                    lane: u32,
                    sums: *const Sums,
                ) void {
                    const partials: [*]addrspace(.global) f32 = @ptrFromInt(p.partials);
                    const first: u64 = (@as(u64, @workGroupId(0)) * 2 + slot) * tokens;
                    inline for (0..2) |h| {
                        const row = warp * 16 + lane / 4 + h * 8;
                        inline for (0..n_tiles) |j| {
                            inline for (0..2) |e| {
                                const token = @as(u32, j) * 8 + lane % 4 * 2 + @as(u32, e);
                                if (token < p.m) {
                                    const index = (first + token) * weight_rows + row;
                                    partials[index] = sums[j][h * 2 + e];
                                }
                            }
                        }
                    }
                }

                fn run(p: Problem, r: Range, base: u32, warp: u32, lane: u32) void {
                    const a_lane = swizzle(warp * 16 + lane % 16, lane / 16);
                    const b_lane = swizzle(lane % 8 + lane / 16 * 8, lane / 8 % 2);
                    const first_tile = r.first / p.k_blocks;
                    var sums: Sums = @splat(zero);
                    var slot: Ring.Slot = .{};
                    var scales = if (r.first < r.end)
                        blockScales(p, r.first, warp, lane)
                    else
                        undefined;
                    var unit = r.first;
                    while (unit < r.end) : (unit += 1) {
                        const next_scales = blockScales(p, @min(unit + 1, r.end - 1), warp, lane);
                        waitBounded(ring.fullBarrier(slot), slot.phase);
                        const stage: [*]addrspace(.shared) Chunk =
                            @ptrFromInt(base + slot.index * stage_bytes);
                        multiply(stage, a_lane, b_lane, scales, &sums);
                        nvptx.barWarpSync(.all);
                        if (lane == 0) ring.consumerRelease(slot);
                        slot = slot.next();
                        scales = next_scales;
                        const tile = unit / p.k_blocks;
                        const tile_end = (tile + 1) * p.k_blocks;
                        if (unit + 1 == tile_end or unit + 1 == r.end) {
                            const whole = tile * p.k_blocks >= r.first and tile_end <= r.end;
                            if (whole) {
                                store(p, tile, warp, lane, &sums);
                            } else {
                                const partial_slot: u32 = @intFromBool(tile != first_tile);
                                storePartial(p, partial_slot, warp, lane, &sums);
                            }
                            sums = @splat(zero);
                        }
                    }
                }
            };
        }

        fn Kernel(comptime recipe: Recipe) type {
            return struct {
                fn kernel(
                    map_w: tma.TensorMap,
                    map_x: tma.TensorMap,
                    out: u64,
                    bias: u64,
                    residual: u64,
                    scale_x: u64,
                    scale_w: u64,
                    partials: u64,
                    m: u32,
                    n: u32,
                    padded_n: u32,
                    k: u32,
                    ctas: u32,
                ) callconv(kernel_convention) void {
                    const p: Problem = .{
                        .out = out,
                        .bias = bias,
                        .residual = residual,
                        .scale_x = scale_x,
                        .scale_w = scale_w,
                        .partials = partials,
                        .m = m,
                        .n = n,
                        .padded_n = padded_n,
                        .k_blocks = k / block_k,
                        .ctas = ctas,
                    };
                    const thread = @workItemId(0);
                    const warp = thread / 32;
                    const lane = thread % 32;
                    const base = std.mem.alignForward(
                        u32,
                        tma.sharedAddress(&storage),
                        configurations.swizzle_alignment,
                    );
                    if (thread == 0) {
                        tma.prefetch(&map_w);
                        tma.prefetch(&map_x);
                        ring.init(1, configurations.consumer_warps);
                    }
                    nvptx.barrierSyncAligned(.work_group, null);
                    const r = range(p);
                    if (warp == configurations.consumer_warps) {
                        if (lane == 0) produce(&map_w, &map_x, base, p, r);
                        return;
                    }
                    Consumer(recipe).run(p, r, base, warp, lane);
                }
            };
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Recipe)) |recipe| {
            for (configurations.token_tiles) |tokens| {
                @export(&Decode(tokens).Kernel(recipe).kernel, .{
                    .name = configurations.kernelName(recipe, tokens),
                });
            }
        }
    }
}
