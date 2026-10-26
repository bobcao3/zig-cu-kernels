//! The product `A Bᵀ` of two FP8 packs on compute capability 12.x. One producer warp copies
//! 128-deep K stages of `A` and `B` by TMA into a ring of shared buffers. Eight consumer warps read
//! fragments with `ldmatrix`, multiply with `mma.sync.m16n8k32` and store BF16 results.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("mm_sm120_configurations.zig");
const mma = @import("mma_sm120.zig");
const tma = @import("tma.zig");
const pipeline = @import("pipeline.zig");
const bf16 = @import("bf16.zig");
const fp8 = @import("fp8.zig");
const fp8_pack = @import("fp8_pack.zig");
const math = nvptx.math;

const Recipe = configurations.Recipe;
const Tile = configurations.Tile;
const Epilogue = configurations.Epilogue;
const Mbarrier = nvptx.Mbarrier;
const Chunk = @Vector(4, u32);
const atom_bytes = 512;
const atom_rows = 128;
const zero: mma.Accumulator = @splat(0);
const negative_log2e: f32 = @bitCast(@as(u32, 0xbfb8_aa3b));
const gelu_linear: f32 = @bitCast(@as(u32, 0x4013_5761));
const gelu_cubic: f32 = @bitCast(@as(u32, 0x3dd2_d3e8));

extern var storage: [0]u8 align(16) addrspace(.shared);

const Problem = struct {
    out: u64,
    bias: u64,
    residual: u64,
    scale_a: u64,
    scale_b: u64,
    m: u32,
    n: u32,
    padded_n: u32,
    k_blocks: u32,
    low: u64 = 0,
    up: u64 = 0,
    rank: u32 = 0,
    h_scales: u64 = 0,
    z: u64 = 0,
    half: u32 = 0,
    hidden: u32 = 0,
    gate_first: bool = false,
    limit: f32 = 0,
    q: u64 = 0,
    kv: u64 = 0,
    qk_scales: u64 = 0,
    scales_bf16: bool = false,
    rope: u64 = 0,
    attn_dim: u32 = 0,
    rope_width: u32 = 0,
    eps: f32 = 0,
    partials: u64 = 0,
    splits: u32 = 0,
    split: u32 = 0,
    split_band: u32 = 0,
    k_first: u32 = 0,
    k_total: u32 = 0,
};

const head_dim = configurations.qkv_head_dim;
const inverse_head_dim: f32 = 1.0 / @as(f32, head_dim);

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
    const half = math.fmaF32(.nearest_even, .{}, sign, magnitude, 0.5);
    const gelu = bf16.round(math.mulF32(.nearest_even, .{}, gate, half));
    return bf16.round(math.mulF32(.nearest_even, .{}, gelu, up));
}

const bf16_mma: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k16,
    .accumulator = .f32,
    .element = .bf16,
};

inline fn shuffleIndex(x: u32, source: u32) u32 {
    return nvptx.shflSync(.idx, u32, x, source, .{ .clamp = 31, .segment_mask = 0 }, .all);
}

/// The rotation of `qkv_prep.zig`: `(y0 c - y1 s, y0 s + y1 c)` without fused multiply-adds.
inline fn rotate(y: [2]f32, turn: @Vector(2, f32)) [2]f32 {
    return .{
        bf16.round(math.subF32(
            .nearest_even,
            .{},
            math.mulF32(.nearest_even, .{}, y[0], turn[0]),
            math.mulF32(.nearest_even, .{}, y[1], turn[1]),
        )),
        bf16.round(math.addF32(
            .nearest_even,
            .{},
            math.mulF32(.nearest_even, .{}, y[0], turn[1]),
            math.mulF32(.nearest_even, .{}, y[1], turn[0]),
        )),
    };
}

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

fn Gemm(comptime tile: Tile, comptime epilogue: Epilogue) type {
    return struct {
        const glu = epilogue == .swiglu or epilogue == .geglu_tanh;
        const split_k = epilogue == .split;
        /// A GLU tile computes `glu_columns` columns of `h`. Each warp multiplies its 32 columns
        /// of `lo` and of `hi`, which the B stage holds as boxes of 32 rows: `lo`, `hi`, `lo`...
        const glu_columns = tile_n / 2;
        const glu_box = 32;
        const Ring = pipeline.Pipeline(tile.stages, .cta);
        const tile_m = tile.m;
        const tile_n = tile.n;
        const tile_k = tile.k;
        const stage_bytes = tile.stageBytes();
        const row_chunks = tile_k / 16;
        const warp_m = tile_m / tile.warps_m;
        const warp_n = tile_n / tile.warps_n;
        const m_tiles = warp_m / 16;
        const n_tiles = warp_n / 8;
        const k_steps = tile_k / 32;
        const a_bytes = tile_m * tile_k;
        const Accumulators = [m_tiles][n_tiles]mma.Accumulator;

        var ring: Ring addrspace(.shared) = undefined;

        /// TMA writes the 16-byte chunk `c` of row `r` of a 128-byte swizzled tile at chunk
        /// `c ^ (r % 8)` of the row (PTX ISA 5.5.7).
        inline fn swizzle(row: u32, column: u32) u32 {
            return row * row_chunks + (column ^ row % 8);
        }

        const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = tile.threads() },
        } };

        const Loader = struct {
            map_a: *const tma.TensorMap,
            map_b: *const tma.TensorMap,
            base: u32,
            tile_row: u32,
            tile_column: u32,
            half: u32,
            k_first: u32 = 0,

            fn load(loader: Loader, slot: Ring.Slot, block: u32) void {
                waitBounded(ring.emptyBarrier(slot), ~slot.phase);
                ring.producerExpect(slot, stage_bytes);
                const stage = loader.base + slot.index * stage_bytes;
                const first = if (split_k) loader.k_first else 0;
                const k: i32 = @intCast((first + block) * tile_k);
                const full = ring.fullBarrier(slot);
                tma.load(2, stage, loader.map_a, .{ k, @intCast(loader.tile_row) }, full, null);
                if (glu) {
                    inline for (0..tile_n / glu_box) |box| {
                        const hi = box % 2 * loader.half;
                        const row: i32 = @intCast(hi + loader.tile_column + box / 2 * glu_box);
                        const target = stage + a_bytes + @as(u32, box) * glu_box * tile_k;
                        tma.load(2, target, loader.map_b, .{ k, row }, full, null);
                    }
                } else {
                    const column: i32 = @intCast(loader.tile_column);
                    tma.load(2, stage + a_bytes, loader.map_b, .{ k, column }, full, null);
                }
            }
        };

        fn produce(loader: Loader, first: u32, k_blocks: u32) void {
            var slot: Ring.Slot = .{};
            slot = slot.advance(first);
            var block = first;
            while (block < k_blocks) : (block += 1) {
                loader.load(slot, block);
                slot = slot.next();
            }
        }

        fn Consumer(comptime recipe: Recipe) type {
            return struct {
                const scaled_rows = recipe == .fp8_b128_fp32 or recipe == .fp8_rowwise;
                const WeightFactor = if (glu) [2]f32 else f32;

                /// Work item `lane` holds rows `lane / 4` and `lane / 4 + 8` of each accumulator
                /// fragment (PTX ISA 9.7.16.5.10).
                inline fn fragmentRow(
                    warp_row: u32,
                    comptime i: u32,
                    comptime h: u32,
                    lane: u32,
                ) u32 {
                    return warp_row + i * 16 + lane / 4 + h * 8;
                }

                inline fn scaleRow(row0: u32, comptime pair: u32, lane: u32) u32 {
                    return row0 + (pair * 2 + lane / 2 % 2) * 16 + lane / 4 + lane % 2 * 8;
                }

                /// The scales of one K block: E8M0 words for the block-scaled MMA or FP32
                /// factors.
                const BlockScales = switch (recipe) {
                    .mxfp8, .fp8_b128_e8m0 => struct { a: [m_tiles / 2]u32, b: [n_tiles / 4]u32 },
                    .fp8_b128_fp32 => struct { a: [m_tiles][2]f32, b: WeightFactor },
                    .fp8_rowwise => void,
                };

                /// With the thread selectors `i % 2` for fragment `i` of `A` and `j % 4` for
                /// fragment `j` of `B`, the block-scaled MMA takes the byte of row `r` of
                /// fragment `i` from work item `4 (r % 8) + 2 (i % 2) + r / 8` and the byte of
                /// column `n` of fragment `j` from work item `4 n + j % 4` (PTX ISA 9.7.16.3).
                inline fn blockScales(
                    p: Problem,
                    block: u32,
                    row0: u32,
                    tile_column0: u32,
                    lane: u32,
                ) BlockScales {
                    const last_column0 = (p.n -| 1) / 128 * 128 + 128 - warp_n;
                    const column0 = if (tile_n > 128 and !glu)
                        @min(tile_column0, last_column0)
                    else
                        tile_column0;
                    const k_blocks = if (split_k) p.k_total else p.k_blocks;
                    const span = (k_blocks + 3) / 4;
                    const weight_index = column0 / 128 * (4 * span) + block;
                    const hi_index = (p.half + column0) / 128 * (4 * span) + block;
                    switch (recipe) {
                        .mxfp8 => {
                            const words_a: [*]addrspace(.global) const u32 = @ptrFromInt(p.scale_a);
                            const words_b: [*]addrspace(.global) const u32 = @ptrFromInt(p.scale_b);
                            var scales: BlockScales = undefined;
                            inline for (&scales.a, 0..) |*word, pair| {
                                const row = scaleRow(row0, pair, lane);
                                word.* = words_a[atomIndex(row, block * 4, k_blocks) / 4];
                            }
                            inline for (&scales.b, 0..) |*word, quad| {
                                const quad_column = if (glu)
                                    @as(u32, quad) * p.half + lane % 4 * 8
                                else
                                    @as(u32, quad) * 32 + lane % 4 * 8;
                                const column = column0 + quad_column + lane / 4;
                                word.* = words_b[atomIndex(column, block * 4, k_blocks) / 4];
                            }
                            return scales;
                        },
                        .fp8_b128_e8m0 => {
                            const bytes_a: [*]addrspace(.global) const u8 = @ptrFromInt(p.scale_a);
                            const bytes_b: [*]addrspace(.global) const u8 = @ptrFromInt(p.scale_b);
                            var scales: BlockScales = undefined;
                            inline for (&scales.a, 0..) |*word, pair| {
                                const row = scaleRow(row0, pair, lane);
                                word.* = bytes_a[atomIndex(row, block, span)];
                            }
                            scales.b = if (glu)
                                .{ bytes_b[weight_index], bytes_b[hi_index] }
                            else
                                @splat(bytes_b[weight_index]);
                            return scales;
                        },
                        .fp8_b128_fp32 => {
                            const Factors = [*]addrspace(.global) const f32;
                            const factors_a: Factors = @ptrFromInt(p.scale_a);
                            const factors_b: Factors = @ptrFromInt(p.scale_b);
                            var scales: BlockScales = undefined;
                            inline for (&scales.a, 0..) |*pair, i| {
                                inline for (pair, 0..) |*factor, h| {
                                    const row = row0 + fragmentRow(0, i, h, lane);
                                    factor.* = factors_a[atomIndex(row, block, span)];
                                }
                            }
                            scales.b = if (glu)
                                .{ factors_b[weight_index], factors_b[hi_index] }
                            else
                                factors_b[weight_index];
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
                    sums: *Accumulators,
                ) void {
                    @setEvalBranchQuota(100_000);
                    const groups = if (recipe == .fp8_b128_fp32) n_tiles / 4 else 1;
                    const group_n = n_tiles / groups;
                    const b_stage = stage + a_bytes / 16;
                    inline for (0..groups) |g| {
                        var partial: [m_tiles][group_n]mma.Accumulator = @splat(@splat(zero));
                        const target = if (recipe == .fp8_b128_fp32) &partial else sums;
                        inline for (0..k_steps) |s| {
                            var a: [m_tiles]mma.A = undefined;
                            inline for (&a, 0..) |*fragment, i| {
                                fragment.* = nvptx.ldmatrix(.x4, false, &stage[
                                    (a_lane ^ s * 2) + i * 16 * row_chunks
                                ]);
                            }
                            var b: [group_n]mma.B = undefined;
                            inline for (0..group_n / 2) |j| {
                                const pair = nvptx.ldmatrix(.x4, false, &b_stage[
                                    (b_lane ^ s * 2) + (g * group_n / 2 + j) * 16 * row_chunks
                                ]);
                                b[j * 2] = .{ pair[0], pair[1] };
                                b[j * 2 + 1] = .{ pair[2], pair[3] };
                            }
                            inline for (0..m_tiles) |i| {
                                inline for (0..group_n) |j| {
                                    target[i][j] = switch (recipe) {
                                        .mxfp8, .fp8_b128_e8m0 => mma.mmaBlockScaled(
                                            .{
                                                .a_byte = if (recipe == .mxfp8) s else 0,
                                                .a_thread = i % 2,
                                                .b_byte = if (recipe == .mxfp8) s else 0,
                                                .b_thread = j % 4,
                                            },
                                            a[i],
                                            b[j],
                                            target[i][j],
                                            scales.a[i / 2],
                                            scales.b[j / 4],
                                        ),
                                        .fp8_b128_fp32,
                                        .fp8_rowwise,
                                        => mma.mma(a[i], b[j], target[i][j]),
                                    };
                                }
                            }
                        }
                        if (recipe == .fp8_b128_fp32) {
                            inline for (0..m_tiles) |i| {
                                inline for (0..group_n) |j| {
                                    inline for (0..4) |e| {
                                        const weight = if (glu) scales.b[g] else scales.b;
                                        const factor = scales.a[i][e / 2] * weight;
                                        const sum = &sums[i][g * group_n + j][e];
                                        sum.* = @mulAdd(f32, partial[i][j][e], factor, sum.*);
                                    }
                                }
                            }
                        }
                    }
                }

                /// Adds `low[row] · up[column]` to the sums with BF16 MMAs. Each MMA takes the K
                /// values `2 (lane % 4)`, `+ 1`, `+ 8` and `+ 9` of its step from the rows
                /// `lane / 4` and `lane / 4 + 8` of `A` and from the column `lane / 4` of `B`
                /// (PTX ISA 9.7.16.5.8).
                inline fn addLowRank(
                    p: Problem,
                    row0: u32,
                    column0: u32,
                    lane: u32,
                    sums: *Accumulators,
                ) void {
                    const low: [*]addrspace(.global) const u32 = @ptrFromInt(p.low);
                    const up: [*]addrspace(.global) const u32 = @ptrFromInt(p.up);
                    const words: u64 = p.rank / 2;
                    var step: u32 = 0;
                    while (step < p.rank / 16) : (step += 1) {
                        const word = step * 8 + lane % 4;
                        var a: [m_tiles]mma.A = @splat(@splat(0));
                        inline for (&a, 0..) |*fragment, i| {
                            inline for (0..4) |r| {
                                const row = fragmentRow(row0, i, r % 2, lane);
                                if (row < p.m) fragment[r] = low[row * words + word + r / 2 * 4];
                            }
                        }
                        inline for (0..n_tiles) |j| {
                            const column = column0 + j * 8 + lane / 4;
                            var b: mma.B = @splat(0);
                            if (column < p.n) {
                                const first = column * words + word;
                                b = .{ up[first], up[first + 4] };
                            }
                            inline for (0..m_tiles) |i| {
                                sums[i][j] = nvptx.matrix.mmaSync(bf16_mma, a[i], b, sums[i][j]);
                            }
                        }
                    }
                }

                fn run(
                    p: Problem,
                    loader: ?Loader,
                    base: u32,
                    tile_row: u32,
                    tile_column: u32,
                    warp: u32,
                    lane: u32,
                ) void {
                    const warp_row = warp / tile.warps_n * warp_m;
                    const warp_column = warp % tile.warps_n * warp_n;
                    const a_lane = swizzle(warp_row + lane % 16, lane / 16);
                    const b_lane = swizzle(warp_column + lane % 8 + lane / 16 * 8, lane / 8 % 2);
                    const row0 = tile_row + warp_row;
                    const column0 = tile_column + if (glu) warp_column / 2 else warp_column;
                    var sums: Accumulators = @splat(@splat(zero));
                    var slot: Ring.Slot = .{};
                    const first = if (split_k) p.k_first else 0;
                    var scales = if (p.k_blocks > 0)
                        blockScales(p, first, row0, column0, lane)
                    else
                        undefined;
                    var block: u32 = 0;
                    while (block < p.k_blocks) : (block += 1) {
                        const next_block = first + @min(block + 1, p.k_blocks - 1);
                        const next_scales = blockScales(p, next_block, row0, column0, lane);
                        waitBounded(ring.fullBarrier(slot), slot.phase);
                        const stage: [*]addrspace(.shared) Chunk =
                            @ptrFromInt(base + slot.index * stage_bytes);
                        multiply(stage, a_lane, b_lane, scales, &sums);
                        nvptx.barWarpSync(.all);
                        if (lane == 0) ring.consumerRelease(slot);
                        if (!tile.producer_warp) {
                            if (loader) |l| {
                                if (block + tile.stages < p.k_blocks) {
                                    l.load(slot.advance(tile.stages), block + tile.stages);
                                }
                            }
                            nvptx.barWarpSync(.all);
                        }
                        slot = slot.next();
                        scales = next_scales;
                    }
                    if (epilogue == .low_rank) addLowRank(p, row0, column0, lane, &sums);
                    if (glu) {
                        storeGlu(p, base, row0, column0, warp, lane, &sums);
                    } else if (epilogue == .qkv_prep) {
                        storeQkv(p, base, row0, column0, warp, lane, &sums);
                    } else if (split_k and p.partials != 0) {
                        storePartial(p, row0, column0, lane, &sums);
                    } else {
                        store(p, row0, column0, lane, &sums);
                    }
                }

                /// Writes the q, k and v columns of `Z` that start at `column0` to `Q` and `KV`
                /// with the order of `qkv_prep.zig`: chunks of 8 columns summed from the lowest
                /// column up, then a pairwise tree over the 16 chunks of a head. The four work
                /// items of a row hold one chunk, and warps `w` and `w ^ 1` hold one head.
                inline fn storeQkv(
                    p: Problem,
                    base: u32,
                    row0: u32,
                    column0: u32,
                    warp: u32,
                    lane: u32,
                    sums: *const Accumulators,
                ) void {
                    const q: [*]addrspace(.global) u32 = @ptrFromInt(p.q);
                    const kv: [*]addrspace(.global) u32 = @ptrFromInt(p.kv);
                    const z: [*]addrspace(.global) u32 = @ptrFromInt(p.z);
                    const partials: [*]addrspace(.shared) f32 = @ptrFromInt(base);
                    const plane = column0 / p.attn_dim;
                    var turns: Turns = if (plane < 2)
                        loadTurns(p, fragmentRow(row0, 0, 0, lane), column0, lane)
                    else
                        undefined;
                    var x: [m_tiles][2][n_tiles]u32 = undefined;
                    inline for (0..m_tiles) |i| {
                        inline for (0..2) |h| {
                            inline for (0..n_tiles) |j| {
                                x[i][h][j] = bf16.pack(sums[i][j][h * 2], sums[i][j][h * 2 + 1]);
                            }
                        }
                    }
                    var coefficients: [m_tiles][2]f32 = undefined;
                    if (plane < 2) {
                        var halves: [m_tiles][2]f32 = undefined;
                        inline for (0..m_tiles) |i| {
                            inline for (0..2) |h| {
                                var chunks: [n_tiles]f32 = undefined;
                                inline for (0..n_tiles) |j| {
                                    const pair = bf16.unpack(x[i][h][j]);
                                    const squares = bf16.pack(
                                        math.mulF32(.nearest_even, .{}, pair[0], pair[0]),
                                        math.mulF32(.nearest_even, .{}, pair[1], pair[1]),
                                    );
                                    var sum: f32 = 0;
                                    inline for (0..4) |source| {
                                        const owner = lane / 4 * 4 + @as(u32, source);
                                        const terms = bf16.unpack(shuffleIndex(squares, owner));
                                        inline for (0..2) |e| {
                                            sum = math.addF32(.nearest_even, .{}, sum, terms[e]);
                                        }
                                    }
                                    chunks[j] = sum;
                                }
                                comptime var width = 1;
                                inline while (width < n_tiles) : (width *= 2) {
                                    inline for (0..n_tiles / (2 * width)) |pair| {
                                        const left = pair * 2 * width;
                                        chunks[left] = math.addF32(
                                            .nearest_even,
                                            .{},
                                            chunks[left],
                                            chunks[left + width],
                                        );
                                    }
                                }
                                halves[i][h] = chunks[0];
                            }
                        }
                        nvptx.barrierSyncAligned(.work_group, null);
                        if (lane % 4 == 3) {
                            inline for (0..m_tiles) |i| {
                                inline for (0..2) |h| {
                                    const slot = i * 16 + h * 8 + lane / 4;
                                    partials[warp * warp_m + slot] = halves[i][h];
                                }
                            }
                        }
                        nvptx.barrierSyncAligned(.work_group, null);
                        const scale_floats: [*]addrspace(.global) const f32 =
                            @ptrFromInt(p.qk_scales);
                        const scale_halves: [*]addrspace(.global) const u16 =
                            @ptrFromInt(p.qk_scales);
                        const head = column0 % p.attn_dim / head_dim;
                        const scale = if (p.scales_bf16)
                            bf16.toF32(scale_halves[head])
                        else
                            scale_floats[head];
                        inline for (0..m_tiles) |i| {
                            inline for (0..2) |h| {
                                const slot = i * 16 + h * 8 + lane / 4;
                                const sum = math.addF32(
                                    .nearest_even,
                                    .{},
                                    partials[warp * warp_m + slot],
                                    partials[(warp ^ 1) * warp_m + slot],
                                );
                                const mean = math.mulF32(.nearest_even, .{}, sum, inverse_head_dim);
                                const root = math.rsqrtApproxF32(
                                    true,
                                    math.addF32(.nearest_even, .{}, mean, p.eps),
                                );
                                coefficients[i][h] =
                                    bf16.round(math.mulF32(.nearest_even, .{}, scale, root));
                            }
                        }
                    }
                    inline for (0..m_tiles * 2) |r| {
                        const i = r / 2;
                        const h = r % 2;
                        var next_turns: Turns = undefined;
                        if (r + 1 < m_tiles * 2 and plane < 2) {
                            const next_row = fragmentRow(row0, (r + 1) / 2, (r + 1) % 2, lane);
                            next_turns = loadTurns(p, next_row, column0, lane);
                        }
                        const row = fragmentRow(row0, i, h, lane);
                        if (row < p.m) {
                            const row64: u64 = row;
                            inline for (0..n_tiles) |j| {
                                const column = column0 + j * 8 + lane % 4 * 2;
                                const pair = bf16.unpack(x[i][h][j]);
                                var y: [2]f32 = .{ pair[0], pair[1] };
                                if (p.z != 0) z[(row64 * p.n + column) / 2] = x[i][h][j];
                                if (plane < 2) {
                                    const c = coefficients[i][h];
                                    inline for (&y) |*value| {
                                        value.* = bf16.round(
                                            math.mulF32(.nearest_even, .{}, value.*, c),
                                        );
                                    }
                                    if (column % head_dim + 1 < p.rope_width) {
                                        y = rotate(y, turns[j]);
                                    }
                                }
                                const plane_column = column - plane * p.attn_dim;
                                const out = bf16.pack(y[0], y[1]);
                                if (plane == 0) {
                                    q[(row64 * p.attn_dim + plane_column) / 2] = out;
                                } else {
                                    const kv_column = plane_column + (plane - 1) * p.attn_dim;
                                    kv[(row64 * 2 * p.attn_dim + kv_column) / 2] = out;
                                }
                            }
                        }
                        turns = next_turns;
                    }
                }

                const Turns = [n_tiles]@Vector(2, f32);

                /// Loads the `(cos, sin)` pairs of the rotary position embedding for the columns
                /// that work item `lane` holds in `row`. The store loop of `storeQkv` loads the
                /// pairs of the next row before it stores the current row, so that the loads of a
                /// row overlap the stores of the row before.
                inline fn loadTurns(p: Problem, row: u32, column0: u32, lane: u32) Turns {
                    const rope: [*]addrspace(.global) const @Vector(2, f32) = @ptrFromInt(p.rope);
                    var turns: Turns = undefined;
                    if (row < p.m) {
                        inline for (&turns, 0..) |*turn, j| {
                            const first = (column0 + j * 8 + lane % 4 * 2) % head_dim;
                            if (first + 1 < p.rope_width) {
                                turn.* = rope[(@as(u64, row) * p.rope_width + first) / 2];
                            }
                        }
                    }
                    return turns;
                }

                const glu_values = n_tiles;
                const GluRows = [m_tiles * 2][glu_values]f32;
                const GluFactors = if (recipe == .fp8_rowwise) struct {
                    rows: [m_tiles * 2]@Vector(2, f32),
                    columns: [n_tiles]@Vector(2, f32),
                } else void;

                /// Loads the row scales of the fragment rows and the column scales of the `lo`
                /// and `hi` columns that work item `lane` holds.
                inline fn gluFactors(p: Problem, row0: u32, column0: u32, lane: u32) GluFactors {
                    if (recipe != .fp8_rowwise) return {};
                    const row_scales: [*]addrspace(.global) const f32 = @ptrFromInt(p.scale_a);
                    const column_scales: [*]addrspace(.global) const f32 = @ptrFromInt(p.scale_b);
                    var factors: GluFactors = undefined;
                    inline for (&factors.rows, 0..) |*factor, r| {
                        const row = fragmentRow(row0, r / 2, r % 2, lane);
                        factor.* = @splat(if (row < p.m) row_scales[row] else 1);
                    }
                    inline for (&factors.columns, 0..) |*factor, j| {
                        const hi = j / (n_tiles / 2) * p.half;
                        const column = hi + column0 + j % (n_tiles / 2) * 8 + lane % 4 * 2;
                        factor.* = .{ column_scales[column], column_scales[column + 1] };
                    }
                    return factors;
                }

                /// Returns the `h` values of fragment row `r`: the GLU products of the BF16 `lo`
                /// and `hi` columns, which the epilogue also writes to the rows of `Z` when `p.z`
                /// is set. The rowwise values of `Z` are `(acc · row scale) · column scale`, as in
                /// `store`.
                inline fn gluRow(
                    p: Problem,
                    comptime r: u32,
                    row0: u32,
                    column0: u32,
                    lane: u32,
                    sums: *const Accumulators,
                    factors: GluFactors,
                ) [glu_values]f32 {
                    const z: [*]addrspace(.global) u32 = @ptrFromInt(p.z);
                    const i = r / 2;
                    const h = r % 2;
                    const row = fragmentRow(row0, i, h, lane);
                    const block_values = n_tiles / 2;
                    var values: [glu_values]f32 = undefined;
                    inline for (0..block_values) |j| {
                        const column = column0 + j * 8 + lane % 4 * 2;
                        var lo: [2]f32 = undefined;
                        var hi: [2]f32 = undefined;
                        if (recipe == .fp8_rowwise) {
                            const lo_sum = sums[i][j];
                            const hi_sum = sums[i][j + block_values];
                            var x: @Vector(2, f32) = .{ lo_sum[h * 2], lo_sum[h * 2 + 1] };
                            var y: @Vector(2, f32) = .{ hi_sum[h * 2], hi_sum[h * 2 + 1] };
                            x = x * factors.rows[r] * factors.columns[j];
                            y = y * factors.rows[r] * factors.columns[j + block_values];
                            inline for (0..2) |e| {
                                lo[e] = bf16.round(x[e]);
                                hi[e] = bf16.round(y[e]);
                            }
                        } else {
                            inline for (0..2) |e| {
                                lo[e] = bf16.round(sums[i][j][h * 2 + e]);
                                hi[e] = bf16.round(sums[i][j + block_values][h * 2 + e]);
                            }
                        }
                        if (p.z != 0 and row < p.m and column < p.hidden) {
                            const index = (@as(u64, row) * 2 * p.hidden + column) / 2;
                            z[index] = bf16.pack(lo[0], lo[1]);
                            z[index + p.hidden / 2] = bf16.pack(hi[0], hi[1]);
                        }
                        inline for (0..2) |e| {
                            const up = if (p.gate_first) hi[e] else lo[e];
                            const gate = if (p.gate_first) lo[e] else hi[e];
                            const product = switch (epilogue) {
                                .swiglu => swiglu(up, gate, p.limit),
                                .geglu_tanh => gegluTanh(up, gate),
                                else => comptime unreachable,
                            };
                            const inside = row < p.m and column + e < p.hidden;
                            values[j * 2 + e] = if (inside) product else 0;
                        }
                    }
                    return values;
                }

                inline fn storeCodes(
                    p: Problem,
                    row: u32,
                    column0: u32,
                    lane: u32,
                    values: [glu_values]f32,
                    factor: f32,
                ) void {
                    const codes: [*]addrspace(.global) u16 = @ptrFromInt(p.out);
                    if (row < p.m) {
                        inline for (0..glu_values / 2) |j| {
                            const column = column0 + j * 8 + lane % 4 * 2;
                            const pair = nvptx.E4m3x4.fromF32(
                                math.mulF32(.nearest_even, .{}, values[j * 2], factor),
                                math.mulF32(.nearest_even, .{}, values[j * 2 + 1], factor),
                                0,
                                0,
                            );
                            const index = (@as(u64, row) * p.half + column) / 2;
                            codes[index] = @truncate(@backingInt(pair));
                        }
                    }
                }

                /// Writes the pack of the 32 columns of `h` that start at `column0`, or the BF16
                /// `h` for the rowwise recipe, whose row scales need whole rows of `h`. The four
                /// work items of a row hold the 32 values of an MXFP8 scale block. A block128
                /// block spans the four warps of a warp row, which exchange their maxima through
                /// the first 2 KB of the ring buffer. A block128 scale of a row from `m` to the end
                /// of the tile, or of a block after the last one up to the end of its scale atom,
                /// is 0, as in the padding of `glu_quant.zig`.
                inline fn storeGlu(
                    p: Problem,
                    base: u32,
                    row0: u32,
                    column0: u32,
                    warp: u32,
                    lane: u32,
                    sums: *const Accumulators,
                ) void {
                    const factors = gluFactors(p, row0, column0, lane);
                    switch (recipe) {
                        .mxfp8 => inline for (0..m_tiles * 2) |r| {
                            const scales: [*]addrspace(.global) u8 = @ptrFromInt(p.h_scales);
                            const row = fragmentRow(row0, r / 2, r % 2, lane);
                            const values = gluRow(p, r, row0, column0, lane, sums, factors);
                            const amax = fp8.maxNanOverLanes(fp8.amaxOf(values), 4);
                            const scaling: fp8_pack.Scaling(.e8m0) = .init(amax);
                            storeCodes(p, row, column0, lane, values, scaling.factor);
                            if (lane % 4 == 0) {
                                scales[atomIndex(row, column0 / 32, p.half / 128)] = scaling.scale;
                            }
                        },
                        .fp8_rowwise => inline for (0..m_tiles * 2) |r| {
                            const h: [*]addrspace(.global) u32 = @ptrFromInt(p.out);
                            const row = fragmentRow(row0, r / 2, r % 2, lane);
                            const values = gluRow(p, r, row0, column0, lane, sums, factors);
                            if (row < p.m) {
                                inline for (0..glu_values / 2) |j| {
                                    const column = column0 + j * 8 + lane % 4 * 2;
                                    const index = (@as(u64, row) * p.hidden + column) / 2;
                                    h[index] = bf16.pack(values[j * 2], values[j * 2 + 1]);
                                }
                            }
                        },
                        .fp8_b128_e8m0, .fp8_b128_fp32 => {
                            const format: fp8.ScaleFormat =
                                if (recipe == .fp8_b128_e8m0) .e8m0 else .f32;
                            const Scale = fp8.Scale(format);
                            const scales: [*]addrspace(.global) Scale = @ptrFromInt(p.h_scales);
                            const partials: [*]addrspace(.shared) f32 = @ptrFromInt(base);
                            var rows: GluRows = undefined;
                            var amaxes: [m_tiles * 2]f32 = undefined;
                            inline for (&rows, &amaxes, 0..) |*values, *amax, r| {
                                values.* = gluRow(p, r, row0, column0, lane, sums, factors);
                                amax.* = fp8.maxNanOverLanes(fp8.amaxOf(values.*), 4);
                            }
                            nvptx.barrierSyncAligned(.work_group, null);
                            if (lane % 4 == 0) {
                                inline for (amaxes, 0..) |amax, r| {
                                    partials[warp * warp_m + r * 8 + lane / 4] = amax;
                                }
                            }
                            nvptx.barrierSyncAligned(.work_group, null);
                            const first_warp = warp / tile.warps_n * tile.warps_n;
                            const block = column0 / 128;
                            const atoms_per_row = (p.half + 511) / 512;
                            const padding_block = block + warp % tile.warps_n;
                            const pads = block + 1 == p.half / 128 and
                                padding_block < 4 * atoms_per_row;
                            inline for (rows, 0..) |values, r| {
                                const row = fragmentRow(row0, r / 2, r % 2, lane);
                                const slot = r * 8 + lane / 4;
                                var amax = partials[first_warp * warp_m + slot];
                                inline for (1..tile.warps_n) |w| {
                                    const partial = partials[(first_warp + w) * warp_m + slot];
                                    amax = fp8.maxNan(amax, partial);
                                }
                                const scaling: fp8_pack.Scaling(format) = .init(amax);
                                storeCodes(p, row, column0, lane, values, scaling.factor);
                                if (lane % 4 == 0) {
                                    if (warp == first_warp) {
                                        const scale = if (row < p.m) scaling.scale else 0;
                                        scales[atomIndex(row, block, atoms_per_row)] = scale;
                                    } else if (pads) {
                                        scales[atomIndex(row, padding_block, atoms_per_row)] = 0;
                                    }
                                }
                            }
                        },
                    }
                }

                /// Writes the FP32 sums of split `p.split` to slot `2 c` of the partials
                /// `[c][2][128][128]` of `decode_reduce.zig`, with `c = column / 128 * splits +
                /// split`, so that the reduce kernel adds the splits of a tile in order.
                inline fn storePartial(
                    p: Problem,
                    row0: u32,
                    column0: u32,
                    lane: u32,
                    sums: *const Accumulators,
                ) void {
                    const partials: [*]addrspace(.global) @Vector(2, f32) = @ptrFromInt(p.partials);
                    const band_row0 = p.split_band * tile_m;
                    inline for (0..m_tiles) |i| {
                        inline for (0..2) |h| {
                            const row = fragmentRow(row0, i, h, lane);
                            if (row < p.m) {
                                inline for (0..n_tiles) |j| {
                                    const column = column0 + j * 8 + lane % 4 * 2;
                                    if (column < p.n) {
                                        const sum = sums[i][j];
                                        const c: u64 = column / 128 * p.splits + p.split;
                                        const offset = (2 * c * 128 + row - band_row0) * 128;
                                        partials[(offset + column % 128) / 2] =
                                            .{ sum[h * 2], sum[h * 2 + 1] };
                                    }
                                }
                            }
                        }
                    }
                }

                /// The epilogue adds the FP32 terms in the order `acc + residual + bias` and then
                /// rounds the sum to BF16.
                inline fn store(
                    p: Problem,
                    row0: u32,
                    column0: u32,
                    lane: u32,
                    sums: *const Accumulators,
                ) void {
                    const out: [*]addrspace(.global) u32 = @ptrFromInt(p.out);
                    const bias: [*]addrspace(.global) const u32 = @ptrFromInt(p.bias);
                    const residual: [*]addrspace(.global) const u32 = @ptrFromInt(p.residual);
                    const row_scales: [*]addrspace(.global) const f32 = @ptrFromInt(p.scale_a);
                    const column_scales: [*]addrspace(.global) const f32 = @ptrFromInt(p.scale_b);
                    var row_factors: [m_tiles][2]f32 = @splat(.{ 1, 1 });
                    var column_factors: [n_tiles]@Vector(2, f32) = @splat(@splat(1));
                    var bias_terms: [n_tiles]@Vector(2, f32) = @splat(@splat(0));
                    inline for (0..n_tiles) |j| {
                        const column = column0 + j * 8 + lane % 4 * 2;
                        if (column < p.n) {
                            if (recipe == .fp8_rowwise) {
                                column_factors[j] = .{
                                    column_scales[column],
                                    column_scales[column + 1],
                                };
                            }
                            if (p.bias != 0) bias_terms[j] = bf16.unpack(bias[column / 2]);
                        }
                    }
                    if (recipe == .fp8_rowwise) {
                        inline for (0..m_tiles) |i| {
                            inline for (0..2) |h| {
                                const row = fragmentRow(row0, i, h, lane);
                                if (row < p.m) row_factors[i][h] = row_scales[row];
                            }
                        }
                    }
                    inline for (0..m_tiles) |i| {
                        inline for (0..2) |h| {
                            const row = fragmentRow(row0, i, h, lane);
                            if (row < p.m) {
                                const row_factor: @Vector(2, f32) = @splat(row_factors[i][h]);
                                inline for (0..n_tiles) |j| {
                                    const column = column0 + j * 8 + lane % 4 * 2;
                                    if (column < p.n) {
                                        const sum = sums[i][j];
                                        var x: @Vector(2, f32) = .{ sum[h * 2], sum[h * 2 + 1] };
                                        if (recipe == .fp8_rowwise) {
                                            x = x * row_factor * column_factors[j];
                                        }
                                        const row64: u64 = row;
                                        if (p.residual != 0) {
                                            const index = (row64 * p.padded_n + column) / 2;
                                            x += bf16.unpack(residual[index]);
                                        }
                                        if (p.bias != 0) x += bias_terms[j];
                                        out[(row64 * p.n + column) / 2] = bf16.pack(x[0], x[1]);
                                    }
                                }
                            }
                        }
                    }
                }
            };
        }

        fn Kernel(comptime recipe: Recipe) type {
            return struct {
                fn kernel(
                    map_a: tma.TensorMap,
                    map_b: tma.TensorMap,
                    out: u64,
                    bias: u64,
                    residual: u64,
                    scale_a: u64,
                    scale_b: u64,
                    m: u32,
                    n: u32,
                    padded_n: u32,
                    k: u32,
                ) callconv(kernel_convention) void {
                    run(&map_a, &map_b, .{
                        .out = out,
                        .bias = bias,
                        .residual = residual,
                        .scale_a = scale_a,
                        .scale_b = scale_b,
                        .m = m,
                        .n = n,
                        .padded_n = padded_n,
                        .k_blocks = (k + tile_k - 1) / tile_k,
                    });
                }

                /// Work groups in the rows `split_band + s` of the grid compute the last row
                /// band over the K blocks of split `s` and write FP32 partials.
                fn kernelSplit(
                    map_a: tma.TensorMap,
                    map_b: tma.TensorMap,
                    out: u64,
                    bias: u64,
                    residual: u64,
                    scale_a: u64,
                    scale_b: u64,
                    m: u32,
                    n: u32,
                    padded_n: u32,
                    k: u32,
                    partials: u64,
                    splits: u32,
                ) callconv(kernel_convention) void {
                    const k_total = (k + tile_k - 1) / tile_k;
                    const split_band = (m + tile_m - 1) / tile_m - 1;
                    const split = @workGroupId(1) -| split_band;
                    const partial = @workGroupId(1) >= split_band;
                    const k_first = if (partial) split * k_total / splits else 0;
                    const k_end = if (partial) (split + 1) * k_total / splits else k_total;
                    run(&map_a, &map_b, .{
                        .out = out,
                        .bias = bias,
                        .residual = residual,
                        .scale_a = scale_a,
                        .scale_b = scale_b,
                        .m = m,
                        .n = n,
                        .padded_n = padded_n,
                        .k_blocks = k_end - k_first,
                        .partials = if (partial) partials else 0,
                        .splits = splits,
                        .split = split,
                        .split_band = split_band,
                        .k_first = k_first,
                        .k_total = k_total,
                    });
                }

                fn kernelLowRank(
                    map_a: tma.TensorMap,
                    map_b: tma.TensorMap,
                    out: u64,
                    bias: u64,
                    residual: u64,
                    scale_a: u64,
                    scale_b: u64,
                    m: u32,
                    n: u32,
                    padded_n: u32,
                    k: u32,
                    low: u64,
                    up: u64,
                    rank: u32,
                ) callconv(kernel_convention) void {
                    run(&map_a, &map_b, .{
                        .out = out,
                        .bias = bias,
                        .residual = residual,
                        .scale_a = scale_a,
                        .scale_b = scale_b,
                        .m = m,
                        .n = n,
                        .padded_n = padded_n,
                        .k_blocks = (k + tile_k - 1) / tile_k,
                        .low = low,
                        .up = up,
                        .rank = rank,
                    });
                }

                fn kernelGlu(
                    map_a: tma.TensorMap,
                    map_b: tma.TensorMap,
                    codes: u64,
                    h_scales: u64,
                    z: u64,
                    scale_a: u64,
                    scale_b: u64,
                    m: u32,
                    half: u32,
                    hidden: u32,
                    k: u32,
                    gate_first: u32,
                    limit: f32,
                ) callconv(kernel_convention) void {
                    run(&map_a, &map_b, .{
                        .out = codes,
                        .bias = 0,
                        .residual = 0,
                        .scale_a = scale_a,
                        .scale_b = scale_b,
                        .m = m,
                        .n = 2 * half,
                        .padded_n = 2 * half,
                        .k_blocks = (k + tile_k - 1) / tile_k,
                        .h_scales = h_scales,
                        .z = z,
                        .half = half,
                        .hidden = hidden,
                        .gate_first = gate_first != 0,
                        .limit = limit,
                    });
                }

                fn kernelQkv(
                    map_a: tma.TensorMap,
                    map_b: tma.TensorMap,
                    q: u64,
                    kv: u64,
                    z: u64,
                    scale_a: u64,
                    scale_b: u64,
                    qk_scales: u64,
                    rope: u64,
                    scales_bf16: u32,
                    m: u32,
                    attn_dim: u32,
                    k: u32,
                    rope_width: u32,
                    eps: f32,
                ) callconv(kernel_convention) void {
                    run(&map_a, &map_b, .{
                        .out = 0,
                        .bias = 0,
                        .residual = 0,
                        .scale_a = scale_a,
                        .scale_b = scale_b,
                        .m = m,
                        .n = 3 * attn_dim,
                        .padded_n = 3 * attn_dim,
                        .k_blocks = (k + tile_k - 1) / tile_k,
                        .z = z,
                        .q = q,
                        .kv = kv,
                        .qk_scales = qk_scales,
                        .scales_bf16 = scales_bf16 != 0,
                        .rope = rope,
                        .attn_dim = attn_dim,
                        .rope_width = rope_width,
                        .eps = eps,
                    });
                }

                inline fn run(
                    map_a: *const tma.TensorMap,
                    map_b: *const tma.TensorMap,
                    p: Problem,
                ) void {
                    const thread = @workItemId(0);
                    const warp = thread / 32;
                    const lane = thread % 32;
                    const base = std.mem.alignForward(
                        u32,
                        tma.sharedAddress(&storage),
                        configurations.swizzle_alignment,
                    );
                    if (thread == 0) {
                        tma.prefetch(map_a);
                        tma.prefetch(map_b);
                        ring.init(1, configurations.consumer_warps);
                    }
                    nvptx.barrierSyncAligned(.work_group, null);
                    const grid_row = @workGroupId(1);
                    const band = if (split_k) @min(grid_row, p.split_band) else grid_row;
                    const tile_row = band * tile_m;
                    const tile_column = @workGroupId(0) * if (glu) glu_columns else tile_n;
                    const loader: Loader = .{
                        .map_a = map_a,
                        .map_b = map_b,
                        .base = base,
                        .tile_row = tile_row,
                        .tile_column = tile_column,
                        .half = p.half,
                        .k_first = p.k_first,
                    };
                    if (tile.producer_warp and warp == configurations.consumer_warps) {
                        if (lane == 0) produce(loader, 0, p.k_blocks);
                        return;
                    }
                    if (!tile.producer_warp and thread == 0) {
                        produce(loader, 0, @min(p.k_blocks, tile.stages));
                    }
                    const refills = !tile.producer_warp and thread == 0;
                    const refill_loader: ?Loader = if (refills) loader else null;
                    const consumer = Consumer(recipe);
                    consumer.run(p, refill_loader, base, tile_row, tile_column, warp, lane);
                }
            };
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Recipe)) |recipe| {
            for (configurations.tiles) |tile| {
                @export(&Gemm(tile, .store).Kernel(recipe).kernel, .{
                    .name = configurations.kernelName(recipe, tile),
                });
            }
        }
        for (configurations.tiles) |tile| {
            if (configurations.hasSplit(tile)) {
                for (std.enums.values(Recipe)) |recipe| {
                    @export(&Gemm(tile, .split).Kernel(recipe).kernelSplit, .{
                        .name = configurations.splitKernelName(recipe, tile),
                    });
                }
            }
            @export(&Gemm(tile, .low_rank).Kernel(.mxfp8).kernelLowRank, .{
                .name = configurations.lowRankKernelName(tile),
            });
            if (!configurations.hasGlu(tile)) continue;
            @export(&Gemm(tile, .qkv_prep).Kernel(.mxfp8).kernelQkv, .{
                .name = configurations.qkvKernelName(tile),
            });
            for (std.enums.values(Recipe)) |recipe| {
                for ([_]Epilogue{ .swiglu, .geglu_tanh }) |activation| {
                    @export(&Gemm(tile, activation).Kernel(recipe).kernelGlu, .{
                        .name = configurations.gluKernelName(recipe, activation, tile),
                    });
                }
            }
        }
    }
}
