//! The kernels compute `C = A B^T` for E4M3 matrices `A` (`[m, k]`) and `B` (`[n, k]`) with FP32 or
//! E8M0 scales and BF16 outputs. Of `g` work groups, work group `i` computes the whole tiles `i`,
//! `i + g` and so on below `data_parallel_tiles`. Then the work group computes an even share of the
//! k tiles of the other tiles. The E4M3 `mma` of sm_89 accumulates with fewer bits than FP32, so a
//! `chains` kernel adds each chain of at most 64 values of k to FP32 sums.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const bf16 = @import("bf16.zig");
const fp8 = @import("fp8.zig");
const epilogue = @import("epilogue.zig");
const fp8_pack = @import("fp8_pack.zig");
const warp_functions = @import("warp.zig");
const configurations = @import("fp8_gemm_configurations.zig");

const Chunk = bf16.Chunk;
const Variant = configurations.Variant;
const Kind = configurations.Kind;
const Tiling = configurations.Tiling;
const scale_group = configurations.scale_group;
const mxfp8_atom_k = configurations.mxfp8_atom_k;
const ScaleFormat = configurations.ScaleFormat;
const GluActivation = configurations.GluActivation;

const mma_options: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k32,
    .element = .e4m3,
    .accumulator = .f32,
};

const low_rank_mma: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k16,
    .element = .bf16,
    .accumulator = .f32,
};

inline fn pick(comptime T: type, pair: [2]T, index: u32) T {
    return if (index == 1) pair[1] else pair[0];
}

fn Gemm(comptime variant: Variant, comptime kind: Kind, comptime tiling: Tiling) type {
    return struct {
        const block_scales = variant.scaling == .block128;
        const atom_scales = variant.scaling == .mxfp8;
        const fast = variant.accumulation == .fast;
        const halves = if (kind == .store) 1 else 2;
        const threads = tiling.threads();
        const tile_m = tiling.tile_m;
        const tile_n = tiling.tile_n;
        const tile_k = tiling.tile_k;
        const stages = tiling.stages;
        const warp_m = tiling.warp_m;
        const warp_n = tiling.warp_n;
        const warps_n = tile_n / warp_n;
        const warps = threads / 32;
        const k_warps = tiling.k_warps;
        const spatial_warps = warps / k_warps;
        const spatial_threads = spatial_warps * 32;
        const prefetch = stages - 1;
        const half_rows = tile_n / 2;
        const block_columns = tile_n / halves;
        const row_chunks = tile_k / 16;
        const k_steps = tile_k / 32 / k_warps;
        const chain_steps = if (atom_scales) 1 else @min(k_steps, 2);
        const chains = k_steps / chain_steps;
        const blocks_per_stage = @max(1, tile_k / scale_group);
        const stages_per_block = @max(1, scale_group / tile_k);
        const a_fragments = warp_m / 16;
        const b_fragments = warp_n / 8;
        const half_fragments = b_fragments / halves;
        const copy_rows = threads / row_chunks;
        const a_passes = tile_m / copy_rows;
        const b_passes = tile_n / copy_rows;
        const a_scale_copies = tile_m / 4;
        const slot_floats = tile_m * tile_n;

        comptime {
            if (tile_m % warp_m != 0 or tile_n % warp_n != 0 or scale_group % half_rows != 0) {
                @compileError("the warp tiles must divide the work group tile");
            }
            if (tile_n > scale_group and half_rows % warp_n != 0) {
                @compileError("a warp tile of a tile wider than a K block lies in one half");
            }
            if (tile_m % copy_rows != 0 or half_rows % copy_rows != 0 or copy_rows % 8 != 0) {
                @compileError("the copy passes must cover the work group tile in whole swizzle rows");
            }
            if (scale_group % tile_k != 0 and tile_k % scale_group != 0) {
                @compileError("a k tile holds a part of a K block or whole K blocks");
            }
            if (k_steps % chain_steps != 0 or k_steps * 32 > scale_group) {
                @compileError("the K steps of a warp lie in one K block");
            }
            if (warp_n % 32 != 0) @compileError("warp_n must be a multiple of 32");
            if (a_fragments % 2 != 0) @compileError("a warp tile has two halves of A fragments");
            if (block_scales and threads < (a_scale_copies + 2) * blocks_per_stage) {
                @compileError("the work items copy the scales of A and of both halves of B");
            }
            if (block_scales and fast) @compileError("block128 kernels scale each chain");
            if (atom_scales and (fast or kind == .gated or tile_k % mxfp8_atom_k != 0)) {
                @compileError("mxfp8 kernels store whole scale atoms of each chain");
            }
            const glu_tile = k_warps == 1 and half_fragments == 4 and
                (atom_scales or block_columns == scale_group);
            if (kind == .glu and !glu_tile) {
                @compileError("the warps of a row of warp tiles hold one scale block of h");
            }
        }

        const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = threads },
            .min_work_groups_per_sm = 1,
        } };

        const Stage = struct {
            a: [tile_m * row_chunks]Chunk,
            b: [tile_n * row_chunks]Chunk,
        };
        const Layout = struct {
            stages: [stages]Stage,
            a_scales: if (block_scales) [stages][blocks_per_stage][tile_m]f32 else void,
            b_scales: if (block_scales) [stages][blocks_per_stage][4]f32 else void,
            atoms: if (atom_scales) [stages][tile_k / mxfp8_atom_k][tile_m + tile_n]f32 else void,
            warp_scales: if (atom_scales) [warps][k_steps][warp_m + warp_n]f32 else void,
        };
        const pass_rows = passRows();
        const WarpSums = [k_warps - 1][pass_rows * b_fragments * spatial_threads]@Vector(4, f32);

        /// Returns the most A fragment rows whose sums of all warps but one fit in a stage.
        fn passRows() u32 {
            var rows: u32 = a_fragments;
            const row_bytes = (k_warps - 1) * b_fragments * spatial_threads * @sizeOf(@Vector(4, f32));
            while (rows > 1 and rows * row_bytes > @sizeOf(Stage)) rows /= 2;
            return rows;
        }
        comptime {
            if (@sizeOf(Layout) != tiling.sharedBytes(variant.scaling)) {
                @compileError("the launch configuration has another shared memory size");
            }
            if (@sizeOf(WarpSums) > @sizeOf(Stage)) @compileError("the warp sums exceed a stage");
        }

        extern var shared_storage: [0]u8 align(128) addrspace(.shared);
        const layout: *addrspace(.shared) Layout = @ptrCast(&shared_storage);
        var full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var last_part: bool addrspace(.shared) = undefined;

        const Accumulators = [a_fragments][b_fragments]@Vector(4, f32);
        const Partial = [b_fragments]@Vector(4, f32);
        const BlockFactors = if (block_scales) [a_fragments][halves][2]f32 else void;

        /// Row `r` of A has the scale with index `r * a_rows + j * a_blocks` in scale group `j`, or
        /// `atomIndex(r, j, a_blocks)` with `a_atoms`.
        /// Row `c` of half `h` of B has the index `c * b_rows[h]` (rowwise) or, in weight tile row
        /// `t`, `t * b_rows[h] + j * b_blocks[h]`. The `e8m0` tensors hold one byte per scale.
        const Scales = struct {
            a: u64,
            b: [2]u64,
            a_rows: u32,
            a_blocks: u32,
            b_rows: [2]u32,
            b_blocks: [2]u32,
            a_e8m0: bool,
            b_e8m0: [2]bool,
            a_atoms: bool,
            a_vectors: bool,
        };

        const Raster = struct {
            panel: u32,
            along_m: bool,
        };

        /// Half 1 of a store tile holds the B rows that follow the rows of half 0. Half 0 of a
        /// gated tile holds rows of `up`, and half 1 holds the `gate` rows of the same columns.
        const Problem = struct {
            a: [*]addrspace(.global) const u8,
            b: [2][*]addrspace(.global) const u8,
            a_stride: u32,
            b_stride: u32,
            scales: Scales,
            m: u32,
            columns: u32,
            k: u32,
            raster: Raster,
            tiles_m: u32,
            tiles_n: u32,

            inline fn tileCount(p: Problem) u32 {
                return p.tiles_m * p.tiles_n;
            }

            /// The raster order visits panels of `panel` tile rows, or tile columns with `along_m`.
            inline fn tileAt(p: Problem, index: u32) Tile {
                var row_tile: u32 = undefined;
                var column_tile: u32 = undefined;
                if (!p.raster.along_m) {
                    const first = index / (p.raster.panel * p.tiles_n) * p.raster.panel;
                    const height = @min(p.raster.panel, p.tiles_m - first);
                    const within = index - first * p.tiles_n;
                    row_tile = first + within % height;
                    column_tile = within / height;
                } else {
                    const first = index / (p.raster.panel * p.tiles_m) * p.raster.panel;
                    const width = @min(p.raster.panel, p.tiles_n - first);
                    const within = index - first * p.tiles_m;
                    column_tile = first + within % width;
                    row_tile = within / width;
                }
                const first_row = column_tile * block_columns;
                return .{
                    .m_offset = row_tile * tile_m,
                    .column_offset = first_row,
                    .b_rows = switch (kind) {
                        .store => .{ first_row, first_row + half_rows },
                        .gated, .glu => .{ first_row, first_row },
                    },
                };
            }

            inline fn weightIndex(p: Problem, half: u32, row: u32, block: u32) u64 {
                const tile_row = @min(row, p.columns - 1) / scale_group;
                return @as(u64, tile_row) * pick(u32, p.scales.b_rows, half) +
                    @as(u64, block) * pick(u32, p.scales.b_blocks, half);
            }
        };

        const Tile = struct {
            m_offset: u32,
            column_offset: u32,
            b_rows: [2]u32,
        };

        const Output = struct {
            out: [*]addrspace(.global) Chunk,
            out_stride: u32,
            bias: [halves]u64,
            residual: u64,
            residual_stride: u32,
            residual_f32: bool = false,
            activation: u32,
            glu: GluOutput = undefined,
            low_rank: LowRank = undefined,
        };

        /// `low` holds `m` rows and `up` holds `n` rows of `rank` BF16 values each.
        const LowRank = struct {
            low: u64,
            up: u64,
            rank: u32,
        };

        /// Row `r` of the pack of `h` holds `code_stride` codes from `codes + r * code_stride`, and
        /// its scales lie in atoms of `scale_spans` per 128 rows. A zero `z_up` or `z_gate` address
        /// stores no product.
        const GluOutput = struct {
            z_up: u64,
            z_gate: u64,
            z_stride: u32,
            codes: u64,
            code_stride: u32,
            scales: u64,
            scale_spans: u32,
            hidden: u32,
            scale_format: ScaleFormat,
            activation: GluActivation,
            limit: f32,
        };

        /// Stream-K tile `s` is k tile `s % k_tiles` of tile `data_parallel_tiles + s / k_tiles`.
        const Schedule = struct {
            k_tiles: u32,
            data_parallel_tiles: u32,
            stream_k_tiles: u32,
            begin: u32,
            end: u32,
            workspace: [*]allowzero addrspace(.global) @Vector(4, f32),
            counters: [*]allowzero addrspace(.global) u32,

            inline fn init(
                problem: Problem,
                data_parallel_tiles: u32,
                workspace: u64,
                counters: u64,
            ) Schedule {
                const k_tiles = (problem.k + tile_k - 1) / tile_k;
                const stream_k_tiles = (problem.tileCount() - data_parallel_tiles) * k_tiles;
                return .{
                    .k_tiles = k_tiles,
                    .data_parallel_tiles = data_parallel_tiles,
                    .stream_k_tiles = stream_k_tiles,
                    .begin = streamKTilesBefore(stream_k_tiles, @workGroupId(0)),
                    .end = streamKTilesBefore(stream_k_tiles, @workGroupId(0) + 1),
                    .workspace = @ptrFromInt(workspace),
                    .counters = @ptrFromInt(counters),
                };
            }

            inline fn streamKTilesBefore(stream_k_tiles: u32, work_group: u32) u32 {
                return stream_k_tiles * work_group / nvptx.nctaid(0);
            }

            inline fn owner(s: Schedule, stream_k_tile: u32) u32 {
                return ((stream_k_tile + 1) * nvptx.nctaid(0) + s.stream_k_tiles - 1) /
                    s.stream_k_tiles - 1;
            }

            inline fn steps(s: Schedule) u32 {
                const first = @workGroupId(0);
                const whole = if (first < s.data_parallel_tiles)
                    (s.data_parallel_tiles - first + nvptx.nctaid(0) - 1) / nvptx.nctaid(0)
                else
                    0;
                return whole * s.k_tiles + s.end - s.begin;
            }

            /// Slot `2 w` holds the first part of work group `w`, and slot `2 w + 1` its last part.
            inline fn slot(s: Schedule, work_group: u32, first_stream_k_tile: u32) u32 {
                return 2 * work_group +
                    @intFromBool(streamKTilesBefore(s.stream_k_tiles, work_group) < first_stream_k_tile);
            }
        };

        /// The current part of tile `tile` covers the k tiles `[k_begin, k_end)`.
        const Cursor = struct {
            tile: u32,
            k_tile: u32,
            k_begin: u32,
            k_end: u32,

            inline fn init(schedule: Schedule) Cursor {
                const first = @workGroupId(0);
                if (first < schedule.data_parallel_tiles) {
                    return .{ .tile = first, .k_tile = 0, .k_begin = 0, .k_end = schedule.k_tiles };
                }
                return .at(schedule, schedule.begin);
            }

            inline fn at(schedule: Schedule, stream_k_tile: u32) Cursor {
                const k_tile = stream_k_tile % schedule.k_tiles;
                return .{
                    .tile = schedule.data_parallel_tiles + stream_k_tile / schedule.k_tiles,
                    .k_tile = k_tile,
                    .k_begin = k_tile,
                    .k_end = @min(schedule.k_tiles, k_tile + (schedule.end -| stream_k_tile)),
                };
            }

            inline fn whole(cursor: Cursor, schedule: Schedule) bool {
                return cursor.k_begin == 0 and cursor.k_end == schedule.k_tiles;
            }

            inline fn firstBlock(cursor: Cursor) u32 {
                return cursor.k_tile * blocks_per_stage / stages_per_block;
            }

            inline fn next(cursor: *Cursor, schedule: Schedule) void {
                cursor.k_tile += 1;
                if (cursor.k_tile < cursor.k_end) return;
                if (cursor.tile < schedule.data_parallel_tiles) {
                    cursor.tile += nvptx.nctaid(0);
                    if (cursor.tile < schedule.data_parallel_tiles) {
                        cursor.k_tile = 0;
                    } else {
                        cursor.* = .at(schedule, schedule.begin);
                    }
                    return;
                }
                cursor.* = .at(
                    schedule,
                    (cursor.tile - schedule.data_parallel_tiles) * schedule.k_tiles + cursor.k_tile,
                );
            }
        };

        /// Assumes that `k` is a positive multiple of 16, or of 128 with `block128` scales. Assumes
        /// that `n % 8 == 0`. Assumes that `panel > 0`. Assumes that the rows of A, B, the output
        /// and the residual start at multiples of 16 bytes. Assumes that FP32 scales and the bias
        /// start at multiples of 4 bytes. Assumes that `data_parallel_tiles` is at most the tile
        /// count. Assumes that the stream-K tile count is 0 or at least the work group count.
        /// Assumes that the stream-K tile count times (work groups + 1) is below 2^32. Stream-K
        /// tiles need `2 * tile_m * tile_n` FP32 values of workspace per work group and a zero
        /// `u32` counter per tile from `data_parallel_tiles` on. A zero `bias` or `residual`
        /// address adds nothing. Bits 0 and 1 of `e8m0` mark E8M0 scales of A and B.
        fn storeKernel(
            a: [*]addrspace(.global) const u8,
            b: [*]addrspace(.global) const u8,
            scale_a: u64,
            scale_b: u64,
            out: [*]addrspace(.global) Chunk,
            bias: u64,
            residual: u64,
            workspace: u64,
            counters: u64,
            a_stride: u32,
            b_stride: u32,
            scale_a_rows: u32,
            scale_a_blocks: u32,
            scale_b_rows: u32,
            scale_b_blocks: u32,
            e8m0: u32,
            out_stride: u32,
            residual_stride: u32,
            m: u32,
            n: u32,
            k: u32,
            activation: u32,
            panel: u32,
            along_m: u32,
            data_parallel_tiles: u32,
        ) callconv(kernel_convention) void {
            runStore(false, a, b, scale_a, scale_b, out, bias, residual, undefined, workspace, counters, .{
                a_stride,   b_stride,   scale_a_rows,    scale_a_blocks,      scale_b_rows, scale_b_blocks,
                e8m0,       out_stride, residual_stride, m,                   n,            k,
                activation, panel,      along_m,         data_parallel_tiles,
            });
        }

        /// The assumptions of `storeKernel` hold, `rank % 16 == 0`, and the rows of `low` and
        /// `up` start at multiples of 4 bytes. The kernel adds `low upᵀ` to the FP32 sums before
        /// the bias, the residual and the one rounding to BF16.
        fn lowRankKernel(
            a: [*]addrspace(.global) const u8,
            b: [*]addrspace(.global) const u8,
            scale_a: u64,
            scale_b: u64,
            out: [*]addrspace(.global) Chunk,
            bias: u64,
            residual: u64,
            low: u64,
            up: u64,
            workspace: u64,
            counters: u64,
            a_stride: u32,
            b_stride: u32,
            scale_a_rows: u32,
            scale_a_blocks: u32,
            scale_b_rows: u32,
            scale_b_blocks: u32,
            e8m0: u32,
            out_stride: u32,
            residual_stride: u32,
            m: u32,
            n: u32,
            k: u32,
            activation: u32,
            rank: u32,
            panel: u32,
            along_m: u32,
            data_parallel_tiles: u32,
        ) callconv(kernel_convention) void {
            const low_rank: LowRank = .{ .low = low, .up = up, .rank = rank };
            runStore(true, a, b, scale_a, scale_b, out, bias, residual, low_rank, workspace, counters, .{
                a_stride,   b_stride,   scale_a_rows,    scale_a_blocks,      scale_b_rows, scale_b_blocks,
                e8m0,       out_stride, residual_stride, m,                   n,            k,
                activation, panel,      along_m,         data_parallel_tiles,
            });
        }

        inline fn runStore(
            comptime with_low_rank: bool,
            a: [*]addrspace(.global) const u8,
            b: [*]addrspace(.global) const u8,
            scale_a: u64,
            scale_b: u64,
            out: [*]addrspace(.global) Chunk,
            bias: u64,
            residual: u64,
            low_rank: LowRank,
            workspace: u64,
            counters: u64,
            sizes: [16]u32,
        ) void {
            const a_stride, const b_stride, const scale_a_rows, const scale_a_blocks = sizes[0..4].*;
            const scale_b_rows, const scale_b_blocks, const e8m0, const out_stride = sizes[4..8].*;
            const residual_stride, const m, const n, const k = sizes[8..12].*;
            const activation, const panel, const along_m, const data_parallel_tiles = sizes[12..16].*;
            const b_e8m0 = e8m0 & 2 != 0;
            const problem = problemOf(a, .{ b, b }, a_stride, b_stride, .{
                .a = scale_a,
                .b = .{ scale_b, scale_b },
                .a_rows = scale_a_rows,
                .a_blocks = scale_a_blocks,
                .b_rows = .{ scale_b_rows, scale_b_rows },
                .b_blocks = .{ scale_b_blocks, scale_b_blocks },
                .a_e8m0 = e8m0 & 1 != 0,
                .a_atoms = e8m0 & 16 != 0,
                .b_e8m0 = .{ b_e8m0, b_e8m0 },
                .a_vectors = undefined,
            }, m, n, k, panel, along_m);
            run(problem, .init(problem, data_parallel_tiles, workspace, counters), .{
                .out = out,
                .out_stride = out_stride,
                .bias = .{bias},
                .residual = residual,
                .residual_stride = residual_stride,
                .residual_f32 = e8m0 & 8 != 0,
                .activation = activation,
                .low_rank = low_rank,
            }, with_low_rank);
        }

        /// The assumptions of `storeKernel` hold with `f` for `n` and `gate` for B. Bit 2 of `e8m0`
        /// marks E8M0 `up` scales. `gate` and `up` hold `f` rows each at the stride `b_stride`.
        fn gatedKernel(
            a: [*]addrspace(.global) const u8,
            gate: [*]addrspace(.global) const u8,
            up: [*]addrspace(.global) const u8,
            scale_a: u64,
            scale_gate: u64,
            scale_up: u64,
            out: [*]addrspace(.global) Chunk,
            bias_gate: u64,
            bias_up: u64,
            residual: u64,
            workspace: u64,
            counters: u64,
            a_stride: u32,
            b_stride: u32,
            scale_a_rows: u32,
            scale_a_blocks: u32,
            scale_gate_rows: u32,
            scale_gate_blocks: u32,
            scale_up_rows: u32,
            scale_up_blocks: u32,
            e8m0: u32,
            out_stride: u32,
            residual_stride: u32,
            m: u32,
            f: u32,
            k: u32,
            activation: u32,
            panel: u32,
            along_m: u32,
            data_parallel_tiles: u32,
        ) callconv(kernel_convention) void {
            const problem = problemOf(a, .{ up, gate }, a_stride, b_stride, .{
                .a = scale_a,
                .b = .{ scale_up, scale_gate },
                .a_rows = scale_a_rows,
                .a_blocks = scale_a_blocks,
                .b_rows = .{ scale_up_rows, scale_gate_rows },
                .b_blocks = .{ scale_up_blocks, scale_gate_blocks },
                .a_e8m0 = e8m0 & 1 != 0,
                .a_atoms = e8m0 & 16 != 0,
                .b_e8m0 = .{ e8m0 & 4 != 0, e8m0 & 2 != 0 },
                .a_vectors = undefined,
            }, m, f, k, panel, along_m);
            run(problem, .init(problem, data_parallel_tiles, workspace, counters), .{
                .out = out,
                .out_stride = out_stride,
                .bias = .{ bias_up, bias_gate },
                .residual = residual,
                .residual_stride = residual_stride,
                .activation = activation,
            }, false);
        }

        /// The assumptions of `gatedKernel` hold, and `f % 128 == 0`. Bit 5 of `e8m0` marks E8M0
        /// scales of `h`. `activation` is a `GluActivation`, and `limit` bounds `gate` and `up`.
        fn gluKernel(
            a: [*]addrspace(.global) const u8,
            gate: [*]addrspace(.global) const u8,
            up: [*]addrspace(.global) const u8,
            scale_a: u64,
            scale_gate: u64,
            scale_up: u64,
            z_gate: u64,
            z_up: u64,
            codes: u64,
            scales: u64,
            workspace: u64,
            counters: u64,
            a_stride: u32,
            b_stride: u32,
            scale_a_rows: u32,
            scale_a_blocks: u32,
            scale_gate_rows: u32,
            scale_gate_blocks: u32,
            scale_up_rows: u32,
            scale_up_blocks: u32,
            e8m0: u32,
            z_stride: u32,
            code_stride: u32,
            scale_spans: u32,
            m: u32,
            f: u32,
            hidden: u32,
            k: u32,
            activation: u32,
            limit: f32,
            panel: u32,
            along_m: u32,
            data_parallel_tiles: u32,
        ) callconv(kernel_convention) void {
            const problem = problemOf(a, .{ up, gate }, a_stride, b_stride, .{
                .a = scale_a,
                .b = .{ scale_up, scale_gate },
                .a_rows = scale_a_rows,
                .a_blocks = scale_a_blocks,
                .b_rows = .{ scale_up_rows, scale_gate_rows },
                .b_blocks = .{ scale_up_blocks, scale_gate_blocks },
                .a_e8m0 = e8m0 & 1 != 0,
                .a_atoms = e8m0 & 16 != 0,
                .b_e8m0 = .{ e8m0 & 4 != 0, e8m0 & 2 != 0 },
                .a_vectors = undefined,
            }, m, f, k, panel, along_m);
            run(problem, .init(problem, data_parallel_tiles, workspace, counters), .{
                .out = undefined,
                .out_stride = 0,
                .bias = .{ 0, 0 },
                .residual = 0,
                .residual_stride = 0,
                .activation = 0,
                .glu = .{
                    .z_up = z_up,
                    .z_gate = z_gate,
                    .z_stride = z_stride,
                    .codes = codes,
                    .code_stride = code_stride,
                    .scales = scales,
                    .scale_spans = scale_spans,
                    .hidden = hidden,
                    .scale_format = if (e8m0 & 32 != 0) .e8m0 else .f32,
                    .activation = @fromBackingInt(activation),
                    .limit = limit,
                },
            }, false);
        }

        /// FP32 activation scales of four rows arrive with one 16-byte copy when the rows lie next
        /// to each other at a multiple of 16 bytes (PTX ISA 9.7.10.28.3.1, 6.4.1).
        inline fn problemOf(
            a: [*]addrspace(.global) const u8,
            b: [2][*]addrspace(.global) const u8,
            a_stride: u32,
            b_stride: u32,
            scales: Scales,
            m: u32,
            columns: u32,
            k: u32,
            panel: u32,
            along_m: u32,
        ) Problem {
            var with_vectors = scales;
            with_vectors.a_vectors = !scales.a_e8m0 and scales.a_rows == 1 and
                scales.a_blocks % 4 == 0 and scales.a % 16 == 0;
            return .{
                .a = a,
                .b = b,
                .a_stride = a_stride,
                .b_stride = b_stride,
                .scales = with_vectors,
                .m = m,
                .columns = columns,
                .k = k,
                .raster = .{ .panel = panel, .along_m = along_m != 0 },
                .tiles_m = (m + tile_m - 1) / tile_m,
                .tiles_n = (columns + block_columns - 1) / block_columns,
            };
        }

        /// Step `s` multiplies k tile `s` of the work group from stage `s % stages`. Once every
        /// warp has arrived on `empty` after step `s + prefetch - stages`, step `s` copies the
        /// operands of step `s + prefetch`. A `bar.warp.sync` before lane 0 arrives orders the
        /// reads of the warp before the release of the arrival (PTX ISA 9.7.15.2, 9.7.15.16.16).
        inline fn run(
            problem: Problem,
            schedule: Schedule,
            output: Output,
            comptime with_low_rank: bool,
        ) void {
            @setEvalBranchQuota(20_000);
            const steps = schedule.steps();
            if (steps == 0) return;
            if (@workItemId(0) == 0) {
                inline for (&full, &empty) |*full_stage, *empty_stage| {
                    full_stage.init(threads);
                    empty_stage.init(warps);
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            var copier: Copier = .init(schedule);
            inline for (0..prefetch) |stage| {
                if (stage < steps) {
                    copier.copy(problem, schedule, stage);
                    cp_async.mbarrierArriveNoinc(&full[stage]);
                }
            }
            const warp: Warp = .init();
            var cursor: Cursor = .init(schedule);
            var tile = problem.tileAt(cursor.tile);
            var active = warp.activeFragments(problem, tile);
            var sums: Accumulators = @splat(@splat(@splat(0)));
            var stage: u32 = 0;
            var refill_stage: u32 = prefetch;
            var step: u32 = 0;
            while (step < steps) : (step += 1) {
                const current = stage;
                while (!full[stage].testWaitParity(@truncate(step / stages))) {}
                if (atom_scales) warp.decodeAtoms(stage);
                const factors: BlockFactors = if (block_scales)
                    warp.tileFactors(problem, tile, cursor, stage)
                else {};
                const refill = step + prefetch < steps;
                const round: u1 = @truncate((step + prefetch) / stages);
                var early = false;
                inline for (0..chains) |chain| {
                    const b = loadB(stage, chain, warp.lanes);
                    const columns: [b_fragments][2]f32 = if (atom_scales)
                        warp.atomColumnScales(chain)
                    else
                        undefined;
                    warp.multiplyRows(&sums, b, columns, factors, stage, chain, 0, active);
                    if (chain == 0) {
                        early = refill and empty[refill_stage].testWaitParity(round ^ 1);
                        if (early) copier.refill(problem, schedule, refill_stage);
                    }
                    warp.multiplyRows(&sums, b, columns, factors, stage, chain, 1, active);
                }
                nvptx.barWarpSync(.all);
                if (warp.lane == 0) _ = empty[stage].arrive();
                if (refill and !early) {
                    while (!empty[refill_stage].testWaitParity(round ^ 1)) {}
                    copier.refill(problem, schedule, refill_stage);
                }
                stage = if (stage == stages - 1) 0 else stage + 1;
                refill_stage = if (refill_stage == stages - 1) 0 else refill_stage + 1;
                if (cursor.k_tile + 1 == cursor.k_end) {
                    warp.addWarpSums(&sums, current);
                    const whole = cursor.whole(schedule) or
                        warp.addParts(problem, schedule, cursor, tile, &sums);
                    if (whole) {
                        if (with_low_rank) warp.addLowRank(problem, output.low_rank, tile, &sums);
                        warp.finish(problem, output, tile, &sums);
                    }
                    sums = @splat(@splat(@splat(0)));
                }
                cursor.next(schedule);
                if (cursor.k_tile == cursor.k_begin and step + 1 < steps) {
                    tile = problem.tileAt(cursor.tile);
                    active = warp.activeFragments(problem, tile);
                }
            }
        }

        /// A copy past the last row of A or B or past the end of k reads nothing and zero-fills its
        /// chunk (PTX ISA 9.7.10.28.3.1). With `k % 16 == 0`, no chunk crosses the end of k.
        const Copier = struct {
            cursor: Cursor,
            tile: Tile,
            a_sources: [a_passes][*]addrspace(.global) const u8,
            b_sources: [b_passes][*]addrspace(.global) const u8,
            valid: u32,
            chunk: u32,

            inline fn init(schedule: Schedule) Copier {
                var copier: Copier = undefined;
                copier.cursor = .init(schedule);
                const tid = @workItemId(0);
                copier.chunk = swizzle(tid / row_chunks, tid % row_chunks);
                return copier;
            }

            inline fn start(copier: *Copier, problem: Problem) void {
                const tid = @workItemId(0);
                const column = tid % row_chunks * 16;
                copier.tile = problem.tileAt(copier.cursor.tile);
                copier.valid = 0;
                inline for (&copier.a_sources, 0..) |*source, pass| {
                    const row = copier.tile.m_offset + tid / row_chunks +
                        @as(u32, pass) * copy_rows;
                    copier.valid |= @as(u32, @intFromBool(row < problem.m)) << pass;
                    source.* = problem.a + @as(u64, @min(row, problem.m - 1)) * problem.a_stride +
                        column;
                }
                inline for (&copier.b_sources, 0..) |*source, pass| {
                    const half = pass / (b_passes / 2);
                    const row = copier.tile.b_rows[half] + tid / row_chunks +
                        @as(u32, pass % (b_passes / 2)) * copy_rows;
                    const valid: u32 = @intFromBool(row < problem.columns);
                    copier.valid |= valid << (a_passes + pass);
                    source.* = problem.b[half] +
                        @as(u64, @min(row, problem.columns - 1)) * problem.b_stride + column;
                }
            }

            inline fn refill(
                copier: *Copier,
                problem: Problem,
                schedule: Schedule,
                stage: u32,
            ) void {
                copier.copy(problem, schedule, stage);
                cp_async.mbarrierArriveNoinc(&full[stage]);
            }

            inline fn copy(copier: *Copier, problem: Problem, schedule: Schedule, stage: u32) void {
                if (copier.cursor.k_tile == copier.cursor.k_begin) copier.start(problem);
                const tid = @workItemId(0);
                const column = tid % row_chunks;
                const inside = (block_scales and tile_k <= scale_group) or
                    copier.cursor.k_tile * tile_k + column * 16 < problem.k;
                const offset = if (inside) copier.cursor.k_tile * tile_k else 0;
                const a_chunks = layout.stages[stage].a[copier.chunk..];
                inline for (copier.a_sources, 0..) |source, pass| {
                    const rows = pass * copy_rows;
                    copier.copyChunk(&a_chunks[rows * row_chunks], source + offset, inside, pass);
                }
                const b_chunks = layout.stages[stage].b[copier.chunk..];
                inline for (copier.b_sources, 0..) |source, pass| {
                    const rows = pass / (b_passes / 2) * half_rows + pass % (b_passes / 2) * copy_rows;
                    copier.copyChunk(
                        &b_chunks[rows * row_chunks],
                        source + offset,
                        inside,
                        a_passes + pass,
                    );
                }
                if (block_scales) copier.copyScales(problem, stage);
                if (atom_scales) copier.copyAtoms(problem, stage);
                copier.cursor.next(schedule);
            }

            inline fn copyChunk(
                copier: *const Copier,
                target: *addrspace(.shared) Chunk,
                source: [*]addrspace(.global) const u8,
                inside: bool,
                comptime pass: u32,
            ) void {
                cp_async.copyZeroFill(
                    .cache_global_level,
                    target,
                    @as(*addrspace(.global) const Chunk, @ptrCast(@alignCast(source))),
                    if (inside and copier.valid >> pass & 1 != 0) 16 else 0,
                );
            }

            /// Copies the scales of the stage's scale groups, or zeros past the end of k or of A.
            inline fn copyScales(copier: *const Copier, problem: Problem, stage: u32) void {
                const tid = @workItemId(0);
                const first_block = copier.cursor.firstBlock();
                const blocks = problem.k / scale_group;
                const scales = problem.scales;
                if (tid < blocks_per_stage * a_scale_copies) {
                    const block = first_block + tid / a_scale_copies;
                    const first_row = tid % a_scale_copies * 4;
                    const row = copier.tile.m_offset + first_row;
                    const count: u32 = if (row < problem.m and block < blocks)
                        @min(problem.m - row, 4)
                    else
                        0;
                    const first = if (scales.a_atoms)
                        atomIndex(row, block, scales.a_blocks)
                    else
                        @as(u64, block) * scales.a_blocks + @as(u64, row) * scales.a_rows;
                    const target = &layout.a_scales[stage][tid / a_scale_copies];
                    if (scales.a_vectors and count == 4) {
                        const source: *addrspace(.global) const Chunk =
                            @ptrFromInt(scales.a + first * 4);
                        const chunk: *addrspace(.shared) Chunk =
                            @ptrCast(@alignCast(&target[first_row]));
                        cp_async.copy(.cache_all_levels, chunk, source);
                    } else {
                        inline for (0..4) |e| {
                            copyScale(
                                &target[first_row + e],
                                scales.a,
                                first + e * scales.a_rows,
                                scales.a_e8m0,
                                e < count,
                            );
                        }
                    }
                } else if (tid < (a_scale_copies + 2) * blocks_per_stage) {
                    const index = tid - blocks_per_stage * a_scale_copies;
                    const half = index % 2;
                    const block = first_block + index / 2;
                    copyScale(
                        &layout.b_scales[stage][index / 2][half],
                        pick(u64, scales.b, half),
                        problem.weightIndex(half, pick(u32, copier.tile.b_rows, half), block),
                        pick(bool, scales.b_e8m0, half),
                        block < blocks,
                    );
                }
            }

            /// Copies the scale words of the stage's rows of A and B. The word of row `r` and 128
            /// values `w` lies at byte `512 * (r / 128 * k / 128 + w) + r % 32 * 16 + r % 128 / 32 * 4`.
            inline fn copyAtoms(copier: *const Copier, problem: Problem, stage: u32) void {
                const words = tile_k / mxfp8_atom_k;
                const rows = tile_m + tile_n;
                const row_atoms = problem.k / mxfp8_atom_k;
                const first_word = copier.cursor.k_tile * words;
                inline for (0..(words * rows + threads - 1) / threads) |pass| {
                    const index = @as(u32, pass) * threads + @workItemId(0);
                    if (index < words * rows) {
                        const word = index / rows;
                        const row = index % rows;
                        const in_a = row < tile_m;
                        const b_row = row -% tile_m;
                        const half = if (kind == .glu) b_row / half_rows else 0;
                        const global_row = if (in_a)
                            copier.tile.m_offset + row
                        else if (kind == .glu)
                            copier.tile.column_offset + b_row % half_rows
                        else
                            copier.tile.column_offset + b_row;
                        const k_word = first_word + word;
                        const limit = if (in_a) problem.m else problem.columns;
                        const atom = @as(u64, global_row / 128) * row_atoms + k_word;
                        const offset = atom * 512 + global_row % 32 * 16 + global_row % 128 / 32 * 4;
                        copyScale(
                            &layout.atoms[stage][word][row],
                            if (in_a) problem.scales.a else pick(u64, problem.scales.b, half),
                            offset / 4,
                            false,
                            k_word < row_atoms and global_row < limit,
                        );
                    }
                }
            }
        };

        /// Returns the index of the scale of `row` and scale group `group` in scale atoms with
        /// `spans` atoms per 128 rows, laid out as the `fp8_b128` activation scales of `packs.py`.
        inline fn atomIndex(row: u32, group: u32, spans: u32) u64 {
            const atom = @as(u64, row / 128) * spans + group / 4;
            return atom * 512 + row % 32 * 16 + row % 128 / 32 * 4 + group % 4;
        }

        /// A one-byte scale arrives in byte `address % 4` of the aligned word that holds it.
        inline fn copyScale(
            target: *addrspace(.shared) f32,
            base: u64,
            index: u64,
            one_byte: bool,
            exists: bool,
        ) void {
            const address = if (!exists) base else if (one_byte) base + index else base + index * 4;
            const source: *addrspace(.global) const f32 = @ptrFromInt(address & ~@as(u64, 3));
            const byte_size = @as(u32, @truncate(address % 4)) + 1;
            const size: u32 = if (!exists) 0 else if (one_byte) byte_size else @sizeOf(f32);
            cp_async.copyZeroFill(.cache_all_levels, target, source, size);
        }

        inline fn decodeByte(word: f32, address: u32) f32 {
            const bits: u32 = @bitCast(word);
            return fp8.e8m0ToF32(@truncate(bits >> @intCast(address % 4 * 8)));
        }

        inline fn swizzle(row: u32, column: u32) u32 {
            return row * row_chunks + (column ^ row % 8 * row_chunks / 8);
        }

        inline fn bRow(warp_column: u32, comptime j: u32) u32 {
            return switch (kind) {
                .store => warp_column * warp_n + j * 16,
                .gated, .glu => half_rows * (j / (warp_n / 32)) + warp_column * (warp_n / 2) +
                    j % (warp_n / 32) * 16,
            };
        }

        /// `ldmatrix` reads the 32 values of k step `s` at the chunks `a ^ 2 s` and `b ^ 2 s`. Each
        /// E4M3 fragment of `mma.m16n8k32` holds the bytes of a BF16 fragment of `mma.m16n8k16`, so
        /// the `ldmatrix` loads of BF16 fragments also load E4M3 fragments (PTX ISA 9.7.16.5.10).
        const Lanes = struct {
            a: u32,
            b: u32,
        };

        inline fn laneOffsets(warp_row: u32, warp_column: u32, lane: u32) Lanes {
            return .{
                .a = swizzle(warp_row + lane % 16, lane / 16),
                .b = swizzle(bRow(warp_column, 0) + lane % 8 + lane / 16 * 8, lane / 8 % 2),
            };
        }

        inline fn loadA(
            stage: u32,
            comptime k_step: u32,
            comptime i: u32,
            lanes: Lanes,
        ) @Vector(4, u32) {
            return nvptx.ldmatrix(.x4, false, &layout.stages[stage].a[
                (lanes.a ^ k_step * 2) + i * 16 * row_chunks
            ]);
        }

        const ChainB = [chain_steps][b_fragments]@Vector(2, u32);

        inline fn loadB(stage: u32, comptime chain: u32, lanes: Lanes) ChainB {
            var fragments: ChainB = undefined;
            inline for (&fragments, chain * chain_steps..) |*step_fragments, k_step| {
                inline for (0..b_fragments / 2) |j| {
                    const pair = nvptx.ldmatrix(.x4, false, &layout.stages[stage].b[
                        (lanes.b ^ k_step * 2) + (bRow(0, j) - bRow(0, 0)) * row_chunks
                    ]);
                    step_fragments[j * 2] = .{ pair[0], pair[1] };
                    step_fragments[j * 2 + 1] = .{ pair[2], pair[3] };
                }
            }
            return fragments;
        }

        /// Work item `lane` holds the sums of rows `lane / 4 + 8 h` and columns `lane % 4 * 2 + e`
        /// in element `2 h + e` of each fragment (PTX ISA 9.7.16.5.10).
        inline fn fragmentRow(comptime i: u32, comptime h: u32, lane: u32) u32 {
            return i * 16 + lane / 4 + h * 8;
        }

        inline fn owns(k_index: u32, comptime i: u32) bool {
            return k_warps == 1 or i % k_warps == k_index;
        }

        inline fn sumIndex(comptime i: u32, comptime j: u32, index: u32) u32 {
            return (i * b_fragments + j) * spatial_threads + index;
        }

        inline fn partIndex(slot: u32, comptime i: u32, comptime j: u32, index: u32) u64 {
            return @as(u64, slot) * (slot_floats / 4) + sumIndex(i, j, index);
        }

        /// In `addParts`, work item 0 of each work group publishes and acquires the parts with an
        /// `atom.acq_rel.gpu` between two `bar.sync`. The parts that other work groups stored
        /// therefore precede the `ld.global.cg` of `loadPart` (PTX ISA 8.8, 8.9.4).
        inline fn loadPart(
            part: *allowzero addrspace(.global) const @Vector(4, f32),
        ) @Vector(4, f32) {
            var x: f32 = undefined;
            var y: f32 = undefined;
            var z: f32 = undefined;
            var w: f32 = undefined;
            asm volatile ("ld.global.cg.v4.f32 {%[x], %[y], %[z], %[w]}, [%[p]];"
                : [x] "=f" (x),
                  [y] "=f" (y),
                  [z] "=f" (z),
                  [w] "=f" (w),
                : [p] "l" (part),
                : .{ .memory = true });
            return .{ x, y, z, w };
        }

        inline fn globalScale(base: u64, index: u32, stride: u32) f32 {
            const scale: *addrspace(.global) const f32 =
                @ptrFromInt(base + @as(u64, index) * stride * @sizeOf(f32));
            return scale.*;
        }

        /// Warp `w` multiplies the `k_steps` k steps of each k tile from k step `k_steps * k_index`
        /// on, where `k_index = w / spatial_warps`. With `k_warps > 1`, warp `w` computes warp tile
        /// `(w + w / 4 * warps_n) % spatial_warps` in place of warp tile `w`. Warps `w` and `w + 4`
        /// therefore multiply different rows of A when the tile has two rows of warp tiles.
        const Warp = struct {
            lane: u32,
            /// The index of the work item among the work items of the warp tiles with `k_index` 0.
            index: u32,
            m: u32,
            n: u32,
            k_index: u32,
            lanes: Lanes,
            box: u32,

            inline fn init() Warp {
                const tid = @workItemId(0);
                const lane = tid % 32;
                const k_index = tid / 32 / spatial_warps;
                const warp_index = tid / 32;
                const spatial = if (k_warps == 1)
                    warp_index
                else
                    (warp_index + warp_index / 4 * warps_n) % spatial_warps;
                const warp_row = spatial / warps_n * warp_m;
                const warp_column = spatial % warps_n;
                const lanes = laneOffsets(warp_row, warp_column, lane);
                return .{
                    .lane = lane,
                    .index = spatial * 32 + lane,
                    .m = warp_row,
                    .n = warp_column,
                    .k_index = k_index,
                    .lanes = .{
                        .a = lanes.a ^ k_index * k_steps * 2,
                        .b = lanes.b ^ k_index * k_steps * 2,
                    },
                    .box = if (halves == 2) 0 else warp_column * warp_n / half_rows,
                };
            }

            /// Adds the warp sums of the warp tile in k order through stage `stage`, `pass_rows` A
            /// fragment rows at a time. Every warp has read the stage, and no copy refills the
            /// stage before the next step. Afterwards the warp that owns a row holds its sums.
            inline fn addWarpSums(warp: Warp, sums: *Accumulators, stage: u32) void {
                if (k_warps == 1) return;
                const buffer: *addrspace(.shared) WarpSums = @ptrCast(&layout.stages[stage]);
                inline for (0..a_fragments / pass_rows) |pass| {
                    nvptx.barrierSyncAligned(.work_group, null);
                    inline for (sums[pass * pass_rows ..][0..pass_rows], 0..) |row, r| {
                        const owner = (pass * pass_rows + r) % k_warps;
                        if (warp.k_index != owner) {
                            const slot = if (warp.k_index < owner) warp.k_index else warp.k_index - 1;
                            inline for (row, 0..) |fragment, j| {
                                buffer[slot][sumIndex(r, j, warp.index)] = fragment;
                            }
                        }
                    }
                    nvptx.barrierSyncAligned(.work_group, null);
                    inline for (sums[pass * pass_rows ..][0..pass_rows], 0..) |*row, r| {
                        const owner = (pass * pass_rows + r) % k_warps;
                        if (warp.k_index == owner) {
                            inline for (row, 0..) |*fragment, j| {
                                var total = if (owner == 0)
                                    fragment.*
                                else
                                    buffer[0][sumIndex(r, j, warp.index)];
                                inline for (1..k_warps) |other| {
                                    total += if (other == owner)
                                        fragment.*
                                    else
                                        buffer[if (other < owner) other else other - 1][
                                            sumIndex(r, j, warp.index)
                                        ];
                                }
                                fragment.* = total;
                            }
                        }
                    }
                }
                nvptx.barrierSyncAligned(.work_group, null);
            }

            inline fn activeFragments(warp: Warp, problem: Problem, tile: Tile) u32 {
                const rows = (problem.m -| tile.m_offset) -| warp.m;
                return @min(a_fragments, (rows + 15) / 16);
            }

            /// Returns the index of the scale group of the warp's chains within the k tile.
            inline fn localBlock(warp: Warp) u32 {
                return warp.k_index * k_steps * 32 / scale_group;
            }

            /// Returns the scale products of the rows of each A fragment row with each half of B.
            /// Rows of a work item lie 8 apart, so their one-byte scales share one position in a
            /// word. The chains of a warp in one k tile lie in one K block and share the products.
            inline fn tileFactors(
                warp: Warp,
                problem: Problem,
                tile: Tile,
                cursor: Cursor,
                stage: u32,
            ) BlockFactors {
                const scales = problem.scales;
                const block = cursor.firstBlock() + warp.localBlock();
                var weights: [halves]f32 = undefined;
                inline for (&weights, 0..) |*weight, half| {
                    const slot = warp.box + @as(u32, half);
                    const word = layout.b_scales[stage][warp.localBlock()][slot];
                    weight.* = if (pick(bool, scales.b_e8m0, slot)) one_byte: {
                        const row = @min(pick(u32, tile.b_rows, slot), problem.columns - 1);
                        const tile_row = row / scale_group;
                        const address = @as(u32, @truncate(pick(u64, scales.b, slot))) +%
                            tile_row *% pick(u32, scales.b_rows, slot) +%
                            block *% pick(u32, scales.b_blocks, slot);
                        break :one_byte decodeByte(word, address);
                    } else word;
                }
                const first_row = tile.m_offset + warp.m + warp.lane / 4;
                const a_address = if (scales.a_atoms)
                    @as(u32, @truncate(scales.a)) +% block
                else
                    @as(u32, @truncate(scales.a)) +% block *% scales.a_blocks +%
                        first_row *% scales.a_rows;
                var factors: BlockFactors = undefined;
                inline for (&factors, 0..) |*row_factors, i| {
                    inline for (0..2) |h| {
                        const row = warp.m + fragmentRow(i, h, warp.lane);
                        const word = layout.a_scales[stage][warp.localBlock()][row];
                        const scale_a = if (scales.a_e8m0) decodeByte(word, a_address) else word;
                        inline for (row_factors, weights) |*pair, weight| pair[h] = scale_a * weight;
                    }
                }
                return factors;
            }

            inline fn atomBlock(warp: Warp, comptime chain: u32) u32 {
                return if (k_warps == 1) chain else warp.k_index * k_steps + chain;
            }

            inline fn warpScales() *addrspace(.shared) [k_steps][warp_m + warp_n]f32 {
                return &layout.warp_scales[@workItemId(0) / 32];
            }

            /// Decodes the E8M0 scales of the warp's rows and columns in the stage. The
            /// `bar.warp.sync` orders the stores before the loads of the warp (PTX ISA 9.7.15.2).
            inline fn decodeAtoms(warp: Warp, stage: u32) void {
                const scales = warpScales();
                inline for (0..(warp_m + warp_n) / 32) |pass| {
                    const row = pass * 32 + warp.lane;
                    const tile_row = if (pass * 32 < warp_m) warp.m + row else tile_row: {
                        const j = (pass * 32 - warp_m) / 16;
                        const rows: [2]u32 = .{ bRow(warp.n, j), bRow(warp.n, j + 1) };
                        break :tile_row tile_m + pick(u32, rows, warp.lane / 16) + warp.lane % 16;
                    };
                    inline for (0..k_steps) |chain| {
                        const block = warp.atomBlock(chain);
                        scales[chain][row] = decodeByte(layout.atoms[stage][block / 4][tile_row], block);
                    }
                }
                nvptx.barWarpSync(.all);
            }

            /// Returns the scales of the two columns of each B fragment of this lane in the chain
            /// `chain`.
            inline fn atomColumnScales(warp: Warp, comptime chain: u32) [b_fragments][2]f32 {
                const chain_scales = &warpScales()[chain];
                var scales: [b_fragments][2]f32 = undefined;
                inline for (&scales, 0..) |*pair, j| {
                    const column = warp_m + j * 8 + warp.lane % 4 * 2;
                    const two: *addrspace(.shared) const @Vector(2, f32) =
                        @ptrCast(@alignCast(&chain_scales[column]));
                    pair.* = two.*;
                }
                return scales;
            }

            /// Returns the products of the E8M0 scales of the rows of A fragment row `i` and of
            /// the columns `columns` in the chain `chain`.
            inline fn atomFactors(
                warp: Warp,
                comptime chain: u32,
                comptime i: u32,
                columns: [b_fragments][2]f32,
            ) Partial {
                const chain_scales = &warpScales()[chain];
                var scale_a: [2]f32 = undefined;
                inline for (&scale_a, 0..) |*scale, h| scale.* = chain_scales[fragmentRow(i, h, warp.lane)];
                var factors: Partial = undefined;
                inline for (&factors, columns) |*factor, pair| {
                    factor.* = .{
                        scale_a[0] * pair[0],
                        scale_a[0] * pair[1],
                        scale_a[1] * pair[0],
                        scale_a[1] * pair[1],
                    };
                }
                return factors;
            }

            inline fn multiplyRows(
                warp: Warp,
                sums: *Accumulators,
                b: ChainB,
                columns: [b_fragments][2]f32,
                block_factors: BlockFactors,
                stage: u32,
                comptime chain: u32,
                comptime part: u32,
                active: u32,
            ) void {
                const rows = a_fragments / 2;
                inline for (sums[part * rows ..][0..rows], part * rows..) |*row, i| {
                    if (i < active) {
                        if (fast) {
                            inline for (b, chain * chain_steps..) |b_step, k_step| {
                                const a = loadA(stage, k_step, i, warp.lanes);
                                inline for (row, b_step) |*sum, fragment| {
                                    sum.* = nvptx.mmaSync(mma_options, a, fragment, sum.*);
                                }
                            }
                        } else {
                            var partial: Partial = @splat(@splat(0));
                            inline for (b, chain * chain_steps..) |b_step, k_step| {
                                const a = loadA(stage, k_step, i, warp.lanes);
                                inline for (&partial, b_step) |*sum, fragment| {
                                    sum.* = nvptx.mmaSync(mma_options, a, fragment, sum.*);
                                }
                            }
                            if (atom_scales) {
                                const factors = warp.atomFactors(chain, i, columns);
                                inline for (row, partial, factors) |*sum, p, factor| {
                                    sum.* = @mulAdd(@Vector(4, f32), p, factor, sum.*);
                                }
                            } else if (!block_scales) {
                                inline for (row, partial) |*sum, p| sum.* += p;
                            } else {
                                inline for (row, partial, 0..) |*sum, p, j| {
                                    const pair = block_factors[i][j / half_fragments];
                                    const factor: @Vector(4, f32) =
                                        .{ pair[0], pair[0], pair[1], pair[1] };
                                    sum.* = @mulAdd(@Vector(4, f32), p, factor, sum.*);
                                }
                            }
                        }
                    }
                }
            }

            /// Writes the sums of a part of the tile to a slot. The last work group to finish a
            /// part takes the sum of all parts in k order and returns true.
            inline fn addParts(
                warp: Warp,
                problem: Problem,
                schedule: Schedule,
                cursor: Cursor,
                tile: Tile,
                sums: *Accumulators,
            ) bool {
                const tid = @workItemId(0);
                const index = warp.index;
                const first_stream_k_tile =
                    (cursor.tile - schedule.data_parallel_tiles) * schedule.k_tiles;
                const first = schedule.owner(first_stream_k_tile);
                const last = schedule.owner(first_stream_k_tile + schedule.k_tiles - 1);
                const rows = (problem.m - tile.m_offset) -| warp.m;
                const own = schedule.slot(@workGroupId(0), first_stream_k_tile);
                inline for (sums, 0..) |row, i| {
                    if (i * 16 < rows and owns(warp.k_index, i)) {
                        inline for (row, 0..) |fragment, j| {
                            schedule.workspace[partIndex(own, i, j, index)] = fragment;
                        }
                    }
                }
                nvptx.barrierSyncAligned(.work_group, null);
                if (tid == 0) {
                    const counter = &schedule.counters[cursor.tile - schedule.data_parallel_tiles];
                    const arrived = nvptx.atom(.gpu, .acq_rel, .add, counter, 1);
                    last_part = arrived == last - first;
                    if (last_part) counter.* = 0;
                }
                nvptx.barrierSyncAligned(.work_group, null);
                if (!last_part) return false;
                var work_group = first;
                while (work_group <= last) : (work_group += 1) {
                    const slot = schedule.slot(work_group, first_stream_k_tile);
                    inline for (sums, 0..) |*row, i| {
                        if (i * 16 < rows and owns(warp.k_index, i)) {
                            inline for (row, 0..) |*fragment, j| {
                                const source = &schedule.workspace[partIndex(slot, i, j, index)];
                                const part = loadPart(source);
                                fragment.* = if (work_group == first) part else fragment.* + part;
                            }
                        }
                    }
                }
                return true;
            }

            /// Multiplies each owned sum by `fp32(scale_a[row] * scale_b[column])`.
            /// Register `r` of a BF16 A fragment of `mma.m16n8k16` holds the k values
            /// `2 (lane % 4) + 8 (r / 2)` and the next one of row `lane / 4 + 8 (r % 2)`. Register
            /// `r` of a B fragment holds the same k values of column `lane / 4` (PTX ISA 9.7.16.5.8).
            inline fn addLowRank(
                warp: Warp,
                problem: Problem,
                low_rank: LowRank,
                tile: Tile,
                sums: *Accumulators,
            ) void {
                const low: [*]addrspace(.global) const u32 = @ptrFromInt(low_rank.low);
                const up: [*]addrspace(.global) const u32 = @ptrFromInt(low_rank.up);
                const words: u64 = low_rank.rank / 2;
                const first_row = tile.m_offset + warp.m;
                const first_column = tile.column_offset + warp.n * warp_n + warp.lane / 4;
                var step: u32 = 0;
                while (step < low_rank.rank / 16) : (step += 1) {
                    const word = step * 8 + warp.lane % 4;
                    var b: [b_fragments]@Vector(2, u32) = @splat(@splat(0));
                    inline for (&b, 0..) |*fragment, j| {
                        const column = first_column + j * 8;
                        if (column < problem.columns) {
                            const first = column * words + word;
                            fragment.* = .{ up[first], up[first + 4] };
                        }
                    }
                    inline for (sums, 0..) |*row_sums, i| {
                        if (owns(warp.k_index, i)) {
                            var a: @Vector(4, u32) = @splat(0);
                            inline for (0..4) |r| {
                                const row = first_row + fragmentRow(i, r % 2, warp.lane);
                                if (row < problem.m) a[r] = low[row * words + word + r / 2 * 4];
                            }
                            inline for (row_sums, b) |*sum, fragment| {
                                sum.* = nvptx.mmaSync(low_rank_mma, a, fragment, sum.*);
                            }
                        }
                    }
                }
            }

            inline fn applyScales(
                warp: Warp,
                problem: Problem,
                tile: Tile,
                sums: *Accumulators,
            ) void {
                const scales = problem.scales;
                var row_scales: [a_fragments][2]f32 = undefined;
                inline for (&row_scales, 0..) |*pair, i| {
                    inline for (pair, 0..) |*scale, h| {
                        const row = tile.m_offset + warp.m + fragmentRow(i, h, warp.lane);
                        scale.* = if (owns(warp.k_index, i))
                            globalScale(scales.a, @min(row, problem.m - 1), scales.a_rows)
                        else
                            0;
                    }
                }
                const quad = warp.lane % 4;
                const first_column = tile.column_offset + warp.n * (warp_n / halves) + quad * 2;
                inline for (0..b_fragments) |j| {
                    const half = j / half_fragments;
                    const column = first_column + j % half_fragments * 8;
                    var column_scales: [2]f32 = undefined;
                    inline for (&column_scales, 0..) |*scale, e| {
                        const row = @min(column + e, problem.columns - 1);
                        scale.* = globalScale(scales.b[half], row, scales.b_rows[half]);
                    }
                    inline for (sums, row_scales, 0..) |*row, pair, i| {
                        if (owns(warp.k_index, i)) {
                            row[j] *= .{
                                pair[0] * column_scales[0],
                                pair[0] * column_scales[1],
                                pair[1] * column_scales[0],
                                pair[1] * column_scales[1],
                            };
                        }
                    }
                }
            }

            inline fn finish(
                warp: Warp,
                problem: Problem,
                output: Output,
                tile: Tile,
                sums: *Accumulators,
            ) void {
                if (variant.scaling == .rowwise) warp.applyScales(problem, tile, sums);
                if (kind == .glu and variant.scaling == .rowwise) {
                    switch (output.glu.activation) {
                        inline else => |activation| storeRowGlu(activation, output.glu, problem, warp, tile, sums),
                    }
                    return;
                }
                if (kind == .glu and atom_scales) {
                    switch (output.glu.activation) {
                        inline else => |activation| storeAtomGlu(
                            activation,
                            output.glu,
                            problem,
                            warp,
                            tile,
                            sums,
                        ),
                    }
                    return;
                }
                if (kind == .glu) {
                    switch (output.glu.activation) {
                        inline else => |activation| switch (output.glu.scale_format) {
                            inline else => |format| storeGlu(
                                activation,
                                format,
                                output.glu,
                                problem,
                                warp,
                                tile,
                                sums,
                            ),
                        },
                    }
                    return;
                }
                const first_row = tile.m_offset + warp.m;
                const first_column = tile.column_offset + warp.n * (warp_n / halves);
                switch (@as(epilogue.Activation, @fromBackingInt(@intCast(output.activation)))) {
                    inline else => |activation| store(
                        activation,
                        output,
                        problem,
                        first_row,
                        first_column,
                        warp.lane,
                        warp.k_index,
                        sums,
                    ),
                }
            }
        };

        /// Work item `q` of a quad receives word `q` of each work item of the quad, in their order.
        inline fn transposeQuad(x: [4]u32, quad: u32) Chunk {
            const upper = quad >= 2;
            var y = x;
            inline for (0..2) |e| {
                const got: u32 = @bitCast(warp_functions.shuffleXor(
                    @bitCast(if (upper) x[e] else x[e + 2]),
                    2,
                ));
                if (upper) y[e] = got else y[e + 2] = got;
            }
            const odd = quad % 2 == 1;
            var z = y;
            inline for (0..2) |e| {
                const got: u32 = @bitCast(warp_functions.shuffleXor(
                    @bitCast(if (odd) y[2 * e] else y[2 * e + 1]),
                    1,
                ));
                if (odd) z[2 * e] = got else z[2 * e + 1] = got;
            }
            return z;
        }

        inline fn storeChunk(out: [*]addrspace(.global) Chunk, index: u64, chunk: Chunk) void {
            asm volatile ("st.global.cs.v4.u32 [%[p]], {%[x], %[y], %[z], %[w]};"
                :
                : [p] "l" (out + index),
                  [x] "r" (chunk[0]),
                  [y] "r" (chunk[1]),
                  [z] "r" (chunk[2]),
                  [w] "r" (chunk[3]),
                : .{ .memory = true });
        }

        inline fn storeCodes(address: u64, words: @Vector(4, u32)) void {
            asm volatile ("st.global.cs.v2.u32 [%[p]], {%[x], %[y]};"
                :
                : [p] "l" (address),
                  [x] "r" (words[0]),
                  [y] "r" (words[1]),
                : .{ .memory = true });
        }

        /// Pairs the BF16 rounding of the sums with the `mm_abt` store, which adds a zero bias.
        inline fn zPair(v: @Vector(4, f32), comptime h: u32) u32 {
            return bf16.pack(v[h * 2] + 0.0, v[h * 2 + 1] + 0.0);
        }

        /// The values of `h` are BF16 values, so the upper 16 bits of an amax hold all of its bits.
        var glu_amax: [if (kind == .glu and block_scales) tile_m else 0][warps_n]u16 addrspace(.shared) = undefined;

        /// After `transposeQuad`, work item `q` of a quad holds 8 columns of `up` and `gate`, the
        /// quad holds the 32 columns of the warp tile, and the `warps_n` warp tiles of a row hold
        /// one scale block of 128 columns. The scale atoms follow `packs.fp8_b128_activation`.
        inline fn storeGlu(
            comptime activation: GluActivation,
            comptime format: ScaleFormat,
            glu: GluOutput,
            problem: Problem,
            warp: Warp,
            tile: Tile,
            sums: *const Accumulators,
        ) void {
            const quad = warp.lane % 4;
            const column = tile.column_offset + warp.n * (warp_n / halves) + quad * 8;
            const scales: [*]addrspace(.global) fp8.Scale(format) = @ptrFromInt(glu.scales);
            const block = tile.column_offset / scale_group;
            inline for (sums, 0..) |row_fragments, i| {
                var row_products: [2]Chunk = undefined;
                inline for (&row_products, 0..) |*chunk, h| {
                    var up_pairs: [half_fragments]u32 = undefined;
                    var gate_pairs: [half_fragments]u32 = undefined;
                    inline for (&up_pairs, &gate_pairs, 0..) |*up_pair, *gate_pair, j| {
                        up_pair.* = zPair(row_fragments[j], h);
                        gate_pair.* = zPair(row_fragments[j + half_fragments], h);
                    }
                    const up = transposeQuad(up_pairs, quad);
                    const gate = transposeQuad(gate_pairs, quad);
                    const tile_row = warp.m + fragmentRow(i, h, warp.lane);
                    const row = tile.m_offset + tile_row;
                    if (row < problem.m) {
                        const index = (@as(u64, row) * glu.z_stride + column) / 8;
                        if (glu.z_up != 0) storeChunk(@ptrFromInt(glu.z_up), index, up);
                        if (glu.z_gate != 0) storeChunk(@ptrFromInt(glu.z_gate), index, gate);
                    }
                    var values: [8]f32 = undefined;
                    inline for (&values, bf16.unpackChunk(up), bf16.unpackChunk(gate)) |*value, u, g| {
                        value.* = switch (activation) {
                            .swiglu => swiglu(u, g, glu.limit),
                            .geglu_tanh => gegluTanh(u, g),
                        };
                    }
                    chunk.* = bf16.packChunk(values);
                    const amax = fp8.maxNanOverLanes(fp8.amaxOf(values), 4);
                    if (quad == 0) glu_amax[tile_row][warp.n] = @truncate(@as(u32, @bitCast(amax)) >> 16);
                }
                nvptx.barrierSyncAligned(.work_group, null);
                inline for (row_products, 0..) |chunk, h| {
                    const tile_row = warp.m + fragmentRow(i, h, warp.lane);
                    const row = tile.m_offset + tile_row;
                    var amax = bf16.toF32(glu_amax[tile_row][0]);
                    inline for (1..warps_n) |w| {
                        amax = fp8.maxNan(amax, bf16.toF32(glu_amax[tile_row][w]));
                    }
                    const scaling: fp8_pack.Scaling(format) = .init(amax);
                    var values: [16]f32 = @splat(0);
                    values[0..8].* = bf16.unpackChunk(chunk);
                    if (row < problem.m) {
                        storeCodes(
                            glu.codes + @as(u64, row) * glu.code_stride + column,
                            fp8_pack.scaledCodes(values, scaling.factor),
                        );
                    }
                    if (warp.n == 0 and quad == 0) {
                        const atom = row / 128 * glu.scale_spans + block / 4;
                        const offset = atom * 512 + row % 32 * 16 + row % 128 / 32 * 4 + block % 4;
                        scales[offset] = if (row < problem.m) scaling.scale else 0;
                    }
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
        }

        /// Stores the BF16 values of `h` to row `r` at `codes + 2 * r * code_stride`.
        inline fn storeRowGlu(
            comptime activation: GluActivation,
            glu: GluOutput,
            problem: Problem,
            warp: Warp,
            tile: Tile,
            sums: *const Accumulators,
        ) void {
            const quad = warp.lane % 4;
            const column = tile.column_offset + warp.n * (warp_n / halves) + quad * 8;
            inline for (sums, 0..) |row_fragments, i| {
                inline for (0..2) |h| {
                    var up_pairs: [half_fragments]u32 = undefined;
                    var gate_pairs: [half_fragments]u32 = undefined;
                    inline for (&up_pairs, &gate_pairs, 0..) |*up_pair, *gate_pair, j| {
                        up_pair.* = zPair(row_fragments[j], h);
                        gate_pair.* = zPair(row_fragments[j + half_fragments], h);
                    }
                    const up = transposeQuad(up_pairs, quad);
                    const gate = transposeQuad(gate_pairs, quad);
                    const row = tile.m_offset + warp.m + fragmentRow(i, h, warp.lane);
                    if (row < problem.m) {
                        const index = (@as(u64, row) * glu.z_stride + column) / 8;
                        if (glu.z_up != 0) storeChunk(@ptrFromInt(glu.z_up), index, up);
                        if (glu.z_gate != 0) storeChunk(@ptrFromInt(glu.z_gate), index, gate);
                        var values: [8]f32 = undefined;
                        inline for (&values, bf16.unpackChunk(up), bf16.unpackChunk(gate)) |*value, u, g| {
                            value.* = switch (activation) {
                                .swiglu => swiglu(u, g, glu.limit),
                                .geglu_tanh => gegluTanh(u, g),
                            };
                        }
                        const index_h = (@as(u64, row) * glu.code_stride + column) / 8;
                        storeChunk(@ptrFromInt(glu.codes), index_h, bf16.packChunk(values));
                    }
                }
            }
        }

        /// After `transposeQuad`, the quad holds the 32 columns of `h` of a warp tile, which form
        /// one MXFP8 scale block. The columns from `hidden` on hold zeros.
        inline fn storeAtomGlu(
            comptime activation: GluActivation,
            glu: GluOutput,
            problem: Problem,
            warp: Warp,
            tile: Tile,
            sums: *const Accumulators,
        ) void {
            const quad = warp.lane % 4;
            const column = tile.column_offset + warp.n * (warp_n / halves) + quad * 8;
            const scales: [*]addrspace(.global) fp8.Scale(.e8m0) = @ptrFromInt(glu.scales);
            const block = (tile.column_offset + warp.n * (warp_n / halves)) / 32;
            inline for (sums, 0..) |row_fragments, i| {
                inline for (0..2) |h| {
                    var up_pairs: [half_fragments]u32 = undefined;
                    var gate_pairs: [half_fragments]u32 = undefined;
                    inline for (&up_pairs, &gate_pairs, 0..) |*up_pair, *gate_pair, j| {
                        up_pair.* = zPair(row_fragments[j], h);
                        gate_pair.* = zPair(row_fragments[j + half_fragments], h);
                    }
                    const up = transposeQuad(up_pairs, quad);
                    const gate = transposeQuad(gate_pairs, quad);
                    const row = tile.m_offset + warp.m + fragmentRow(i, h, warp.lane);
                    if (row < problem.m) {
                        const index = (@as(u64, row) * glu.z_stride + column) / 8;
                        if (glu.z_up != 0) storeChunk(@ptrFromInt(glu.z_up), index, up);
                        if (glu.z_gate != 0) storeChunk(@ptrFromInt(glu.z_gate), index, gate);
                    }
                    var values: [16]f32 = @splat(0);
                    inline for (values[0..8], bf16.unpackChunk(up), bf16.unpackChunk(gate), 0..) |*value, u, g, e| {
                        const product = switch (activation) {
                            .swiglu => swiglu(u, g, glu.limit),
                            .geglu_tanh => gegluTanh(u, g),
                        };
                        value.* = if (row < problem.m and column + e < glu.hidden) product else 0;
                    }
                    const amax = fp8.maxNanOverLanes(fp8.amaxOf(values[0..8].*), 4);
                    const scaling: fp8_pack.Scaling(.e8m0) = .init(amax);
                    if (row < problem.m) {
                        storeCodes(
                            glu.codes + @as(u64, row) * glu.code_stride + column,
                            fp8_pack.scaledCodes(values, scaling.factor),
                        );
                    }
                    if (quad == 0) {
                        const atom = row / 128 * glu.scale_spans + block / 4;
                        scales[atom * 512 + row % 32 * 16 + row % 128 / 32 * 4 + block % 4] =
                            scaling.scale;
                    }
                }
            }
        }

        inline fn activateChunk(comptime activation: epilogue.Activation, chunk: Chunk) Chunk {
            if (activation == .none) return chunk;
            var values = bf16.unpackChunk(chunk);
            inline for (&values) |*value| value.* = epilogue.activate(activation, value.*);
            return bf16.packChunk(values);
        }

        /// `mxfp8` kernels add the BF16 or FP32 residual to the FP32 sums before the one rounding to
        /// BF16. Bit 3 of `e8m0` marks an FP32 residual.
        inline fn residualPair(output: Output, row: u32, column: u32, problem: Problem) @Vector(2, f32) {
            if (output.residual == 0 or row >= problem.m or column >= problem.columns) return @splat(0);
            const index = (@as(u64, row) * output.residual_stride + column) / 2;
            if (output.residual_f32) {
                const pairs: [*]addrspace(.global) const [2]f32 = @ptrFromInt(output.residual);
                return pairs[index];
            }
            const words: [*]addrspace(.global) const u32 = @ptrFromInt(output.residual);
            return bf16.unpack(words[index]);
        }

        inline fn biasPair(bias: u64, column: u32, columns: u32) @Vector(2, f32) {
            if (bias == 0 or column >= columns) return @splat(0);
            const words: [*]addrspace(.global) const u32 = @ptrFromInt(bias);
            return bf16.unpack(words[column / 2]);
        }

        /// After a transpose in the quad, work item `q` holds the 8 columns of fragment `4 s + q`
        /// of each group `s` of up to four fragments.
        inline fn store(
            comptime activation: epilogue.Activation,
            output: Output,
            problem: Problem,
            first_row: u32,
            first_column: u32,
            lane: u32,
            k_index: u32,
            sums: *const Accumulators,
        ) void {
            const quad = lane % 4;
            const residual: [*]allowzero addrspace(.global) const Chunk =
                @ptrFromInt(output.residual);
            inline for (sums, 0..) |row_fragments, i| {
                if (owns(k_index, i)) {
                    inline for (0..2) |h| {
                        const row = first_row + fragmentRow(i, h, lane);
                        var pairs: [half_fragments]u32 = undefined;
                        inline for (&pairs, 0..) |*pair, j| {
                            const v = row_fragments[j];
                            const column = first_column + @as(u32, j) * 8 + quad * 2;
                            const bias = biasPair(output.bias[0], column, problem.columns);
                            pair.* = switch (kind) {
                                .store => if (atom_scales) store: {
                                    const c = residualPair(output, row, column, problem);
                                    break :store bf16.pack(
                                        v[h * 2] + c[0] + bias[0],
                                        v[h * 2 + 1] + c[1] + bias[1],
                                    );
                                } else bf16.pack(v[h * 2] + bias[0], v[h * 2 + 1] + bias[1]),
                                .glu => unreachable,
                                .gated => blk: {
                                    const g = row_fragments[j + half_fragments];
                                    const gate_bias = biasPair(output.bias[1], column, problem.columns);
                                    var gated: [2]f32 = undefined;
                                    inline for (&gated, 0..) |*value, e| {
                                        value.* = epilogue.gate(
                                            activation,
                                            bf16.round(g[h * 2 + e] + gate_bias[e]),
                                            bf16.round(v[h * 2 + e] + bias[e]),
                                        );
                                    }
                                    break :blk bf16.pack(gated[0], gated[1]);
                                },
                            };
                        }
                        inline for (0..(half_fragments + 3) / 4) |group_index| {
                            var words: [4]u32 = @splat(0);
                            inline for (&words, group_index * 4..) |*word, j| {
                                if (j < half_fragments) word.* = pairs[j];
                            }
                            var chunk = transposeQuad(words, quad);
                            const fragment = @as(u32, group_index) * 4 + quad;
                            const column = first_column + fragment * 8;
                            if (fragment < half_fragments and row < problem.m and
                                column < problem.columns)
                            {
                                if (kind == .store) chunk = activateChunk(activation, chunk);
                                if (!atom_scales and output.residual != 0) {
                                    chunk = epilogue.addChunk(
                                        chunk,
                                        residual[(@as(u64, row) * output.residual_stride + column) / 8],
                                    );
                                }
                                storeChunk(
                                    output.out,
                                    (@as(u64, row) * output.out_stride + column) / 8,
                                    chunk,
                                );
                            }
                        }
                    }
                }
            }
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (configurations.variants) |variant| {
            for (configurations.tilingsOf(variant)) |tiling| {
                const Store = Gemm(variant, .store, tiling);
                @export(&Store.storeKernel, .{
                    .name = configurations.gemmName(variant, .store, tiling),
                });
                if (variant.scaling == .mxfp8) {
                    @export(&Store.lowRankKernel, .{
                        .name = configurations.lowRankName(tiling),
                    });
                }
                if (variant.scaling != .mxfp8) {
                    const Gated = Gemm(variant, .gated, tiling);
                    @export(&Gated.gatedKernel, .{
                        .name = configurations.gemmName(variant, .gated, tiling),
                    });
                }
                if (configurations.hasGlu(variant, tiling)) {
                    @export(&Gemm(variant, .glu, tiling).gluKernel, .{
                        .name = configurations.gemmName(variant, .glu, tiling),
                    });
                }
            }
        }
    }
}

const negative_log2e: f32 = @bitCast(@as(u32, 0xbfb8_aa3b));
const gelu_linear: f32 = @bitCast(@as(u32, 0x4013_5761));
const gelu_cubic: f32 = @bitCast(@as(u32, 0x3dd2_d3e8));

/// The GLU products give the bits of the products of `glu_quant.zig`. That file flushes the
/// subnormal operands of the multiply and the add of the sigmoid, which changes no result:
/// `ex2.approx.ftz` flushes a subnormal input and never returns a subnormal result, and a product
/// with a normal `g` and `log2(e)` is normal (PTX ISA 9.7.3.21, 9.7.3.13).
inline fn swiglu(up: f32, gate: f32, limit: f32) f32 {
    const math = nvptx.math;
    const g = @min(gate, limit);
    const u = @max(@min(up, limit), -limit);
    const exponent = math.mulF32(.nearest_even, .{}, g, negative_log2e);
    const power = math.ex2ApproxF32(true, exponent);
    const sigmoid = math.rcpApproxF32(true, math.addF32(.nearest_even, .{}, power, 1));
    const silu = bf16.round(math.mulF32(.nearest_even, .{}, g, sigmoid));
    return bf16.round(math.mulF32(.nearest_even, .{}, u, silu));
}

inline fn gegluTanh(up: f32, gate: f32) f32 {
    const math = nvptx.math;
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
