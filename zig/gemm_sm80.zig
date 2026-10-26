//! The kernels compute `C = A B^T` of a row-major `[m, k]` A and a row-major `[n, k]` B with BF16
//! tensor cores and FP32 sums. A zero address means no bias or no residual. With `splits > 1`, the
//! last work group of a tile to finish adds the FP32 partial sums of the parts of k in their order.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const bf16 = @import("bf16.zig");
const epilogue = @import("epilogue.zig");
const configurations = @import("gemm_configurations.zig");

const Chunk = bf16.Chunk;
const Activation = epilogue.Activation;
const Kind = configurations.Kind;
const Tiling = configurations.Tiling;
const FewRows = configurations.FewRows;

const Global = [*]addrspace(.global) const u16;
const Sums = @Vector(4, f32);

extern var shared_storage: [0]u8 align(128) addrspace(.shared);

pub const Position = struct {
    row_tile: u32,
    column_tile: u32,
};

/// Returns the tile of C of work group `tile`. The work groups visit the tiles in panels of `panel`
/// tile rows, or of `panel` tile columns when `by_columns` holds, so that concurrent work groups
/// share tiles of A and B in the L2 cache. Assumes a positive `panel`. Assumes that a `u32` holds
/// `panel * tiles_n`, or `panel * tiles_m` when `by_columns` holds.
pub inline fn rasterPosition(
    panel: u32,
    by_columns: bool,
    tile: u32,
    tiles_m: u32,
    tiles_n: u32,
) Position {
    if (!by_columns) {
        const first = tile / (panel * tiles_n) * panel;
        const height = @min(panel, tiles_m - first);
        const within = tile - first * tiles_n;
        return .{ .row_tile = first + within % height, .column_tile = within / height };
    }
    const first = tile / (panel * tiles_m) * panel;
    const width = @min(panel, tiles_n - first);
    const within = tile - first * tiles_m;
    return .{ .row_tile = within / width, .column_tile = first + within % width };
}

const Operands = struct {
    a: Global,
    b_up: Global,
    b_gate: Global,
    a_stride: u32,
    b_stride: u32,
};

const Problem = struct {
    m: u32,
    columns: u32,
    k: u32,
    panel: u32,
    by_columns: u32,
    splits: u32,
    partials: u64,
    counters: u64,
};

const Output = struct {
    out: u64,
    bias_up: u64,
    bias_gate: u64,
    residual: u64 = 0,
    out_stride: u32,
    residual_stride: u32 = 0,
    activation: u32 = 0,
};

fn Entry(comptime kind: Kind, comptime Body: type) type {
    return struct {
        const convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = Body.threads },
        } };

        pub const kernel = switch (kind) {
            .store_bf16 => storeBf16,
            .store_f32 => storeF32,
            .gated => gated,
        };

        /// Assumes positive `m` and `n` and multiples of 8 for `k`, `n` and the strides. Assumes
        /// 16-byte aligned matrices (PTX ISA 6.4.1, 9.7.10.28.3.1) and output rows that do not
        /// overlap. Assumes an `Activation` in `activation` and a `panel` that `rasterPosition`
        /// accepts. Assumes `splits` from 1 to `configurations.max_splits`. Assumes room for the
        /// partial sums at `partials` and zeroed counters at `counters` when `splits > 1`.
        fn storeBf16(
            a: Global,
            b: Global,
            out: u64,
            bias: u64,
            residual: u64,
            partials: u64,
            counters: u64,
            a_stride: u32,
            b_stride: u32,
            out_stride: u32,
            residual_stride: u32,
            m: u32,
            n: u32,
            k: u32,
            panel: u32,
            by_columns: u32,
            splits: u32,
            activation: u32,
        ) callconv(convention) void {
            Body.run(
                .{ .a = a, .b_up = b, .b_gate = b, .a_stride = a_stride, .b_stride = b_stride },
                .{
                    .m = m,
                    .columns = n,
                    .k = k,
                    .panel = panel,
                    .by_columns = by_columns,
                    .splits = splits,
                    .partials = partials,
                    .counters = counters,
                },
                .{
                    .out = out,
                    .bias_up = bias,
                    .bias_gate = bias,
                    .residual = residual,
                    .out_stride = out_stride,
                    .residual_stride = residual_stride,
                    .activation = activation,
                },
            );
        }

        /// Assumes what `storeBf16` assumes, with a multiple of 4 for `out_stride`.
        fn storeF32(
            a: Global,
            b: Global,
            out: u64,
            bias: u64,
            partials: u64,
            counters: u64,
            a_stride: u32,
            b_stride: u32,
            out_stride: u32,
            m: u32,
            n: u32,
            k: u32,
            panel: u32,
            by_columns: u32,
            splits: u32,
        ) callconv(convention) void {
            Body.run(
                .{ .a = a, .b_up = b, .b_gate = b, .a_stride = a_stride, .b_stride = b_stride },
                .{
                    .m = m,
                    .columns = n,
                    .k = k,
                    .panel = panel,
                    .by_columns = by_columns,
                    .splits = splits,
                    .partials = partials,
                    .counters = counters,
                },
                .{ .out = out, .bias_up = bias, .bias_gate = bias, .out_stride = out_stride },
            );
        }

        /// Assumes what `storeBf16` assumes, with `f` in place of `n`.
        fn gated(
            a: Global,
            gate_weight: Global,
            up_weight: Global,
            out: u64,
            gate_bias: u64,
            up_bias: u64,
            residual: u64,
            partials: u64,
            counters: u64,
            a_stride: u32,
            b_stride: u32,
            out_stride: u32,
            residual_stride: u32,
            m: u32,
            f: u32,
            k: u32,
            panel: u32,
            by_columns: u32,
            splits: u32,
            activation: u32,
        ) callconv(convention) void {
            Body.run(
                .{
                    .a = a,
                    .b_up = up_weight,
                    .b_gate = gate_weight,
                    .a_stride = a_stride,
                    .b_stride = b_stride,
                },
                .{
                    .m = m,
                    .columns = f,
                    .k = k,
                    .panel = panel,
                    .by_columns = by_columns,
                    .splits = splits,
                    .partials = partials,
                    .counters = counters,
                },
                .{
                    .out = out,
                    .bias_up = up_bias,
                    .bias_gate = gate_bias,
                    .residual = residual,
                    .out_stride = out_stride,
                    .residual_stride = residual_stride,
                    .activation = activation,
                },
            );
        }
    };
}

const mma_options: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k16,
    .accumulator = .f32,
    .element = .bf16,
};

inline fn activationOf(output: Output) Activation {
    return @fromBackingInt(output.activation);
}

inline fn biasedChunk(sums: [8]f32, bias: u64, column: u32) Chunk {
    if (bias == 0) return bf16.packChunk(sums);
    const biases: [*]addrspace(.global) const Chunk = @ptrFromInt(bias);
    const values = bf16.unpackChunk(biases[column / 8]);
    var biased: [8]f32 = undefined;
    inline for (&biased, sums, values) |*value, sum, bias_value| value.* = sum + bias_value;
    return bf16.packChunk(biased);
}

/// `values` holds the biased sums of `store_bf16`, or the biased up sums of `gated`.
inline fn storeChunk(
    comptime kind: Kind,
    comptime activation: Activation,
    output: Output,
    values: Chunk,
    gates: Chunk,
    row: u32,
    column: u32,
) void {
    var chunk = if (kind == .gated)
        epilogue.gateChunk(activation, gates, values)
    else
        epilogue.activateChunk(activation, values);
    if (output.residual != 0) {
        const residual: [*]addrspace(.global) const Chunk = @ptrFromInt(output.residual);
        const index = (@as(u64, row) * output.residual_stride + column) / 8;
        chunk = epilogue.addChunk(chunk, residual[index]);
    }
    const out: [*]addrspace(.global) Chunk = @ptrFromInt(output.out);
    out[(@as(u64, row) * output.out_stride + column) / 8] = chunk;
}

fn Tiled(comptime kind: Kind, comptime tiling: Tiling) type {
    const row_chunks = tiling.tile_k / 8;
    const copy_rows = tiling.threads() / row_chunks;
    const half = tiling.tile_n / 2;
    if (row_chunks != 4 and row_chunks != 8) @compileError("tile_k must be 32 or 64");
    if (configurations.run_k % tiling.tile_k != 0) @compileError("tile_k must divide run_k");
    if (tiling.tile_m % copy_rows != 0 or tiling.tile_n % copy_rows != 0)
        @compileError("the copy passes must cover the tiles");
    if (tiling.warp_m % 16 != 0 or tiling.warp_n % 16 != 0)
        @compileError("warp tiles hold 16x16 matrices");
    if (half % tiling.warp_n != 0) @compileError("a warp tile lies in one half of the B tile");
    return struct {
        pub const threads = tiling.threads();
        /// A `gated` B tile holds the up rows of `half` columns of C, then their gate rows.
        const block_columns = if (kind == .gated) half else tiling.tile_n;
        const Tiles = struct {
            a: [tiling.stages][tiling.tile_m * row_chunks]Chunk,
            b: [tiling.stages][tiling.tile_n * row_chunks]Chunk,
        };
        const Staged = [tiling.tile_m][tiling.tile_n + configurations.staged_padding]u16;
        const Accumulators = [tiling.warp_m / 16][tiling.warp_n / 8]Sums;
        const warps_m = tiling.tile_m / tiling.warp_m;
        const warps_n = tiling.tile_n / tiling.warp_n;
        const k_steps = tiling.tile_k / 16;
        const a_passes = tiling.tile_m / copy_rows;
        const b_passes = tiling.tile_n / copy_rows;

        comptime {
            std.debug.assert(@sizeOf(Tiles) <= tiling.sharedBytes());
            std.debug.assert(@sizeOf(Staged) <= tiling.sharedBytes());
        }

        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);

        pub inline fn run(operands: Operands, problem: Problem, output: Output) void {
            @setEvalBranchQuota(100_000);
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            const m = problem.m;
            const columns = problem.columns;
            const split = @workGroupId(0) % problem.splits;
            const position = tileOf(problem);
            const column_tile = position.column_tile;
            const m_offset = position.row_tile * tiling.tile_m;
            const all_k_tiles = (problem.k + tiling.tile_k - 1) / tiling.tile_k;
            // The parts of k interleave their k tiles, so that the work groups of a tile read
            // neighboring bytes of each row of B at about the same time.
            const k_tiles = (all_k_tiles -| split + problem.splits - 1) / problem.splits;

            var copy: Copy = undefined;
            const chunk = tid % row_chunks;
            copy.column = split * tiling.tile_k + chunk * 8;
            copy.stride = problem.splits * tiling.tile_k;
            copy.slot = swizzle(tid / row_chunks, chunk, tid / row_chunks % 8);
            copy.k = problem.k;
            inline for (0..a_passes) |pass| {
                const row = m_offset + tid / row_chunks + @as(u32, pass) * copy_rows;
                copy.a_valid[pass] = row < m;
                copy.a_rows[pass] = operands.a + @as(u64, @min(row, m - 1)) * operands.a_stride;
            }
            inline for (0..b_passes) |pass| {
                const tile_row = tid / row_chunks + @as(u32, pass) * copy_rows;
                const row = bRow(column_tile, tile_row);
                const b = if ((pass + 1) * copy_rows <= half)
                    operands.b_up
                else if (pass * copy_rows >= half)
                    operands.b_gate
                else if (tile_row < half) operands.b_up else operands.b_gate;
                copy.b_valid[pass] = row < columns;
                copy.b_rows[pass] = b + @as(u64, @min(row, columns - 1)) * operands.b_stride;
            }

            inline for (0..tiling.stages - 1) |stage| {
                if (stage < k_tiles) copy.tile(stage);
                cp_async.commitGroup();
            }
            const warp_row = warp / warps_n;
            const warp_n = warp % warps_n * tiling.warp_n;
            const live = liveFragments(m - m_offset, warp_row);
            var accumulators: Accumulators = @splat(@splat(@splat(0)));
            var totals: Accumulators = @splat(@splat(@splat(0)));
            cp_async.waitGroup(tiling.stages - 2);
            nvptx.barrierSyncAligned(.work_group, null);
            const warp_tile: WarpTile = .{ .row = warp_row, .column = warp_n, .lane = lane };
            switch (live) {
                inline 0...tiling.warp_m / 16 => |count| {
                    multiplyTiles(count, &copy, warp_tile, k_tiles, &accumulators, &totals);
                },
                else => unreachable,
            }
            cp_async.waitAll();
            if (kind == .store_f32 and k_tiles % run_tiles != 0) addRun(&totals, &accumulators);
            const sums = if (kind == .store_f32) &totals else &accumulators;
            if (problem.splits > 1 and !addSplits(problem, split, tid, live, sums)) return;
            const tile = tileOf(problem);
            const row_offset = tile.row_tile * tiling.tile_m;
            finish(output, m, columns, row_offset, tile.column_tile, warp_row, warp_n, lane, live, sums);
        }

        const WarpTile = struct {
            row: u32,
            column: u32,
            lane: u32,
        };

        /// Multiplies the `k_tiles` k tiles of the work group. The warp multiplies only its first
        /// `live` fragments of A, so that the code has no branches between `mma` instructions.
        inline fn multiplyTiles(
            comptime live: u32,
            copy: *Copy,
            warp_tile: WarpTile,
            k_tiles: u32,
            accumulators: *Accumulators,
            totals: *Accumulators,
        ) void {
            // The warps load the fragments of each k step one step ahead. The loop body takes one k
            // tile of each stage, so that the stages are constants.
            var fragments: [2]Fragments = undefined;
            fragments[0] = loadFragments(live, 0, 0, warp_tile);
            var k_tile: u32 = 0;
            tiles: while (true) {
                inline for (0..tiling.stages) |stage| {
                    if (k_tile == k_tiles) break :tiles;
                    inline for (0..k_steps) |k_step| {
                        if (k_step == k_steps - 1) {
                            cp_async.waitGroup(tiling.stages - 2);
                            nvptx.barrierSyncAligned(.work_group, null);
                        }
                        const next = (k_step + 1) % k_steps;
                        const next_stage = if (next == 0) (stage + 1) % tiling.stages else stage;
                        fragments[next % 2] = loadFragments(live, next_stage, next, warp_tile);
                        multiply(live, accumulators, fragments[k_step % 2]);
                        if (k_step == 0) {
                            const refill_stage = (stage + tiling.stages - 1) % tiling.stages;
                            if (k_tile + tiling.stages - 1 < k_tiles) copy.tile(refill_stage);
                            cp_async.commitGroup();
                        }
                    }
                    if (kind == .store_f32 and (k_tile + 1) % run_tiles == 0) {
                        addRun(totals, accumulators);
                    }
                    k_tile += 1;
                }
            }
        }

        /// The fragments of the warps interleave, so that the rows of a tile that lie below `m`
        /// spread evenly over the warps.
        inline fn fragmentRow(warp_row: u32, comptime i: u32) u32 {
            return (i * warps_m + warp_row) * 16;
        }

        /// Returns the number of fragments of the warps in row `warp_row` that hold rows of C below
        /// `rows`. The row of a fragment grows with its index, so the warps skip every later one.
        inline fn liveFragments(rows: u32, warp_row: u32) u32 {
            const row_fragments = (rows + 15) / 16;
            return @min(tiling.warp_m / 16, (row_fragments -| warp_row + warps_m - 1) / warps_m);
        }

        inline fn tileOf(problem: Problem) Position {
            return rasterPosition(
                problem.panel,
                problem.by_columns != 0,
                @workGroupId(0) / problem.splits,
                (problem.m + tiling.tile_m - 1) / tiling.tile_m,
                (problem.columns + block_columns - 1) / block_columns,
            );
        }

        /// A long chain of `mma` instructions into one FP32 accumulator loses precision, so
        /// `store_f32` sums each run of `configurations.run_k` values of k in fresh accumulators.
        const run_tiles = configurations.run_k / tiling.tile_k;

        inline fn addRun(totals: *Accumulators, accumulators: *Accumulators) void {
            inline for (totals, accumulators) |*total_row, *row| {
                inline for (total_row, row) |*total, *fragment| {
                    total.* += fragment.*;
                    fragment.* = @splat(0);
                }
            }
        }

        const fragment_count = tiling.warp_m / 16 * (tiling.warp_n / 8);

        /// Returns whether this work group is the last of its tile to finish. The last work group
        /// zeroes the counter of the tile and sets `sums` to the partial sums added in part order.
        inline fn addSplits(
            problem: Problem,
            split: u32,
            tid: u32,
            live: u32,
            sums: *Accumulators,
        ) bool {
            const tile_index = @workGroupId(0) / problem.splits;
            const partials: [*]addrspace(.global) Sums = @ptrFromInt(problem.partials);
            const counters: [*]addrspace(.global) u32 = @ptrFromInt(problem.counters);
            const tile_base = @as(u64, tile_index) * problem.splits * fragment_count * threads;
            const own = tile_base + @as(u64, split) * fragment_count * threads + tid;
            inline for (0..fragment_count) |f| {
                if (isLive(f, live)) partials[own + f * threads] = fragmentAt(sums, f).*;
            }
            // The barriers around the release and acquire `atom` of work item 0 order every store
            // of a part before the loads of the last work group (PTX ISA 8.9.4, 8.9.5, 9.7.15.1).
            nvptx.barrierSyncAligned(.work_group, null);
            const last: *addrspace(.shared) u32 = @ptrCast(@alignCast(&shared_storage));
            if (tid == 0) {
                const arrived = nvptx.atom(.gpu, .acq_rel, .add, &counters[tile_index], 1);
                last.* = @intFromBool(arrived == problem.splits - 1);
                if (arrived == problem.splits - 1) counters[tile_index] = 0;
            }
            nvptx.barrierSyncAligned(.work_group, null);
            if (last.* == 0) return false;
            switch (problem.splits) {
                inline 2...configurations.max_splits => |parts| {
                    addParts(parts, problem, tile_base + tid, live, sums);
                },
                else => unreachable,
            }
            return true;
        }

        /// Replaces `sums` with the `parts` partial sums of the tile, added in their order.
        inline fn addParts(
            comptime parts: u32,
            problem: Problem,
            base: u64,
            live: u32,
            sums: *Accumulators,
        ) void {
            const partials: [*]addrspace(.global) Sums = @ptrFromInt(problem.partials);
            inline for (0..fragment_count) |f| {
                if (isLive(f, live)) {
                    var total: Sums = undefined;
                    inline for (0..parts) |part| {
                        const partial = &partials[base + (part * fragment_count + f) * threads];
                        const value = nvptx.cache.load(.{ .operator = .cache_global_level }, partial);
                        total = if (part == 0) value else total + value;
                    }
                    fragmentAt(sums, f).* = total;
                }
            }
        }

        inline fn isLive(comptime f: u32, live: u32) bool {
            return f / (tiling.warp_n / 8) < live;
        }

        inline fn fragmentAt(sums: *Accumulators, comptime f: u32) *Sums {
            return &sums[f / (tiling.warp_n / 8)][f % (tiling.warp_n / 8)];
        }

        inline fn bRow(column_tile: u32, row: u32) u32 {
            if (kind != .gated) return column_tile * tiling.tile_n + row;
            return column_tile * half + row % half;
        }

        const Copy = struct {
            column: u32,
            stride: u32,
            slot: u32,
            k: u32,
            a_valid: [a_passes]bool,
            a_rows: [a_passes]Global,
            b_valid: [b_passes]bool,
            b_rows: [b_passes]Global,

            /// Copies the next k tile into `stage`. A copy of source size 0 writes 16 zero bytes
            /// (PTX ISA 9.7.10.28.3.1), and the zero bytes add nothing to the sums.
            inline fn tile(copy: *Copy, comptime stage: u32) void {
                const column = copy.column;
                copy.column += copy.stride;
                inline for (0..a_passes) |pass| {
                    const slot = copy.slot + pass * copy_rows * row_chunks;
                    const valid = copy.a_valid[pass] and column < copy.k;
                    issue(&tiles.a[stage][slot], copy.a_rows[pass], column, valid);
                }
                inline for (0..b_passes) |pass| {
                    const slot = copy.slot + pass * copy_rows * row_chunks;
                    const valid = copy.b_valid[pass] and column < copy.k;
                    issue(&tiles.b[stage][slot], copy.b_rows[pass], column, valid);
                }
            }

            inline fn issue(
                slot: *addrspace(.shared) Chunk,
                row: Global,
                column: u32,
                valid: bool,
            ) void {
                const source: *addrspace(.global) const Chunk =
                    @ptrCast(@alignCast(row + if (valid) column else 0));
                cp_async.copyZeroFill(.cache_global_level, slot, source, if (valid) 16 else 0);
            }
        };

        /// The swizzle makes an `ldmatrix` matrix use all 32 banks. The callers pass `row % 8`
        /// as `phase` so that the fragment addresses of a work item differ by constants.
        inline fn swizzle(row: u32, column: u32, phase: u32) u32 {
            return row * row_chunks + (column ^ phase * row_chunks / 8);
        }

        const Fragments = struct {
            a: [tiling.warp_m / 16]@Vector(4, u32),
            b: [tiling.warp_n / 8]@Vector(2, u32),
        };

        // PTX ISA 9.7.16.5.8 and 9.7.16.5.15 give the `mma` and `ldmatrix` layouts.
        inline fn loadFragments(
            comptime live: u32,
            comptime stage: u32,
            comptime k_step: u32,
            warp_tile: WarpTile,
        ) Fragments {
            const lane = warp_tile.lane;
            var fragments: Fragments = undefined;
            if (live == 0) return fragments;
            inline for (fragments.a[0..live], 0..) |*fragment, i| {
                const row = fragmentRow(warp_tile.row, i) + lane % 16;
                const slot = swizzle(row, k_step * 2 + lane / 16, lane % 8);
                fragment.* = nvptx.ldmatrix(.x4, false, &tiles.a[stage][slot]);
            }
            // One `ldmatrix.x4` loads the two k halves of n rows 0-7, then of n rows 8-15.
            inline for (0..tiling.warp_n / 16) |j| {
                const pair = nvptx.ldmatrix(.x4, false, &tiles.b[stage][
                    swizzle(
                        warp_tile.column + @as(u32, j) * 16 + lane % 8 + lane / 16 * 8,
                        k_step * 2 + lane / 8 % 2,
                        lane % 8,
                    )
                ]);
                fragments.b[j * 2] = .{ pair[0], pair[1] };
                fragments.b[j * 2 + 1] = .{ pair[2], pair[3] };
            }
            return fragments;
        }

        inline fn multiply(
            comptime live: u32,
            accumulators: *Accumulators,
            fragments: Fragments,
        ) void {
            inline for (accumulators[0..live], fragments.a[0..live]) |*row, a_fragment| {
                inline for (row, fragments.b) |*fragment, b_fragment| {
                    fragment.* = nvptx.mmaSync(mma_options, a_fragment, b_fragment, fragment.*);
                }
            }
        }

        inline fn biasPair(bias: u64, row: u32, rows: u32) @Vector(2, f32) {
            if (bias == 0 or row >= rows) return @splat(0);
            const words: [*]addrspace(.global) const u32 = @ptrFromInt(bias);
            return bf16.unpack(words[row / 2]);
        }

        inline fn finish(
            output: Output,
            m: u32,
            columns: u32,
            m_offset: u32,
            column_tile: u32,
            warp_row: u32,
            warp_n: u32,
            lane: u32,
            live: u32,
            accumulators: *const Accumulators,
        ) void {
            const staged: *addrspace(.shared) Staged = @ptrCast(&shared_storage);
            const bias = if (warp_n < half) output.bias_up else output.bias_gate;
            if (kind != .store_f32) nvptx.barrierSyncAligned(.work_group, null);
            inline for (accumulators, 0..) |row_fragments, i| {
                if (i < live) {
                    inline for (row_fragments, 0..) |fragment, j| {
                        const column = warp_n + @as(u32, j) * 8 + lane % 4 * 2;
                        const sums = biasPair(bias, bRow(column_tile, column), columns);
                        inline for (0..2) |row_half| {
                            const row = fragmentRow(warp_row, i) + lane / 4 + row_half * 8;
                            const v0 = fragment[row_half * 2] + sums[0];
                            const v1 = fragment[row_half * 2 + 1] + sums[1];
                            if (kind == .store_f32) {
                                const out_row = m_offset + row;
                                const out_column = column_tile * tiling.tile_n + column;
                                if (out_row < m and out_column < columns) {
                                    const out: [*]addrspace(.global) [2]f32 = @ptrFromInt(output.out);
                                    out[(@as(u64, out_row) * output.out_stride + out_column) / 2] =
                                        .{ v0, v1 };
                                }
                            } else {
                                const pair: *addrspace(.shared) u32 =
                                    @ptrCast(@alignCast(&staged[row][column]));
                                pair.* = bf16.pack(v0, v1);
                            }
                        }
                    }
                }
            }
            if (kind == .store_f32) return;
            nvptx.barrierSyncAligned(.work_group, null);
            switch (activationOf(output)) {
                inline else => |activation| store(
                    activation,
                    staged,
                    output,
                    m,
                    columns,
                    m_offset,
                    column_tile * block_columns,
                ),
            }
        }

        inline fn chunkAt(tile: *addrspace(.shared) const Staged, row: u32, column: u32) Chunk {
            const chunk: *addrspace(.shared) const Chunk = @ptrCast(@alignCast(&tile[row][column]));
            return chunk.*;
        }

        inline fn store(
            comptime activation: Activation,
            staged: *addrspace(.shared) const Staged,
            output: Output,
            m: u32,
            columns: u32,
            tile_row: u32,
            tile_column: u32,
        ) void {
            const chunk_columns = block_columns / 8;
            const passes = tiling.tile_m * chunk_columns / threads;
            comptime std.debug.assert(passes * threads == tiling.tile_m * chunk_columns);
            const tid = @workItemId(0);
            const out: [*]addrspace(.global) Chunk = @ptrFromInt(output.out);
            var residual_chunks: [passes]Chunk = undefined;
            if (output.residual != 0) {
                const residual: [*]addrspace(.global) const Chunk = @ptrFromInt(output.residual);
                inline for (&residual_chunks, 0..) |*residual_chunk, pass| {
                    const index = tid + @as(u32, pass) * threads;
                    const row = tile_row + index / chunk_columns;
                    const column = tile_column + index % chunk_columns * 8;
                    if (row < m and column < columns) residual_chunk.* =
                        residual[(@as(u64, row) * output.residual_stride + column) / 8];
                }
            }
            inline for (residual_chunks, 0..) |residual_chunk, pass| {
                const index = tid + @as(u32, pass) * threads;
                const local_row = index / chunk_columns;
                const local_column = index % chunk_columns * 8;
                const row = tile_row + local_row;
                const column = tile_column + local_column;
                if (row < m and column < columns) {
                    var chunk = chunkAt(staged, local_row, local_column);
                    chunk = if (kind == .gated) epilogue.gateChunk(
                        activation,
                        chunkAt(staged, local_row, half + local_column),
                        chunk,
                    ) else epilogue.activateChunk(activation, chunk);
                    if (output.residual != 0) chunk = epilogue.addChunk(chunk, residual_chunk);
                    out[(@as(u64, row) * output.out_stride + column) / 8] = chunk;
                }
            }
        }
    };
}

/// A work group multiplies 16 rows of B by up to `shape.max_m` rows of A. For `gated`, the rows of
/// B are the up rows and the gate rows of 8 columns of C. The rows of B form the A operand of
/// `mma`, and the rows of A form the B operand of `mma`, 8 rows at a time. A sum over k may add its
/// terms in any order, so the two operands only need to agree on the value of k at each position.
fn FewRowsGemm(comptime kind: Kind, comptime shape: FewRows) type {
    const a_tiles = shape.max_m / 8;
    const warp_k = shape.warp_k;
    const row_chunks = warp_k / 32;
    const b_rows = configurations.few_rows_b_rows;
    if (b_rows != 16) @compileError("the rows of B fill the 16 rows of one mma operand");
    if (warp_k % 32 != 0) @compileError("warp_k / 4 must be a whole number of chunks");
    return struct {
        pub const threads = shape.threads();
        const Partials = [shape.warps][shape.max_m][configurations.partial_stride]f32;

        comptime {
            std.debug.assert(@sizeOf(Partials) <= shape.sharedBytes());
            std.debug.assert(shape.max_m * 4 <= threads);
        }

        /// A work item reads rows `group_id + 8 j` of A, row `group_id` of the B tile as `low` and
        /// row `group_id + 8` as `high`. For `gated`, `low` and `high` are the up and gate rows.
        const Sources = struct {
            low: Global,
            high: Global,
            a: [a_tiles]Global,
            low_valid: bool,
            high_valid: bool,
            a_valid: [a_tiles]bool,
            k: u32,
            thread_in_group: u32,

            inline fn load(sources: Sources, k_turn: u32) Loaded {
                var loaded: Loaded = undefined;
                inline for (0..row_chunks) |h| {
                    const column = k_turn * warp_k + sources.thread_in_group * (warp_k / 4) +
                        @as(u32, h) * 8;
                    const in_k = column < sources.k;
                    loaded.low[h] = loadChunk(sources.low, column, in_k, sources.low_valid);
                    loaded.high[h] = loadChunk(sources.high, column, in_k, sources.high_valid);
                    inline for (&loaded.a, sources.a, sources.a_valid) |*chunks, row, valid| {
                        chunks[h] = loadChunk(row, column, in_k, valid);
                    }
                }
                return loaded;
            }
        };

        const Loaded = struct {
            low: [row_chunks]Chunk,
            high: [row_chunks]Chunk,
            a: [a_tiles][row_chunks]Chunk,

            /// Fresh accumulators for each `k_turn` keep the chains of `mma` instructions short.
            inline fn multiply(loaded: Loaded, totals: *[a_tiles]Sums) void {
                var sums: [a_tiles]Sums = @splat(@splat(0));
                inline for (0..row_chunks) |h| {
                    inline for (0..2) |s| {
                        const b_fragment: @Vector(4, u32) = .{
                            loaded.low[h][2 * s],
                            loaded.high[h][2 * s],
                            loaded.low[h][2 * s + 1],
                            loaded.high[h][2 * s + 1],
                        };
                        inline for (&sums, loaded.a) |*sum, chunks| {
                            const a_fragment: @Vector(2, u32) = .{
                                chunks[h][2 * s],
                                chunks[h][2 * s + 1],
                            };
                            sum.* = nvptx.mmaSync(mma_options, b_fragment, a_fragment, sum.*);
                        }
                    }
                }
                inline for (totals, sums) |*total, sum| total.* += sum;
            }
        };

        inline fn loadChunk(row: Global, column: u32, in_k: bool, valid: bool) Chunk {
            const chunk: *addrspace(.global) const Chunk =
                @ptrCast(@alignCast(row + if (in_k) column else 0));
            const loaded = chunk.*;
            return if (in_k and valid) loaded else @splat(0);
        }

        pub inline fn run(operands: Operands, problem: Problem, output: Output) void {
            @setEvalBranchQuota(20_000);
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            // PTX ISA 9.7.16.5.8 calls the next two values `groupID` and `threadID_in_group`.
            const group_id = lane / 4;
            const thread_in_group = lane % 4;
            const m = problem.m;
            const columns = problem.columns;
            const first_column = @workGroupId(0) * if (kind == .gated) b_rows / 2 else b_rows;
            const low_row = first_column + group_id;
            const high_row = if (kind == .gated) low_row else low_row + 8;
            var sources: Sources = .{
                .low = operands.b_up + @as(u64, @min(low_row, columns - 1)) * operands.b_stride,
                .high = operands.b_gate + @as(u64, @min(high_row, columns - 1)) * operands.b_stride,
                .a = undefined,
                .low_valid = low_row < columns,
                .high_valid = high_row < columns,
                .a_valid = undefined,
                .k = problem.k,
                .thread_in_group = thread_in_group,
            };
            inline for (&sources.a, &sources.a_valid, 0..) |*row, *valid, j| {
                const a_row = @as(u32, j) * 8 + group_id;
                valid.* = a_row < m;
                row.* = operands.a + @as(u64, @min(a_row, m - 1)) * operands.a_stride;
            }

            var totals: [a_tiles]Sums = @splat(@splat(0));
            const groups = (problem.k + warp_k - 1) / warp_k;
            // Each warp loads its next `k_turn` before it multiplies the current one.
            var k_turn = warp;
            var current = if (k_turn < groups) sources.load(k_turn) else undefined;
            while (k_turn < groups) : (k_turn += shape.warps) {
                const next = if (k_turn + shape.warps < groups)
                    sources.load(k_turn + shape.warps)
                else
                    undefined;
                current.multiply(&totals);
                current = next;
            }

            const partials: *addrspace(.shared) Partials = @ptrCast(&shared_storage);
            inline for (totals, 0..) |total, j| {
                const a_row = @as(u32, j) * 8 + thread_in_group * 2;
                partials[warp][a_row][group_id] = total[0];
                partials[warp][a_row + 1][group_id] = total[1];
                partials[warp][a_row][group_id + 8] = total[2];
                partials[warp][a_row + 1][group_id + 8] = total[3];
            }
            nvptx.barrierSyncAligned(.work_group, null);
            switch (kind) {
                .store_f32 => storeF32(partials, output, m, columns, first_column, tid),
                else => switch (activationOf(output)) {
                    inline else => |activation| storeBf16(
                        activation,
                        partials,
                        output,
                        m,
                        columns,
                        first_column,
                        tid,
                    ),
                },
            }
        }

        /// Adds the sums of the warps in the order of the warps.
        inline fn warpSum(
            comptime count: u32,
            partials: *addrspace(.shared) const Partials,
            row: u32,
            column: u32,
        ) [count]f32 {
            var sums: [count]f32 = undefined;
            inline for (0..shape.warps) |w| {
                inline for (0..count / 4) |q| {
                    const vector: *addrspace(.shared) const Sums =
                        @ptrCast(@alignCast(&partials[w][row][column + q * 4]));
                    const values: [4]f32 = vector.*;
                    inline for (0..4) |e| {
                        sums[q * 4 + e] = if (w == 0) values[e] else sums[q * 4 + e] + values[e];
                    }
                }
            }
            return sums;
        }

        inline fn storeBf16(
            comptime activation: Activation,
            partials: *addrspace(.shared) const Partials,
            output: Output,
            m: u32,
            columns: u32,
            first_column: u32,
            tid: u32,
        ) void {
            const chunks_per_row = if (kind == .gated) 1 else 2;
            const row = tid / chunks_per_row;
            const chunk = tid % chunks_per_row;
            const column = first_column + chunk * 8;
            if (row >= @min(m, shape.max_m) or column >= columns) return;
            const sums = warpSum(8, partials, row, chunk * 8);
            const values = biasedChunk(sums, output.bias_up, column);
            const gates = if (kind == .gated)
                biasedChunk(warpSum(8, partials, row, 8), output.bias_gate, column)
            else
                undefined;
            storeChunk(kind, activation, output, values, gates, row, column);
        }

        inline fn storeF32(
            partials: *addrspace(.shared) const Partials,
            output: Output,
            m: u32,
            columns: u32,
            first_column: u32,
            tid: u32,
        ) void {
            const row = tid / 4;
            const column = first_column + tid % 4 * 4;
            if (row >= @min(m, shape.max_m) or column >= columns) return;
            var values = warpSum(4, partials, row, tid % 4 * 4);
            if (output.bias_up != 0) {
                const biases: [*]addrspace(.global) const [2]u32 = @ptrFromInt(output.bias_up);
                const words = biases[column / 4];
                inline for (0..2) |w| {
                    const pair = bf16.unpack(words[w]);
                    values[2 * w] += pair[0];
                    values[2 * w + 1] += pair[1];
                }
            }
            const out: [*]addrspace(.global) [4]f32 = @ptrFromInt(output.out);
            out[(@as(u64, row) * output.out_stride + column) / 4] = values;
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Kind)) |kind| {
            for (configurations.tilingsOf(kind)) |tiling| {
                const name = configurations.kernelName(kind, tiling);
                @export(&Entry(kind, Tiled(kind, tiling)).kernel, .{ .name = name });
            }
            for (configurations.few_rows) |shape| {
                const name = configurations.fewRowsName(kind, shape);
                @export(&Entry(kind, FewRowsGemm(kind, shape)).kernel, .{ .name = name });
            }
        }
    }
}

test "rasterPosition maps the work groups of a 7 by 5 grid onto each tile once" {
    const tiles_m = 7;
    const tiles_n = 5;
    for ([_]bool{ false, true }) |by_columns| {
        for ([_]u32{ 1, 2, 3, 4, 6, 8, 16 }) |panel| {
            var seen: [tiles_m][tiles_n]bool = @splat(@splat(false));
            for (0..tiles_m * tiles_n) |tile| {
                const p = rasterPosition(panel, by_columns, @intCast(tile), tiles_m, tiles_n);
                try std.testing.expect(!seen[p.row_tile][p.column_tile]);
                seen[p.row_tile][p.column_tile] = true;
            }
        }
    }
}
