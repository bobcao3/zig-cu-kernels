//! `Y = A Bᵀ` of FP8 packs with `wgmma` on SM90. A cluster of two CTAs computes two tiles of `Y`
//! that share their columns. In each CTA, a producer warpgroup copies `A` and half of `B` with
//! tensor copies into a ring of 128-deep K stages, multicasting `B` into both CTAs, and two or three
//! consumer warpgroups multiply 64 rows each. A promoted kernel multiplies each K block into a
//! temporary accumulator and adds it, scaled for the block-scaled recipes, into FP32 sums. The
//! rowwise recipe scales the sums in the epilogue.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("mm_sm90_configurations.zig");
const bf16 = @import("bf16.zig");
const bulk = @import("bulk.zig");
const cluster = @import("cluster.zig");
const fp8 = @import("fp8.zig");
const fp8_pack = @import("fp8_pack.zig");
const pipeline = @import("pipeline.zig");
const tma = @import("tma.zig");
const wgmma = @import("wgmma.zig");
const warp = @import("warp.zig");

const Variant = configurations.Variant;
const k_block = configurations.k_block;
const ctas = configurations.cluster_ctas;
const math = nvptx.math;
const negative_log2e: f32 = @bitCast(@as(u32, 0xbfb8_aa3b));
const gelu_linear: f32 = @bitCast(@as(u32, 0x4013_5761));
const gelu_cubic: f32 = @bitCast(@as(u32, 0x3dd2_d3e8));

extern var shared_storage: [0]u8 align(1024) addrspace(.shared);

inline fn waitPhase(barrier: *addrspace(.shared) nvptx.Mbarrier, phase: u1) void {
    if (barrier.tryWaitPhase(phase, null, null)) return;
    const start = nvptx.globaltimer();
    while (!barrier.tryWaitPhase(phase, null, null)) {
        if (nvptx.globaltimer() -% start > configurations.wait_limit_ns) asm volatile ("trap;");
    }
}

inline fn e8m0(byte: u8) f32 {
    const bits: u32 = if (byte == 0) 0x0040_0000 else @as(u32, byte) << 23;
    return @bitCast(bits);
}

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

/// The byte of `(row, block)` in a 4-D scale tensor whose rows hold `atoms_per_row` atoms.
inline fn atomIndex(row: u32, block: u32, atoms_per_row: u32) u32 {
    const atom = row / 128 * atoms_per_row + block / 4;
    return atom * 512 + row % 32 * 16 + row % 128 / 32 * 4 + block % 4;
}

const Tile = struct { m: u32, n: u32 };

/// Each cluster computes the tiles `first`, `first + stride`, ... of a walk through bands of
/// `band_pairs` cluster rows, column by column, so that the tiles running at the same time share
/// rows of `A` and columns of `B` in L2.
const Walk = struct {
    first: u32,
    stride: u32,
    pairs_m: u32,
    tiles_n: u32,
    band_pairs: u32,

    inline fn clusters(walk: Walk) u32 {
        return walk.pairs_m * walk.tiles_n;
    }

    inline fn tile(walk: Walk, cluster_index: u32, rank: u32) Tile {
        const band_size = walk.band_pairs * walk.tiles_n;
        const band = cluster_index / band_size;
        const within = cluster_index % band_size;
        const first = band * walk.band_pairs;
        const height = @min(walk.band_pairs, walk.pairs_m - first);
        return .{ .m = 2 * (first + within % height) + rank, .n = within / height };
    }
};

const Problem = struct {
    y: [*]addrspace(.global) u32,
    s_a: u64,
    s_b: u64,
    bias: [*]addrspace(.global) const u32,
    m: u32,
    n: u32,
    k_blocks: u32,
    scale_blocks: u32,
    h_scales: u64 = 0,
    z: u64 = 0,
    half: u32 = 0,
    hidden: u32 = 0,
    gate_first: bool = false,
    limit: f32 = 0,
    partials: u64 = 0,
    counters: u64 = 0,
    whole: u32 = 0,
    splits: u32 = 1,
};

fn Kernel(comptime variant: Variant) type {
    return struct {
        const n = variant.n;
        const tile_m = variant.tileM();
        const registers = variant.registers();
        const Ring = pipeline.Pipeline(variant.stages, .cta);
        const a_bytes = tile_m * k_block;
        const b_rows_per_cta = n / ctas;
        const stage_bytes = variant.stageBytes();
        const Scale = if (variant.recipe == .fp8_b128_e8m0) u8 else f32;
        const scale_stage_bytes = variant.scaleStageBytes();
        const atom_bytes = 512 * @sizeOf(Scale);
        const options: std.lang.CallingConvention.NvptxKernelOptions = .{
            .required_work_group_size = .{ .x = variant.threads() },
            .cluster_size = .{ .x = ctas },
        };

        var ring: Ring addrspace(.shared) = undefined;

        /// A probe kernel writes the first four fields, the clocks at the end of the tile and of
        /// `%globaltimer`, `promotion` and `release` as the low and high halves of one word, and
        /// `issue` and `scales` in the same way, per CTA, tile and warp role to the bias argument.
        const Clocks = struct {
            start: u64 = 0,
            stage_wait: u32 = 0,
            mma_wait: u32 = 0,
            loop_end: u64 = 0,
            promotion: u32 = 0,
            issue: u32 = 0,
            scales: u32 = 0,
            release: u32 = 0,
            timeline: if (variant.timeline) Timeline else void = if (variant.timeline) .{} else {},

            inline fn stamp() u64 {
                return if (variant.probe) nvptx.clock64() else 0;
            }

            inline fn now() u32 {
                return if (variant.probe) nvptx.clock() else 0;
            }

            const Field = enum { stage_wait, mma_wait, promotion, issue, scales, release };

            inline fn add(clocks: *Clocks, comptime field: Field, since: u32) void {
                if (variant.probe) @field(clocks, @tagName(field)) +%= nvptx.clock() -% since;
            }

            inline fn at(clocks: *Clocks, block: u32) void {
                if (variant.timeline) clocks.timeline.at(block);
            }

            inline fn mark(clocks: *Clocks, comptime point: u32) void {
                if (variant.timeline) clocks.timeline.mark(point);
            }

            inline fn write(clocks: Clocks, buffer: [*]addrspace(.global) const u32, tile: u32, role: u32) void {
                if (!variant.probe or tile >= configurations.probe_tiles) return;
                const records: [*]addrspace(.global) u64 = @ptrFromInt(@intFromPtr(buffer));
                const record = (@workGroupId(0) * configurations.probe_tiles + tile) * (variant.consumers + 1) + role;
                records[8 * record ..][0..8].* = .{
                    clocks.start,
                    clocks.stage_wait,
                    clocks.mma_wait,
                    clocks.loop_end,
                    nvptx.clock64(),
                    nvptx.globaltimer(),
                    clocks.promotion | @as(u64, clocks.release) << 32,
                    clocks.issue | @as(u64, clocks.scales) << 32,
                };
            }
        };

        /// The record of K block `entry` of warp role `role` in CTA `c` holds eight clocks at word
        /// `((c * roles + role) * timeline_blocks + entry) * 8`.
        const Timeline = struct {
            words: [*]addrspace(.global) u32 = undefined,
            first: u32 = configurations.timeline_blocks,
            record: ?[*]addrspace(.global) u32 = null,

            inline fn init(buffer: [*]addrspace(.global) const u32, role: u32, first: u32, recording: bool) Timeline {
                const cta = @workGroupId(0);
                const words: [*]addrspace(.global) u32 = @ptrFromInt(@intFromPtr(buffer));
                const blocks = configurations.timeline_blocks;
                return .{
                    .words = words + (cta * (variant.consumers + 1) + role) * blocks * 8,
                    .first = if (recording and cta < configurations.timeline_ctas) first else blocks,
                };
            }

            inline fn at(timeline: *Timeline, block: u32) void {
                const entry = timeline.first + block;
                timeline.record = if (entry < configurations.timeline_blocks) timeline.words + 8 * entry else null;
            }

            inline fn mark(timeline: Timeline, comptime point: u32) void {
                if (timeline.record) |record| record[point] = nvptx.clock();
            }
        };

        fn kernel(
            a_map: tma.TensorMap,
            b_map: tma.TensorMap,
            y: [*]addrspace(.global) u32,
            s_a: u64,
            s_b: u64,
            bias: [*]addrspace(.global) const u32,
            m: u32,
            width: u32,
            k_blocks: u32,
            scale_blocks: u32,
            tiles_n: u32,
        ) callconv(.{ .nvptx_kernel = options }) void {
            start(&a_map, &b_map, tiles_n, .{
                .y = y,
                .s_a = s_a,
                .s_b = s_b,
                .bias = bias,
                .m = m,
                .n = width,
                .k_blocks = k_blocks,
                .scale_blocks = scale_blocks,
            });
        }

        fn kernelSplit(
            a_map: tma.TensorMap,
            b_map: tma.TensorMap,
            y: [*]addrspace(.global) u32,
            s_a: u64,
            s_b: u64,
            partials: u64,
            counters: u64,
            m: u32,
            width: u32,
            k_blocks: u32,
            scale_blocks: u32,
            tiles_n: u32,
            whole: u32,
            splits: u32,
        ) callconv(.{ .nvptx_kernel = options }) void {
            start(&a_map, &b_map, tiles_n, .{
                .y = y,
                .s_a = s_a,
                .s_b = s_b,
                .bias = @ptrFromInt(8),
                .m = m,
                .n = width,
                .k_blocks = k_blocks,
                .scale_blocks = scale_blocks,
                .partials = partials,
                .counters = counters,
                .whole = whole,
                .splits = splits,
            });
        }

        fn kernelGlu(
            a_map: tma.TensorMap,
            b_map: tma.TensorMap,
            codes: [*]addrspace(.global) u32,
            h_scales: u64,
            z: u64,
            s_a: u64,
            s_b: u64,
            m: u32,
            half: u32,
            hidden: u32,
            k_blocks: u32,
            scale_blocks: u32,
            gate_first: u32,
            limit: f32,
        ) callconv(.{ .nvptx_kernel = options }) void {
            start(&a_map, &b_map, half / (n / 2), .{
                .y = codes,
                .s_a = s_a,
                .s_b = s_b,
                .bias = @ptrFromInt(8),
                .m = m,
                .n = 2 * half,
                .k_blocks = k_blocks,
                .scale_blocks = scale_blocks,
                .h_scales = h_scales,
                .z = z,
                .half = half,
                .hidden = hidden,
                .gate_first = gate_first != 0,
                .limit = limit,
            });
        }

        inline fn start(
            a_map: *const tma.TensorMap,
            b_map: *const tma.TensorMap,
            tiles_n: u32,
            problem: Problem,
        ) void {
            const m = problem.m;
            const base: u32 = @intCast(@intFromPtr(&shared_storage));
            const tiles = std.mem.alignForward(u32, base, 1024);
            const tid = @workItemId(0);
            const rank = cluster.rank();
            const walk: Walk = .{
                .first = @workGroupId(0) / ctas,
                .stride = nvptx.nctaid(0) / ctas,
                .pairs_m = std.math.divCeil(u32, m, 2 * tile_m) catch unreachable,
                .tiles_n = tiles_n,
                .band_pairs = variant.band_pairs,
            };
            if (tid == 0) ring.init(1, 4 * variant.consumers * ctas);
            cluster.sync();
            if (tid < 128) {
                cluster.setMaxRegisters(.dec, registers.producer);
                if (tid == 0) produce(tiles, a_map, b_map, walk, rank, problem);
            } else {
                cluster.setMaxRegisters(.inc, registers.consumer);
                consume(tiles, tid / 128 - 1, tid % 128, walk, rank, problem);
            }
            cluster.sync();
        }

        const Work = struct { cluster_index: u32, begin: u32, end: u32, part: u32 };

        /// A split kernel computes the first `whole` cluster tiles over all K blocks and splits the
        /// K blocks of each later cluster tile into `splits` ranges, one work item per range.
        inline fn work(walk: Walk, item: u32, problem: Problem) Work {
            if (!variant.split_tail or item < problem.whole)
                return .{ .cluster_index = item, .begin = 0, .end = problem.k_blocks, .part = 0 };
            const tail = walk.clusters() - problem.whole;
            const split = item - problem.whole;
            const part = split / tail;
            return .{
                .cluster_index = problem.whole + split % tail,
                .begin = part * problem.k_blocks / problem.splits,
                .end = (part + 1) * problem.k_blocks / problem.splits,
                .part = part,
            };
        }

        inline fn items(walk: Walk, problem: Problem) u32 {
            if (!variant.split_tail) return walk.clusters();
            return problem.whole + (walk.clusters() - problem.whole) * problem.splits;
        }

        fn produce(
            tiles: u32,
            a_map: *const tma.TensorMap,
            b_map: *const tma.TensorMap,
            walk: Walk,
            rank: u32,
            problem: Problem,
        ) void {
            const bias = problem.bias;
            const b_offset = rank * b_rows_per_cta;
            const everyone: u16 = (1 << ctas) - 1;
            const shared_scales = comptime variant.shared_scales;
            comptime std.debug.assert(!shared_scales or tile_m == 128);
            var slot: Ring.Slot = .{};
            var cluster_index = walk.first;
            var count: u32 = 0;
            while (cluster_index < items(walk, problem)) : ({
                cluster_index += walk.stride;
                count += 1;
            }) {
                var clocks: Clocks = .{ .start = Clocks.stamp() };
                if (comptime variant.timeline) clocks.timeline = .init(bias, 0, count * problem.k_blocks, true);
                const job = work(walk, cluster_index, problem);
                const tile = walk.tile(job.cluster_index, rank);
                const a_row: i32 = @intCast(tile.m * tile_m);
                const b_row: i32 = @intCast(if (variant.glu())
                    tile.n * b_rows_per_cta + rank * problem.half
                else
                    tile.n * n + b_offset);
                const columns: ColumnScales = if (shared_scales) .init(problem, tile.n) else undefined;
                const atom_row = if (shared_scales) @min(tile.m, (problem.m - 1) / 128) * (problem.scale_blocks / 4) else 0;
                var next_columns: [2]Scale = if (shared_scales) columns.load(problem, job.begin) else undefined;
                var block: u32 = job.begin;
                while (block < job.end) : (block += 1) {
                    const current_columns = next_columns;
                    if (shared_scales) next_columns = columns.load(problem, @min(block + 1, job.end - 1));
                    clocks.at(block - job.begin);
                    clocks.mark(0);
                    const since = Clocks.now();
                    waitPhase(ring.emptyBarrier(slot), ~slot.phase);
                    clocks.add(.stage_wait, since);
                    clocks.mark(1);
                    const scale_area = if (shared_scales) tiles + variant.stages * stage_bytes + slot.index * scale_stage_bytes else 0;
                    if (shared_scales) {
                        const pair: *addrspace(.shared) [2]Scale = @ptrFromInt(scale_area + atom_bytes);
                        pair.* = current_columns;
                    }
                    ring.producerExpect(slot, stage_bytes + if (shared_scales) atom_bytes else 0);
                    const k: i32 = @intCast(block * k_block);
                    const stage = tiles + slot.index * stage_bytes;
                    const full = ring.fullBarrier(slot);
                    tma.load(2, stage, a_map, .{ k, a_row }, full, null);
                    const b_stage = stage + a_bytes + b_offset * k_block;
                    tma.loadMulticast(2, b_stage, b_map, .{ k, b_row }, full, everyone, null);
                    if (shared_scales) {
                        const atom = (atom_row + block / 4) * atom_bytes;
                        bulk.load(@ptrFromInt(scale_area), @ptrFromInt(problem.s_a + atom), atom_bytes, full);
                    }
                    clocks.mark(2);
                    slot = slot.next();
                }
                clocks.loop_end = Clocks.stamp();
                clocks.write(bias, count, 0);
            }
        }

        inline fn release(slot: Ring.Slot, thread: u32) void {
            if (variant.early) {
                const lane = thread % 32;
                if (lane < ctas) ring.consumerReleaseRemote(slot, lane);
                return;
            }
            if (thread % 32 != 0) return;
            inline for (0..ctas) |target| ring.consumerReleaseRemote(slot, target);
        }

        inline fn rowScale(problem: Problem, row: u32, block: u32) Scale {
            if (row >= problem.m) return 0;
            const blocks = problem.scale_blocks;
            const atom = row / 128 * (blocks / 4) + block / 4;
            const index = atom * 512 + row % 32 * 16 + row % 128 / 32 * 4 + block % 4;
            return @as([*]addrspace(.global) const Scale, @ptrFromInt(problem.s_a))[index];
        }

        inline fn decode(scale: Scale) f32 {
            return if (Scale == u8) e8m0(scale) else scale;
        }

        /// The undecoded scales of the rows `row` and `row + 8` and of the two 128-column scale
        /// tiles of `B` that a tile wider than 128 columns reaches.
        const Raw = struct { rows: [2]Scale, low: Scale, high: Scale };

        /// The scale products of the rows `row` and `row + 8` for the columns of the tile before
        /// column 64, between columns 64 and 128, and from column 128 on.
        const Scales = [3][2]f32;
        const ones: Scales = @splat(.{ 1, 1 });

        inline fn loadScales(problem: Problem, row: u32, tile_n: u32, block: u32) Raw {
            const s_b: [*]addrspace(.global) const Scale = @ptrFromInt(problem.s_b);
            const column_tile = tile_n * n / 128;
            const low = s_b[column_tile * problem.scale_blocks + block];
            const high = if (n > 128) high: {
                const last = (problem.n - 1) / 128;
                break :high s_b[@min(column_tile + 1, last) * problem.scale_blocks + block];
            } else low;
            return .{
                .rows = .{ rowScale(problem, row, block), rowScale(problem, row + 8, block) },
                .low = low,
                .high = high,
            };
        }

        /// The element offsets of the scales of row `row` and of both column tiles at K block 0. A
        /// row at or past `m` reads the row of the same index in the last 128-row atom and is never
        /// stored. Row `row + 8` sits 8 · 16 elements after row `row`, because `row % 16 < 8`.
        const ScaleBases = struct {
            row: u32,
            low: u32,
            high: u32,

            inline fn init(problem: Problem, row: u32, tile_n: u32) ScaleBases {
                const clamped = if (row < problem.m) row else (problem.m - 1) / 128 * 128 + row % 128;
                const column_tile = tile_n * n / 128;
                const high_tile = if (n > 128) @min(column_tile + 1, (problem.n - 1) / 128) else column_tile;
                return .{
                    .row = atomIndex(clamped, 0, problem.scale_blocks / 4),
                    .low = column_tile * problem.scale_blocks,
                    .high = high_tile * problem.scale_blocks,
                };
            }

            inline fn load(bases: ScaleBases, problem: Problem, block: u32) Raw {
                const s_a: [*]addrspace(.global) const Scale = @ptrFromInt(problem.s_a);
                const s_b: [*]addrspace(.global) const Scale = @ptrFromInt(problem.s_b);
                const row_index = bases.row + block / 4 * 512 + block % 4;
                return .{
                    .rows = .{ s_a[row_index], s_a[row_index + 8 * 16] },
                    .low = s_b[bases.low + block],
                    .high = s_b[bases.high + block],
                };
            }
        };

        /// The element offsets of the scales of the two column tiles of `B` at K block 0.
        const ColumnScales = struct {
            low: u32,
            high: u32,

            inline fn init(problem: Problem, tile_n: u32) ColumnScales {
                const column_tile = tile_n * n / 128;
                const high_tile = if (n > 128) @min(column_tile + 1, (problem.n - 1) / 128) else column_tile;
                return .{ .low = column_tile * problem.scale_blocks, .high = high_tile * problem.scale_blocks };
            }

            inline fn load(columns: ColumnScales, problem: Problem, block: u32) [2]Scale {
                const s_b: [*]addrspace(.global) const Scale = @ptrFromInt(problem.s_b);
                return .{ s_b[columns.low + block], s_b[columns.high + block] };
            }
        };

        /// Reads the scales of a K block from its stage. `atom_offset` is the offset of row `row`
        /// in the scale atom, and row `row + 8` sits 8 · 16 elements after it.
        inline fn sharedScales(tiles: u32, slot: Ring.Slot, atom_offset: u32, block: u32) Raw {
            const scale_area = tiles + variant.stages * stage_bytes + slot.index * scale_stage_bytes;
            const atom: [*]addrspace(.shared) const Scale = @ptrFromInt(scale_area);
            const index = atom_offset + block % 4;
            return .{ .rows = .{ atom[index], atom[index + 8 * 16] }, .low = atom[512], .high = atom[513] };
        }

        inline fn decodeScales(raw: Raw, tile_n: u32) Scales {
            const low = decode(raw.low);
            const high = decode(raw.high);
            const middle = if (tile_n * n % 128 == 0) low else high;
            const rows: [2]f32 = .{ decode(raw.rows[0]), decode(raw.rows[1]) };
            return .{
                .{ rows[0] * low, rows[1] * low },
                .{ rows[0] * middle, rows[1] * middle },
                .{ rows[0] * high, rows[1] * high },
            };
        }

        fn consume(tiles: u32, consumer: u32, thread: u32, walk: Walk, rank: u32, problem: Problem) void {
            var slot: Ring.Slot = .{};
            var cluster_index = walk.first;
            var count: u32 = 0;
            while (cluster_index < items(walk, problem)) : ({
                cluster_index += walk.stride;
                count += 1;
            }) {
                var clocks: Clocks = .{ .start = Clocks.stamp() };
                if (comptime variant.timeline)
                    clocks.timeline = .init(problem.bias, 1 + consumer, count * problem.k_blocks, thread == 0);
                const job = work(walk, cluster_index, problem);
                const tile = walk.tile(job.cluster_index, rank);
                const row = tile.m * tile_m + consumer * 64 + thread / 32 * 16 + thread % 32 / 4;
                var accumulator: wgmma.Accumulator(n) = @splat(0);
                if (comptime variant.glu()) {
                    promoteGlu(tiles, consumer, thread, tile, problem, row, &slot, &accumulator, &clocks);
                    const column = tile.n * b_rows_per_cta + 2 * (thread % 4);
                    if (comptime variant.blockScaled())
                        storeGlu(problem, row, column, accumulator)
                    else
                        storeGluRowwise(problem, row, column, accumulator);
                    continue;
                }
                if (variant.split_tail)
                    promoteShared(tiles, consumer, thread, tile, row, job.begin, job.end, &slot, &accumulator, &clocks)
                else if (variant.promoted and variant.overlapped)
                    promote(tiles, consumer, thread, tile, problem, row, &slot, &accumulator, &clocks)
                else if (variant.promoted and variant.deferred)
                    promoteEach(tiles, consumer, thread, tile, problem, row, &slot, &accumulator, &clocks)
                else if (variant.promoted)
                    promoteEager(tiles, consumer, thread, tile, problem, row, &slot, &accumulator, &clocks)
                else
                    accumulate(tiles, consumer, thread, problem, &slot, &accumulator, &clocks);
                clocks.loop_end = Clocks.stamp();
                if (variant.split_tail and cluster_index >= problem.whole) {
                    const group = ((job.cluster_index - problem.whole) * ctas + rank) * variant.consumers + consumer;
                    if (!addRanges(problem, job.part, group, consumer, thread, &accumulator)) continue;
                }
                store(problem, row, tile.n * n + 2 * (thread % 4), accumulator);
                if (thread == 0) clocks.write(problem.bias, count, 1 + consumer);
            }
        }

        inline fn issue(
            tiles: u32,
            consumer: u32,
            slot: Ring.Slot,
            partial: *wgmma.Accumulator(n),
            clocks: *Clocks,
            ready: *bool,
        ) void {
            waitFull(slot, clocks, ready.*);
            issueFull(tiles, consumer, slot, partial, clocks, ready);
        }

        inline fn waitFull(slot: Ring.Slot, clocks: *Clocks, ready: bool) void {
            const since = Clocks.now();
            if (!variant.early or !ready) waitPhase(ring.fullBarrier(slot), slot.phase);
            clocks.add(.stage_wait, since);
        }

        inline fn issueFull(
            tiles: u32,
            consumer: u32,
            slot: Ring.Slot,
            partial: *wgmma.Accumulator(n),
            clocks: *Clocks,
            ready: *bool,
        ) void {
            clocks.mark(1);
            const issuing = Clocks.now();
            const stage = tiles + slot.index * stage_bytes;
            const a: wgmma.Descriptor = .kMajor(stage + consumer * 64 * k_block, .bytes_128);
            const b: wgmma.Descriptor = .kMajor(stage + a_bytes, .bytes_128);
            wgmma.fence();
            const advance: wgmma.Advance = if (variant.hoisted) .low else .wide;
            wgmma.mmaKStepsAdvancing(advance, n, 4, .bytes_128, 64, n, partial, a, b, false);
            wgmma.commit();
            clocks.add(.issue, issuing);
            clocks.mark(2);
            if (variant.early and !variant.late_try_wait) {
                const next = slot.next();
                ready.* = if (variant.test_wait)
                    ring.fullBarrier(next).testWaitPhase(next.phase)
                else
                    ring.fullBarrier(next).tryWaitPhase(next.phase, null, null);
            }
            clocks.mark(3);
        }

        inline fn add(accumulator: *wgmma.Accumulator(n), partial: wgmma.Accumulator(n), scales: Scales) void {
            inline for (0..n / 2) |index| {
                const group = @min(8 * (index / 4) / 64, 2);
                accumulator[index] = if (comptime variant.blockScaled())
                    @mulAdd(f32, partial[index], scales[group][index / 2 % 2], accumulator[index])
                else
                    accumulator[index] + partial[index];
            }
        }

        /// Two temporary accumulators alternate between the K blocks, so that the `wgmma` of one
        /// block runs while the consumer adds the product of the previous block to the sums.
        inline fn promote(
            tiles: u32,
            consumer: u32,
            thread: u32,
            tile: Tile,
            problem: Problem,
            row: u32,
            slot: *Ring.Slot,
            accumulator: *wgmma.Accumulator(n),
            clocks: *Clocks,
        ) void {
            const block_scaled = comptime variant.blockScaled();
            var partials: [2]wgmma.Accumulator(n) = .{ @splat(0), @splat(0) };
            var raws: [2]Raw = undefined;
            var previous: Ring.Slot = slot.*;
            var ready = false;
            var block: u32 = 0;
            while (block < problem.k_blocks) : (block += 2) {
                inline for (0..2) |parity| {
                    const current = block + @as(u32, parity);
                    if (current < problem.k_blocks) {
                        issue(tiles, consumer, slot.*, &partials[parity], clocks, &ready);
                        if (block_scaled) raws[parity] = loadScales(problem, row, tile.n, current);
                        if (current != 0) {
                            const since = Clocks.now();
                            wgmma.wait(1);
                            clocks.add(.mma_wait, since);
                            wgmma.fenceOperand(&partials[1 - parity]);
                            release(previous, thread);
                            const scales = if (block_scaled) decodeScales(raws[1 - parity], tile.n) else ones;
                            add(accumulator, partials[1 - parity], scales);
                        }
                        previous = slot.*;
                        slot.* = slot.next();
                    }
                }
            }
            wgmma.wait(0);
            release(previous, thread);
            const last = (problem.k_blocks - 1) % 2;
            inline for (0..2) |parity| {
                if (last == parity) {
                    wgmma.fenceOperand(&partials[parity]);
                    const scales = if (block_scaled) decodeScales(raws[parity], tile.n) else ones;
                    add(accumulator, partials[parity], scales);
                }
            }
        }

        /// The scales of each K block load one block ahead and decode after the `wgmma` wait. The
        /// loop body is unrolled by 2 so that the two sets of scales alternate between fixed
        /// registers, and each load has one whole block to complete.
        inline fn promoteEach(
            tiles: u32,
            consumer: u32,
            thread: u32,
            tile: Tile,
            problem: Problem,
            row: u32,
            slot: *Ring.Slot,
            accumulator: *wgmma.Accumulator(n),
            clocks: *Clocks,
        ) void {
            const block_scaled = comptime variant.blockScaled();
            var partial: wgmma.Accumulator(n) = @splat(0);
            var raws: [2]Raw = undefined;
            if (block_scaled) raws[0] = loadScales(problem, row, tile.n, 0);
            var ready = false;
            var block: u32 = 0;
            while (block < problem.k_blocks) : (block += 2) {
                inline for (0..2) |parity| {
                    const current = block + @as(u32, parity);
                    if (current < problem.k_blocks) {
                        issue(tiles, consumer, slot.*, &partial, clocks, &ready);
                        const loading = Clocks.now();
                        const next_block = @min(current + 1, problem.k_blocks - 1);
                        if (block_scaled) raws[1 - parity] = loadScales(problem, row, tile.n, next_block);
                        clocks.add(.scales, loading);
                        const since = Clocks.now();
                        wgmma.wait(0);
                        clocks.add(.mma_wait, since);
                        wgmma.fenceOperand(&partial);
                        const promoting = Clocks.now();
                        release(slot.*, thread);
                        const scales = if (block_scaled) decodeScales(raws[parity], tile.n) else ones;
                        add(accumulator, partial, scales);
                        clocks.add(.promotion, promoting);
                        slot.* = slot.next();
                    }
                }
            }
        }

        inline fn rawScales(problem: Problem, row: u32, tile_n: u32, bases: ScaleBases, block: u32) Raw {
            return if (variant.hoisted) bases.load(problem, block) else loadScales(problem, row, tile_n, block);
        }

        inline fn eagerScales(problem: Problem, row: u32, tile_n: u32, bases: ScaleBases, block: u32) Scales {
            return decodeScales(rawScales(problem, row, tile_n, bases, block), tile_n);
        }

        inline fn promoteEager(
            tiles: u32,
            consumer: u32,
            thread: u32,
            tile: Tile,
            problem: Problem,
            row: u32,
            slot: *Ring.Slot,
            accumulator: *wgmma.Accumulator(n),
            clocks: *Clocks,
        ) void {
            const block_scaled = comptime variant.blockScaled();
            if (comptime variant.shared_scales) {
                promoteShared(tiles, consumer, thread, tile, row, 0, problem.k_blocks, slot, accumulator, clocks);
                return;
            }
            const bases: ScaleBases = if (variant.hoisted) .init(problem, row, tile.n) else undefined;
            var partial: wgmma.Accumulator(n) = @splat(0);
            comptime std.debug.assert(variant.scales_ahead == 1 or variant.scales_ahead == 2);
            const two_ahead = comptime block_scaled and variant.scales_ahead == 2;
            var scales: Scales = if (block_scaled) eagerScales(problem, row, tile.n, bases, 0) else ones;
            var ahead: if (two_ahead) Raw else void = if (two_ahead)
                rawScales(problem, row, tile.n, bases, @min(1, problem.k_blocks - 1))
            else {};
            var ready = false;
            var block: u32 = 0;
            while (block < problem.k_blocks) : (block += 1) {
                issue(tiles, consumer, slot.*, &partial, clocks, &ready);
                const loading = Clocks.now();
                const next_block = @min(block + 1, problem.k_blocks - 1);
                const next = if (two_ahead) next: {
                    const later = rawScales(problem, row, tile.n, bases, @min(block + 2, problem.k_blocks - 1));
                    const decoded = decodeScales(ahead, tile.n);
                    ahead = later;
                    break :next decoded;
                } else if (block_scaled) eagerScales(problem, row, tile.n, bases, next_block) else scales;
                clocks.add(.scales, loading);
                const since = Clocks.now();
                wgmma.wait(0);
                clocks.add(.mma_wait, since);
                wgmma.fenceOperand(&partial);
                const releasing = Clocks.now();
                release(slot.*, thread);
                clocks.add(.release, releasing);
                const promoting = Clocks.now();
                add(accumulator, partial, scales);
                clocks.add(.promotion, promoting);
                scales = next;
                slot.* = slot.next();
            }
        }

        /// The consumer reads the scales of a K block from its stage after the full barrier and
        /// before the `wgmma` wait, so the reads precede the release of the stage.
        inline fn promoteShared(
            tiles: u32,
            consumer: u32,
            thread: u32,
            tile: Tile,
            row: u32,
            begin: u32,
            end: u32,
            slot: *Ring.Slot,
            accumulator: *wgmma.Accumulator(n),
            clocks: *Clocks,
        ) void {
            const atom_offset = row % 32 * 16 + row % 128 / 32 * 4;
            var partial: wgmma.Accumulator(n) = @splat(0);
            var ready = false;
            var block: u32 = begin;
            while (block < end) : (block += 1) {
                const scales = if (comptime variant.scales_first) first: {
                    waitFull(slot.*, clocks, ready);
                    const raw = sharedScales(tiles, slot.*, atom_offset, block);
                    issueFull(tiles, consumer, slot.*, &partial, clocks, &ready);
                    const decoding = Clocks.now();
                    const decoded = decodeScales(raw, tile.n);
                    clocks.add(.scales, decoding);
                    break :first decoded;
                } else after: {
                    clocks.at(block - begin);
                    clocks.mark(0);
                    if (variant.ordered and consumer == 1) orderBarrier();
                    issue(tiles, consumer, slot.*, &partial, clocks, &ready);
                    if (variant.ordered and consumer == 0) orderBarrier();
                    const loading = Clocks.now();
                    const decoded = decodeScales(sharedScales(tiles, slot.*, atom_offset, block), tile.n);
                    clocks.add(.scales, loading);
                    break :after decoded;
                };
                clocks.mark(4);
                const since = Clocks.now();
                wgmma.wait(0);
                clocks.add(.mma_wait, since);
                wgmma.fenceOperand(&partial);
                clocks.mark(5);
                const releasing = Clocks.now();
                release(slot.*, thread);
                if (variant.late_try_wait) {
                    const next = slot.next();
                    ready = ring.fullBarrier(next).tryWaitPhase(next.phase, null, null);
                }
                clocks.add(.release, releasing);
                clocks.mark(6);
                const promoting = Clocks.now();
                add(accumulator, partial, scales);
                clocks.add(.promotion, promoting);
                clocks.mark(7);
                slot.* = slot.next();
            }
        }

        /// Named barrier 2 counts the 256 threads of both consumers. PTX ISA 9.7.15.1 (`bar.sync`).
        inline fn orderBarrier() void {
            asm volatile ("bar.sync 2, 256;" ::: .{ .memory = true });
        }

        /// Writes the sums of K range `part` to the partials of `group`, one quarter of a cluster
        /// tile, and counts the arrival. The warpgroup that arrives last replaces its sums with the
        /// total of the partials of all ranges in range order and returns true.
        inline fn addRanges(
            problem: Problem,
            part: u32,
            group: u32,
            consumer: u32,
            thread: u32,
            accumulator: *wgmma.Accumulator(n),
        ) bool {
            const Quad = @Vector(4, f32);
            const quads = n / 8;
            const partials: [*]addrspace(.global) Quad = @ptrFromInt(problem.partials);
            const counters: [*]addrspace(.global) u32 = @ptrFromInt(problem.counters);
            const base = group * problem.splits * quads * 128 + thread;
            const own = base + part * quads * 128;
            inline for (0..quads) |quad| partials[own + quad * 128] = .{
                accumulator[4 * quad],
                accumulator[4 * quad + 1],
                accumulator[4 * quad + 2],
                accumulator[4 * quad + 3],
            };
            const barrier: nvptx.NamedBarrier = @fromBackingInt(@intCast(1 + consumer));
            nvptx.barrierSyncAligned(barrier, 128);
            var last = false;
            if (thread == 0) {
                nvptx.fence(.acq_rel, .gpu);
                last = nvptx.atom(.gpu, .acq_rel, .add, &counters[group], 1) == problem.splits - 1;
            }
            if (nvptx.barrierRedPopcAligned(barrier, 128, last) == 0) return false;
            accumulator.* = @splat(0);
            var range: u32 = 0;
            while (range < problem.splits) : (range += 1) {
                const sums = base + range * quads * 128;
                inline for (0..quads) |quad| {
                    const quarter = partials[sums + quad * 128];
                    inline for (0..4) |within| accumulator[4 * quad + within] += quarter[within];
                }
            }
            return true;
        }

        inline fn accumulate(
            tiles: u32,
            consumer: u32,
            thread: u32,
            problem: Problem,
            slot: *Ring.Slot,
            accumulator: *wgmma.Accumulator(n),
            clocks: *Clocks,
        ) void {
            var previous: Ring.Slot = slot.*;
            var block: u32 = 0;
            while (block < problem.k_blocks) : (block += 1) {
                const since = Clocks.now();
                waitPhase(ring.fullBarrier(slot.*), slot.phase);
                clocks.add(.stage_wait, since);
                const stage = tiles + slot.index * stage_bytes;
                const a: wgmma.Descriptor = .kMajor(stage + consumer * 64 * k_block, .bytes_128);
                const b: wgmma.Descriptor = .kMajor(stage + a_bytes, .bytes_128);
                wgmma.fence();
                wgmma.mmaKSteps(n, 4, .bytes_128, 64, n, accumulator, a, b, block != 0);
                wgmma.commit();
                const waited = Clocks.now();
                wgmma.wait(1);
                clocks.add(.mma_wait, waited);
                wgmma.fenceOperand(accumulator);
                if (block != 0) release(previous, thread);
                previous = slot.*;
                slot.* = slot.next();
            }
            wgmma.wait(0);
            wgmma.fenceOperand(accumulator);
            release(previous, thread);
        }

        /// The scale products of the rows `row` and `row + 8` for the `lo` and `hi` columns.
        inline fn gluScales(problem: Problem, row: u32, tile_n: u32, block: u32) [2][2]f32 {
            const s_b: [*]addrspace(.global) const Scale = @ptrFromInt(problem.s_b);
            const lo = decode(s_b[tile_n * problem.scale_blocks + block]);
            const hi = decode(s_b[(problem.half / 128 + tile_n) * problem.scale_blocks + block]);
            const rows: [2]f32 = .{
                decode(rowScale(problem, row, block)),
                decode(rowScale(problem, row + 8, block)),
            };
            return .{ .{ rows[0] * lo, rows[1] * lo }, .{ rows[0] * hi, rows[1] * hi } };
        }

        /// The `lo` and `hi` halves of a K block run one after the other through one temporary
        /// accumulator, which holds half as many registers as the sums.
        inline fn promoteGlu(
            tiles: u32,
            consumer: u32,
            thread: u32,
            tile: Tile,
            problem: Problem,
            row: u32,
            slot: *Ring.Slot,
            accumulator: *wgmma.Accumulator(n),
            clocks: *Clocks,
        ) void {
            const part_n = n / 2;
            var partial: wgmma.Accumulator(part_n) = @splat(0);
            var block: u32 = 0;
            while (block < problem.k_blocks) : (block += 1) {
                const since = Clocks.now();
                waitPhase(ring.fullBarrier(slot.*), slot.phase);
                clocks.add(.stage_wait, since);
                const stage = tiles + slot.index * stage_bytes;
                const a: wgmma.Descriptor = .kMajor(stage + consumer * 64 * k_block, .bytes_128);
                const scales: [2][2]f32 = if (comptime variant.blockScaled())
                    gluScales(problem, row, tile.n, block)
                else
                    @splat(.{ 1, 1 });
                inline for (0..2) |part| {
                    const b_start = stage + a_bytes + @as(u32, part) * part_n * k_block;
                    const b: wgmma.Descriptor = .kMajor(b_start, .bytes_128);
                    wgmma.fence();
                    wgmma.mmaKSteps(part_n, 4, .bytes_128, 64, part_n, &partial, a, b, false);
                    wgmma.commit();
                    const waited = Clocks.now();
                    wgmma.wait(0);
                    clocks.add(.mma_wait, waited);
                    wgmma.fenceOperand(&partial);
                    if (part == 1) release(slot.*, thread);
                    inline for (0..part_n / 2) |index| {
                        const target = comptime part * part_n / 2 + index;
                        accumulator[target] = if (comptime variant.blockScaled())
                            @mulAdd(f32, partial[index], scales[part][index / 2 % 2], accumulator[target])
                        else
                            accumulator[target] + partial[index];
                    }
                }
                slot.* = slot.next();
            }
        }

        inline fn gluRow(
            problem: Problem,
            lo: [2]f32,
            hi: [2]f32,
            fragment_row: u32,
            lo_column: u32,
        ) [2]f32 {
            var values: [2]f32 = undefined;
            inline for (0..2) |e| {
                const up = if (problem.gate_first) hi[e] else lo[e];
                const gate = if (problem.gate_first) lo[e] else hi[e];
                const product = if (variant.epilogue == .swiglu)
                    swiglu(up, gate, problem.limit)
                else
                    gegluTanh(up, gate);
                const inside = fragment_row < problem.m and lo_column + e < problem.hidden;
                values[e] = if (inside) product else 0;
            }
            return values;
        }

        inline fn storeZ(
            problem: Problem,
            fragment_row: u32,
            group_column: u32,
            quad: u32,
            lo: [4]u32,
            hi: [4]u32,
        ) void {
            if (problem.z == 0) return;
            const lo_chunk = transposeQuad(lo, quad);
            const hi_chunk = transposeQuad(hi, quad);
            if (fragment_row >= problem.m) return;
            const z: [*]addrspace(.global) u32 = @ptrFromInt(problem.z);
            const word = (@as(u64, fragment_row) * 2 * problem.hidden + group_column + 8 * quad) / 2;
            storeChunk(@intFromPtr(z + word), lo_chunk);
            storeChunk(@intFromPtr(z + word + problem.hidden / 2), hi_chunk);
        }

        /// Each word of `words` holds the codes of two steps. Work item `q` of a quad stores the
        /// codes of the steps `2q` and `2q + 1` of each group of 8 steps, 16 adjacent bytes of `h`.
        inline fn storeCodes(
            problem: Problem,
            fragment_row: u32,
            tile_column: u32,
            quad: u32,
            words: [n / 32]u32,
        ) void {
            const codes: [*]addrspace(.global) u32 = @ptrCast(problem.y);
            inline for (0..n / 128) |group| {
                const pairs = transposeQuad(words[4 * group ..][0..4].*, quad);
                const chunk: bf16.Chunk = .{
                    (pairs[0] & 0xffff) | (pairs[1] << 16),
                    (pairs[2] & 0xffff) | (pairs[3] << 16),
                    (pairs[0] >> 16) | (pairs[1] & 0xffff_0000),
                    (pairs[2] >> 16) | (pairs[3] & 0xffff_0000),
                };
                if (fragment_row < problem.m) {
                    const byte = @as(u64, fragment_row) * problem.half + tile_column + 64 * group + 16 * quad;
                    storeChunk(@intFromPtr(codes + byte / 4), chunk);
                }
            }
        }

        /// Writes `Z` when `problem.z` is set and the pack of `h`. The launcher runs this kernel only
        /// when `hidden` equals `half`, so the 16-byte stores of a row below `m` stay inside `Z` and
        /// `h`. The four work items of a row hold the 128 columns of a block128 block. The scale of a
        /// row from `m` to the end of its scale atom, or of a block after the last one up to the end
        /// of its scale atom, is 0, as in the padding of `glu_quant.zig`.
        inline fn storeGlu(
            problem: Problem,
            row: u32,
            column: u32,
            accumulator: wgmma.Accumulator(n),
        ) void {
            const steps = n / 16;
            const quad = column % 8 / 2;
            const tile_column = column - 2 * quad;
            const format: fp8.ScaleFormat = if (variant.recipe == .fp8_b128_e8m0) .e8m0 else .f32;
            const scales: [*]addrspace(.global) fp8.Scale(format) = @ptrFromInt(problem.h_scales);
            const block = tile_column / 128;
            const atoms_per_row = (problem.half + 511) / 512;
            const padding_block = block + quad;
            inline for (0..2) |half| {
                const fragment_row = row + 8 * @as(u32, half);
                var values: [2 * steps]f32 = undefined;
                inline for (0..steps / 4) |group| {
                    var lo_words: [4]u32 = undefined;
                    var hi_words: [4]u32 = undefined;
                    inline for (0..4) |within| {
                        const step = 4 * group + within;
                        var lo: [2]f32 = undefined;
                        var hi: [2]f32 = undefined;
                        inline for (0..2) |e| {
                            lo[e] = bf16.round(accumulator[4 * step + 2 * half + e]);
                            hi[e] = bf16.round(accumulator[4 * (step + steps) + 2 * half + e]);
                        }
                        lo_words[within] = bf16.pack(lo[0], lo[1]);
                        hi_words[within] = bf16.pack(hi[0], hi[1]);
                        const pair = gluRow(problem, lo, hi, fragment_row, column + 8 * @as(u32, step));
                        values[2 * step ..][0..2].* = pair;
                    }
                    storeZ(problem, fragment_row, tile_column + 32 * @as(u32, group), quad, lo_words, hi_words);
                }
                const amax = fp8.maxNanOverLanes(fp8.amaxOf(values), 4);
                const scaling: fp8_pack.Scaling(format) = .init(amax);
                var words: [steps / 2]u32 = undefined;
                inline for (&words, 0..) |*word, index| {
                    var scaled: [4]f32 = undefined;
                    inline for (&scaled, values[4 * index ..][0..4]) |*target, value|
                        target.* = math.mulF32(.nearest_even, .{}, value, scaling.factor);
                    word.* = @backingInt(nvptx.E4m3x4.fromF32(scaled[0], scaled[1], scaled[2], scaled[3]));
                }
                storeCodes(problem, fragment_row, tile_column, quad, words);
                if (fragment_row < (problem.m + 127) / 128 * 128) {
                    if (quad == 0) {
                        const scale = if (fragment_row < problem.m) scaling.scale else 0;
                        scales[atomIndex(fragment_row, block, atoms_per_row)] = scale;
                    } else if (block + 1 == problem.half / 128 and padding_block < 4 * atoms_per_row) {
                        scales[atomIndex(fragment_row, padding_block, atoms_per_row)] = 0;
                    }
                }
            }
        }

        /// The column scales of the `lo` and `hi` columns of 4 steps of a work item.
        inline fn loadGluScales(
            problem: Problem,
            column: u32,
            comptime first_step: u32,
        ) [4][2]@Vector(2, f32) {
            const s_b: [*]addrspace(.global) const @Vector(2, f32) = @ptrFromInt(problem.s_b);
            var loaded: [4][2]@Vector(2, f32) = undefined;
            inline for (&loaded, 0..) |*pairs, within| {
                const lo_column = column + 8 * (first_step + @as(u32, within));
                pairs.* = .{ s_b[lo_column / 2], s_b[(problem.half + lo_column) / 2] };
            }
            return loaded;
        }

        /// Writes `Z` when `problem.z` is set and the BF16 `h`, whose row scales need whole rows
        /// of `h`. The column scales of the next 4 steps load before the current steps store, as
        /// in `store`, and the values of `Z` are `(acc · row scale) · column scale`.
        inline fn storeGluRowwise(
            problem: Problem,
            row: u32,
            column: u32,
            accumulator: wgmma.Accumulator(n),
        ) void {
            const steps = n / 16;
            const quad = column % 8 / 2;
            const tile_column = column - 2 * quad;
            const s_a: [*]addrspace(.global) const f32 = @ptrFromInt(problem.s_a);
            const row_scales: [2]f32 = .{
                s_a[@min(row, problem.m - 1)],
                s_a[@min(row + 8, problem.m - 1)],
            };
            var next = loadGluScales(problem, column, 0);
            inline for (0..steps / 4) |chunk| {
                const current = next;
                if (chunk + 1 < steps / 4) next = loadGluScales(problem, column, (chunk + 1) * 4);
                const group_column = tile_column + 32 * @as(u32, chunk);
                inline for (0..2) |half| {
                    const fragment_row = row + 8 * @as(u32, half);
                    var lo_words: [4]u32 = undefined;
                    var hi_words: [4]u32 = undefined;
                    var h_words: [4]u32 = undefined;
                    inline for (0..4) |within| {
                        const step = chunk * 4 + within;
                        var lo: [2]f32 = undefined;
                        var hi: [2]f32 = undefined;
                        inline for (0..2) |e| {
                            const x = accumulator[4 * step + 2 * half + e];
                            const y = accumulator[4 * (step + steps) + 2 * half + e];
                            lo[e] = bf16.round(x * row_scales[half] * current[within][0][e]);
                            hi[e] = bf16.round(y * row_scales[half] * current[within][1][e]);
                        }
                        lo_words[within] = bf16.pack(lo[0], lo[1]);
                        hi_words[within] = bf16.pack(hi[0], hi[1]);
                        const values = gluRow(problem, lo, hi, fragment_row, column + 8 * @as(u32, step));
                        h_words[within] = bf16.pack(values[0], values[1]);
                    }
                    storeZ(problem, fragment_row, group_column, quad, lo_words, hi_words);
                    const h = transposeQuad(h_words, quad);
                    if (fragment_row < problem.m) {
                        const word = (@as(u64, fragment_row) * problem.hidden + group_column + 8 * quad) / 2;
                        storeChunk(@intFromPtr(problem.y + word), h);
                    }
                }
            }
        }

        const chunk_steps = @min(n / 8, 8);

        /// The column scales and bias terms of `chunk_steps` 8-column steps of a thread.
        const Chunk = struct { scales: [chunk_steps][2]f32, terms: [chunk_steps]u32 };

        /// The indices are clamped to the last column so that every load issues without a branch.
        inline fn loadChunk(problem: Problem, column: u32, comptime first_step: u32) Chunk {
            const last_column = problem.n - 1;
            var loaded: Chunk = .{ .scales = @splat(.{ 1, 1 }), .terms = @splat(0) };
            inline for (0..chunk_steps) |step| {
                const fragment_column = column + 8 * (first_step + step);
                if (!variant.blockScaled()) {
                    const s_b: [*]addrspace(.global) const f32 = @ptrFromInt(problem.s_b);
                    loaded.scales[step] = .{
                        s_b[@min(fragment_column, last_column)],
                        s_b[@min(fragment_column + 1, last_column)],
                    };
                }
                if (variant.epilogue == .bias)
                    loaded.terms[step] = problem.bias[@min(fragment_column, last_column) / 2];
            }
            return loaded;
        }

        /// Work item `q` of a quad receives word `q` of each work item of the quad, in their order.
        inline fn transposeQuad(words: [4]u32, quad: u32) bf16.Chunk {
            const upper = quad >= 2;
            var pairs = words;
            inline for (0..2) |word| {
                const sent = if (upper) words[word] else words[word + 2];
                const got: u32 = @bitCast(warp.shuffleXor(@bitCast(sent), 2));
                if (upper) pairs[word] = got else pairs[word + 2] = got;
            }
            const odd = quad % 2 == 1;
            var result = pairs;
            inline for (0..2) |pair| {
                const sent = if (odd) pairs[2 * pair] else pairs[2 * pair + 1];
                const got: u32 = @bitCast(warp.shuffleXor(@bitCast(sent), 1));
                if (odd) result[2 * pair] = got else result[2 * pair + 1] = got;
            }
            return result;
        }

        inline fn storeChunk(address: u64, chunk: bf16.Chunk) void {
            asm volatile ("st.global.v4.u32 [%[p]], {%[x], %[y], %[z], %[w]};"
                :
                : [p] "l" (address),
                  [x] "r" (chunk[0]),
                  [y] "r" (chunk[1]),
                  [z] "r" (chunk[2]),
                  [w] "r" (chunk[3]),
                : .{ .memory = true });
        }

        /// The accumulator fragment layout is that of PTX ISA 9.7.17.5.1.1.3. The next chunk of
        /// scales and bias terms loads before the current chunk stores. If the width of `Y` is a
        /// multiple of 8, the four work items of a row transpose four whole 8-column steps, and
        /// each work item stores the 16 bytes of one step.
        inline fn store(
            problem: Problem,
            row: u32,
            column: u32,
            accumulator: wgmma.Accumulator(n),
        ) void {
            comptime std.debug.assert(chunk_steps % 4 == 0);
            const chunks = n / 8 / chunk_steps;
            const quad = column % 8 / 2;
            const tile_column = column - 2 * quad;
            const aligned = problem.n % 8 == 0;
            var row_scales: [2]f32 = .{ 1, 1 };
            if (!variant.blockScaled()) {
                const s_a: [*]addrspace(.global) const f32 = @ptrFromInt(problem.s_a);
                row_scales = .{ s_a[@min(row, problem.m - 1)], s_a[@min(row + 8, problem.m - 1)] };
            }
            var next = loadChunk(problem, column, 0);
            inline for (0..chunks) |index| {
                const current = next;
                if (index + 1 < chunks) next = loadChunk(problem, column, (index + 1) * chunk_steps);
                inline for (0..2) |half| {
                    const fragment_row = row + 8 * half;
                    var words: [chunk_steps]u32 = undefined;
                    inline for (0..chunk_steps) |within| {
                        const step = index * chunk_steps + within;
                        var x0 = accumulator[4 * step + 2 * half];
                        var x1 = accumulator[4 * step + 2 * half + 1];
                        if (!variant.blockScaled()) {
                            x0 = x0 * row_scales[half] * current.scales[within][0];
                            x1 = x1 * row_scales[half] * current.scales[within][1];
                        }
                        if (variant.epilogue == .bias) {
                            const terms = bf16.unpack(current.terms[within]);
                            x0 += terms[0];
                            x1 += terms[1];
                        }
                        words[within] = bf16.pack(x0, x1);
                    }
                    const row_start = @as(u64, fragment_row) * problem.n + tile_column;
                    inline for (0..chunk_steps / 4) |group| {
                        const step = index * chunk_steps + 4 * group;
                        if (aligned and tile_column + 8 * (step + 4) <= problem.n) {
                            const chunk = transposeQuad(words[4 * group ..][0..4].*, quad);
                            const word = (row_start + 8 * (step + quad)) / 2;
                            if (fragment_row < problem.m) storeChunk(@intFromPtr(problem.y + word), chunk);
                        } else if (fragment_row < problem.m) {
                            inline for (0..4) |within| {
                                if (column + 8 * (step + within) < problem.n)
                                    problem.y[(row_start + 2 * quad) / 2 + 4 * (step + within)] = words[4 * group + within];
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
            const entry = if (variant.glu())
                &Kernel(variant).kernelGlu
            else if (variant.split_tail)
                &Kernel(variant).kernelSplit
            else
                &Kernel(variant).kernel;
            @export(entry, .{ .name = configurations.kernelName(variant) });
        }
    }
}
