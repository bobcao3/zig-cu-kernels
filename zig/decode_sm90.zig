//! `Y = A Bᵀ` of FP8 packs for at most 128 rows of `A` on SM90. The kernel computes `Yᵀ = B Aᵀ`
//! with `wgmma`: 128 weight rows of `B` form the M operand and the token tile of `A` forms the N
//! operand. A unit is one 128-deep K block of one 128-row weight tile, the units of a tile are
//! consecutive, and CTA `c` of `C` takes the units from `c·U/C` to `(c+1)·U/C`. A CTA that holds
//! all units of a tile stores BF16 `Y`; for any other tile it writes its FP32 sums to slot `2c`
//! (the first tile of its range) or `2c + 1` (the last tile) of the workspace, and a second
//! kernel adds the slots of the tile.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("decode_sm90_configurations.zig");
const bf16 = @import("bf16.zig");
const bulk = @import("bulk.zig");
const cluster = @import("cluster.zig");
const pipeline = @import("pipeline.zig");
const tma = @import("tma.zig");
const wgmma = @import("wgmma.zig");

const Variant = configurations.Variant;
const k_block = configurations.k_block;
const weight_rows = configurations.weight_rows;

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

const Problem = struct {
    y: [*]addrspace(.global) u16,
    workspace: [*]addrspace(.global) f32,
    s_a: u64,
    s_w: u64,
    bias: [*]addrspace(.global) const u16,
    m: u32,
    n: u32,
    k_blocks: u32,
    scale_blocks: u32,
    first_unit: u32,
    end_unit: u32,
};

fn Kernel(comptime variant: Variant) type {
    return struct {
        const tokens = variant.tokens;
        const Ring = pipeline.Pipeline(variant.stages(), .cta);
        const weight_bytes = weight_rows * k_block;
        const stage_bytes = variant.stageBytes();
        const scale_bytes = variant.scaleBytes();
        const scales_offset = variant.stages() * stage_bytes;
        const atom_bytes = configurations.scale_atom_rows * 4 * @sizeOf(Scale);
        const halves = if (tokens == 128) 2 else 1;
        const half_tokens = tokens / halves;
        const Half = wgmma.Accumulator(half_tokens);
        const Scale = if (variant.recipe == .fp8_b128_e8m0) u8 else f32;
        const Sums = wgmma.Accumulator(tokens);
        const steps = tokens / 8;
        const options: std.lang.CallingConvention.NvptxKernelOptions = .{
            .required_work_group_size = .{ .x = configurations.threads },
        };

        var ring: Ring addrspace(.shared) = undefined;

        fn kernel(
            w_map: tma.TensorMap,
            a_map: tma.TensorMap,
            y: [*]addrspace(.global) u16,
            workspace: [*]addrspace(.global) f32,
            s_a: u64,
            s_w: u64,
            bias: [*]addrspace(.global) const u16,
            m: u32,
            n: u32,
            tiles: u32,
            k_blocks: u32,
            scale_blocks: u32,
        ) callconv(.{ .nvptx_kernel = options }) void {
            const units = tiles * k_blocks;
            const cta: u32 = @workGroupId(0);
            const ctas = nvptx.nctaid(0);
            const problem: Problem = .{
                .y = y,
                .workspace = workspace,
                .s_a = s_a,
                .s_w = s_w,
                .bias = bias,
                .m = m,
                .n = n,
                .k_blocks = k_blocks,
                .scale_blocks = scale_blocks,
                .first_unit = cta * units / ctas,
                .end_unit = (cta + 1) * units / ctas,
            };
            const base: u32 = @intCast(@intFromPtr(&shared_storage));
            const stages = std.mem.alignForward(u32, base, 1024);
            const tid = @workItemId(0);
            if (tid == 0) ring.init(1, 8);
            nvptx.barrierSyncAligned(.work_group, null);
            cluster.griddepcontrol(.wait);
            cluster.griddepcontrol(.launch_dependents);
            if (tid == 0) produce(stages, &w_map, &a_map, problem);
            if (tid >= 128) consume(stages, tid / 128 - 1, tid % 128, problem);
        }

        fn produce(
            stages: u32,
            w_map: *const tma.TensorMap,
            a_map: *const tma.TensorMap,
            problem: Problem,
        ) void {
            const evict_first = tma.createPolicy(.evict_first);
            const evict_last = tma.createPolicy(.evict_last);
            var slot: Ring.Slot = .{};
            var unit = problem.first_unit;
            while (unit < problem.end_unit) : (unit += 1) {
                const tile = unit / problem.k_blocks;
                const block = unit % problem.k_blocks;
                waitPhase(ring.emptyBarrier(slot), ~slot.phase);
                ring.producerExpect(slot, stage_bytes + scale_bytes);
                const k: i32 = @intCast(block * k_block);
                const stage = stages + slot.index * stage_bytes;
                const full = ring.fullBarrier(slot);
                const w_row: i32 = @intCast(tile * weight_rows);
                tma.load(2, stage, w_map, .{ k, w_row }, full, evict_first);
                tma.load(2, stage + weight_bytes, a_map, .{ k, 0 }, full, evict_last);
                if (comptime variant.blockScaled()) {
                    const atom: [*]addrspace(.global) const u8 = @ptrFromInt(problem.s_a);
                    const destination = stages + scales_offset + slot.index * scale_bytes;
                    const source = atom + @as(u64, block / 4) * atom_bytes;
                    bulk.load(@ptrFromInt(destination), source, scale_bytes, full);
                }
                slot = slot.next();
            }
        }

        inline fn weightScale(problem: Problem, tile: u32, block: u32) f32 {
            const s_w: [*]addrspace(.global) const Scale = @ptrFromInt(problem.s_w);
            const raw = s_w[tile * problem.scale_blocks + block];
            return if (Scale == u8) e8m0(raw) else raw;
        }

        /// The products of the token scales and the weight scale for the tokens of one half of a
        /// work item, in the order of the accumulator fragment. `scales` points at the scale of
        /// token `2·(lane % 4)` in the copy of the scale atom of the K block.
        inline fn blockScales(
            scales: [*]addrspace(.shared) const Scale,
            weight: f32,
            comptime half: u32,
        ) [half_tokens / 8][2]f32 {
            var products: [half_tokens / 8][2]f32 = undefined;
            inline for (0..half_tokens / 8) |local| {
                const step = half * (half_tokens / 8) + local;
                inline for (0..2) |e| {
                    const raw = scales[comptime step % 4 * 128 + e * 16 + step / 4 * 4];
                    products[local][e] = (if (Scale == u8) e8m0(raw) else raw) * weight;
                }
            }
            return products;
        }

        fn consume(stages: u32, consumer: u32, thread: u32, problem: Problem) void {
            if (problem.first_unit >= problem.end_unit) return;
            const k_blocks = problem.k_blocks;
            const first_tile = problem.first_unit / k_blocks;
            const last_tile = (problem.end_unit - 1) / k_blocks;
            var slot: Ring.Slot = .{};
            var tile = first_tile;
            while (tile <= last_tile) : (tile += 1) {
                const start = @max(problem.first_unit, tile * k_blocks) - tile * k_blocks;
                const end = @min(problem.end_unit, (tile + 1) * k_blocks) - tile * k_blocks;
                var sums: Sums = @splat(0);
                var partials: [halves]Half = @splat(@splat(0));
                var block = start;
                while (block < end) : (block += 1) {
                    waitPhase(ring.fullBarrier(slot), slot.phase);
                    const stage = stages + slot.index * stage_bytes;
                    const a: wgmma.Descriptor = .kMajor(stage + consumer * 64 * k_block, .bytes_128);
                    wgmma.fence();
                    inline for (0..halves) |half| {
                        const b_start = stage + weight_bytes + @as(u32, half) * half_tokens * k_block;
                        const b: wgmma.Descriptor = .kMajor(b_start, .bytes_128);
                        const partial = &partials[half];
                        wgmma.mmaKSteps(half_tokens, 4, .bytes_128, 64, half_tokens, partial, a, b, false);
                        wgmma.commit();
                    }
                    const atom: [*]addrspace(.shared) const Scale =
                        @ptrFromInt(stages + scales_offset + slot.index * scale_bytes);
                    const lane_scales = atom + 32 * (thread % 4) + block % 4;
                    const weight = if (comptime variant.blockScaled())
                        weightScale(problem, tile, block)
                    else
                        1;
                    inline for (0..halves) |half| {
                        const scales = if (comptime variant.blockScaled())
                            blockScales(lane_scales, weight, half)
                        else
                            undefined;
                        wgmma.wait(halves - 1 - half);
                        wgmma.fenceOperand(&partials[half]);
                        if (half == halves - 1 and thread % 32 == 0) ring.consumerRelease(slot);
                        inline for (0..half_tokens / 2) |within| {
                            const index = half * (half_tokens / 2) + within;
                            const product = partials[half][within];
                            sums[index] = if (comptime variant.blockScaled())
                                @mulAdd(f32, product, scales[within / 4][within % 2], sums[index])
                            else
                                sums[index] + product;
                        }
                    }
                    slot = slot.next();
                }
                const row = tile * weight_rows + consumer * 64 + thread / 32 * 16 + thread % 32 / 4;
                const token = 2 * (thread % 4);
                if (start == 0 and end == k_blocks) {
                    store(problem, row, token, sums);
                } else {
                    const cta = @workGroupId(0);
                    const index = 2 * cta + @intFromBool(tile != first_tile);
                    storePartial(problem, index, row - tile * weight_rows, token, sums);
                }
            }
        }

        /// The accumulator fragment layout is that of PTX ISA 9.7.17.5.1.1.3: element
        /// `4·step + 2·half + e` belongs to weight row `row + 8·half` and token `token + 8·step + e`.
        inline fn store(problem: Problem, row: u32, token: u32, sums: Sums) void {
            inline for (0..2) |half| {
                const column = row + 8 * @as(u32, half);
                if (column < problem.n) {
                    var weight: f32 = 1;
                    if (!variant.blockScaled())
                        weight = @as([*]addrspace(.global) const f32, @ptrFromInt(problem.s_w))[column];
                    const term = if (variant.epilogue == .bias) bf16.toF32(problem.bias[column]) else 0;
                    inline for (0..steps) |step| {
                        inline for (0..2) |e| {
                            const t = token + 8 * @as(u32, step) + @as(u32, e);
                            if (t < problem.m) {
                                var x = sums[4 * step + 2 * half + e];
                                if (!variant.blockScaled()) {
                                    const s_a: [*]addrspace(.global) const f32 = @ptrFromInt(problem.s_a);
                                    x = x * s_a[t] * weight;
                                }
                                if (variant.epilogue == .bias) x += term;
                                problem.y[@as(u64, t) * problem.n + column] = bf16.fromF32(x);
                            }
                        }
                    }
                }
            }
        }

        /// Slot `index` of the workspace holds `tokens` rows of 128 FP32 sums, one per weight row
        /// of the tile, and the kernel writes the rows of the first `m` tokens.
        inline fn storePartial(problem: Problem, index: u32, local_row: u32, token: u32, sums: Sums) void {
            const slot = problem.workspace + @as(u64, index) * tokens * weight_rows;
            inline for (0..2) |half| {
                inline for (0..steps) |step| {
                    inline for (0..2) |e| {
                        const t = token + 8 * @as(u32, step) + @as(u32, e);
                        if (t < problem.m)
                            slot[t * weight_rows + local_row + 8 * half] = sums[4 * step + 2 * half + e];
                    }
                }
            }
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (configurations.variants) |variant| {
            @export(&Kernel(variant).kernel, .{ .name = configurations.kernelName(variant) });
        }
    }
}
