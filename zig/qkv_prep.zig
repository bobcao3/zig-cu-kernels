//! The kernels turn `Z = [q | k | v]` `[M, 3·H·D]` into `Q` `[M, H·D]` and `KV = [k | v]` by
//! RMSNorm and interleaved-pair RoPE of each q and k head. `D / 8` lanes own a head, 8 values each.
//! A lane sums its 8 squares from the lowest column up, and a butterfly over the lane masks 1, 2,
//! 4 and so on adds the lane sums, which gives every lane of the head the same sum.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const math = nvptx.math;
const bf16 = @import("bf16.zig");
const configurations = @import("qkv_prep_configurations.zig");

const Words = bf16.Chunk;
const chunk_values = 8;
const threads = configurations.threads;
const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = threads },
} };

inline fn shuffleXor(x: f32, mask: u32) f32 {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return nvptx.shflSync(.bfly, f32, x, mask, clamp, .all);
}

inline fn sumOfSquares(comptime lanes: u32, x: [chunk_values]f32) f32 {
    var sum: f32 = 0;
    inline for (x) |value| {
        const square = bf16.round(math.mulF32(.nearest_even, .{}, value, value));
        sum = math.addF32(.nearest_even, .{}, sum, square);
    }
    comptime var mask = 1;
    inline while (mask < lanes) : (mask *= 2) {
        sum = math.addF32(.nearest_even, .{}, sum, shuffleXor(sum, mask));
    }
    return sum;
}

fn QkvPrep(comptime head_dim: u32) type {
    return struct {
        const lanes = head_dim / chunk_values;
        const inverse_head_dim: f32 = 1.0 / @as(f32, @floatFromInt(head_dim));

        /// Row `r` of `rope` holds the cosine and sine of pair `p` at `2p` and `2p + 1`.
        fn kernel(
            z: [*]addrspace(.global) const Words,
            q: [*]addrspace(.global) Words,
            kv: [*]addrspace(.global) Words,
            qk_scales: [*]addrspace(.global) const f32,
            scales_bf16: u32,
            rope: [*]addrspace(.global) const @Vector(2, f32),
            rows: u32,
            attn_dim: u32,
            rope_width: u32,
            eps: f32,
        ) callconv(kernel_convention) void {
            const chunks_per_plane = attn_dim / chunk_values;
            const i = @workGroupId(0) * threads + @workItemId(0);
            const row = i / (3 * chunks_per_plane);
            if (row >= rows) return;
            const chunk = i % (3 * chunks_per_plane);
            const plane = chunk / chunks_per_plane;
            const column = chunk % chunks_per_plane * chunk_values;
            const words = z[i];
            const row_words = @as(u64, row) * (attn_dim / chunk_values);
            if (plane == 2) {
                kv[2 * row_words + chunks_per_plane + column / chunk_values] = words;
                return;
            }
            const x = bf16.unpackChunk(words);
            const sum = sumOfSquares(lanes, x);
            const mean = math.mulF32(.nearest_even, .{}, sum, inverse_head_dim);
            const root = math.rsqrtApproxF32(true, math.addF32(.nearest_even, .{}, mean, eps));
            const head = column / head_dim;
            const bf16_scales: [*]addrspace(.global) const u16 = @ptrCast(qk_scales);
            const scale = if (scales_bf16 != 0) bf16.toF32(bf16_scales[head]) else qk_scales[head];
            const coefficient = bf16.round(math.mulF32(.nearest_even, .{}, scale, root));
            var y: [chunk_values]f32 = undefined;
            inline for (&y, x) |*value, input| {
                value.* = bf16.round(math.mulF32(.nearest_even, .{}, input, coefficient));
            }
            const first = column % head_dim;
            inline for (0..chunk_values / 2) |p| {
                if (first + 2 * p + 1 < rope_width) {
                    const turn = rope[(@as(u64, row) * rope_width + first) / 2 + p];
                    const y0 = y[2 * p];
                    const y1 = y[2 * p + 1];
                    y[2 * p] = bf16.round(math.subF32(
                        .nearest_even,
                        .{},
                        math.mulF32(.nearest_even, .{}, y0, turn[0]),
                        math.mulF32(.nearest_even, .{}, y1, turn[1]),
                    ));
                    y[2 * p + 1] = bf16.round(math.addF32(
                        .nearest_even,
                        .{},
                        math.mulF32(.nearest_even, .{}, y0, turn[1]),
                        math.mulF32(.nearest_even, .{}, y1, turn[0]),
                    ));
                }
            }
            const out = bf16.packChunk(y);
            if (plane == 0) {
                q[row_words + column / chunk_values] = out;
            } else {
                kv[2 * row_words + column / chunk_values] = out;
            }
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (configurations.head_dims) |head_dim| {
            @export(&QkvPrep(head_dim).kernel, .{ .name = configurations.kernelName(head_dim) });
        }
    }
}
