//! The kernels issue independent FP8 `mma.sync` instructions of compute capability 12.x on register
//! operands that each warp loads from `operands` and touch no memory in the loop.

const std = @import("std");
const builtin = @import("builtin");
const configurations = @import("mma_rate_sm120_configurations.zig");
const mma = @import("mma_sm120.zig");

const Kind = configurations.Kind;
const chains = configurations.chains;
const words_per_lane = configurations.words_per_lane;

fn Peak(comptime kind: Kind) type {
    return struct {
        fn kernel(
            out: [*]addrspace(.global) f32,
            operands: [*]addrspace(.global) const u32,
            iterations: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = configurations.threads },
        } }) void {
            const tid = @workItemId(0);
            const lane_words = operands + (tid & 31) * words_per_lane;
            var a: [chains]mma.A = undefined;
            var b: [chains]mma.B = undefined;
            inline for (&a, &b, 0..) |*a_chain, *b_chain, chain| {
                inline for (0..4) |word| a_chain[word] = lane_words[chain * 6 + word];
                inline for (0..2) |word| b_chain[word] = lane_words[chain * 6 + 4 + word];
            }
            const scale_a = lane_words[chains * 6];
            const scale_b = lane_words[chains * 6 + 1];
            var sums: [chains]mma.Accumulator = @splat(@splat(0));
            var iteration: u32 = 0;
            while (iteration < iterations) : (iteration += 1) {
                inline for (&sums, a, b) |*sum, a_chain, b_chain| {
                    sum.* = switch (kind) {
                        .f8f6f4 => mma.mma(a_chain, b_chain, sum.*),
                        .mxf8f6f4 => mma.mmaBlockScaled(
                            .{},
                            a_chain,
                            b_chain,
                            sum.*,
                            scale_a,
                            scale_b,
                        ),
                    };
                }
            }
            var total: f32 = 0;
            inline for (sums) |sum| total += sum[0] + sum[1] + sum[2] + sum[3];
            if (std.math.isInf(total)) out[tid] = total;
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Kind)) |kind| {
            @export(&Peak(kind).kernel, .{ .name = configurations.kernelName(kind) });
        }
    }
}
