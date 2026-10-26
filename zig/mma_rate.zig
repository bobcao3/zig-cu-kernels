//! The kernels issue independent `mma.sync` instructions on register operands with pseudo-random
//! signs and significands and touch no memory in the loop.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const config = @import("config");
const configurations = @import("mma_rate_configurations.zig");

const Element = configurations.Element;
const threads = configurations.threads;
const chains = configurations.chains;

fn Peak(comptime element: Element) type {
    return struct {
        const options: nvptx.matrix.MmaOptions = switch (element) {
            .bf16 => .{ .shape = .m16n8k16, .accumulator = .f32, .element = .bf16 },
            .e4m3 => .{ .shape = .m16n8k32, .accumulator = .f32, .element = .e4m3 },
        };
        const sign_and_significand: u32 = switch (element) {
            .bf16 => 0x807F_807F,
            .e4m3 => 0x8787_8787,
        };
        const exponent_of_one: u32 = switch (element) {
            .bf16 => 0x3F80_3F80,
            .e4m3 => 0x3838_3838,
        };

        fn mix(value: u32) u32 {
            var x = value *% 0x9E37_79B9;
            x ^= x >> 16;
            x *%= 0x85EB_CA6B;
            x ^= x >> 13;
            x *%= 0xC2B2_AE35;
            return x ^ (x >> 16);
        }

        fn operand(seed: u32, tid: u32, index: u32) u32 {
            return mix(seed ^ mix(tid * 64 + index)) & sign_and_significand | exponent_of_one;
        }

        fn kernel(
            out: [*]addrspace(.global) f32,
            iterations: u32,
            seed: u32,
        ) callconv(.{ .nvptx_kernel = .{ .required_work_group_size = .{ .x = threads } } }) void {
            const tid = @workItemId(0);
            var a: [chains]@Vector(4, u32) = undefined;
            var b: [chains]@Vector(2, u32) = undefined;
            inline for (&a, &b, 0..) |*a_chain, *b_chain, chain| {
                inline for (0..4) |word| a_chain[word] = operand(seed, tid, chain * 6 + word);
                inline for (0..2) |word| b_chain[word] = operand(seed, tid, chain * 6 + 4 + word);
            }
            var sums: [chains]@Vector(4, f32) = @splat(@splat(0));
            var iteration: u32 = 0;
            while (iteration < iterations) : (iteration += 1) {
                inline for (&sums, a, b) |*sum, a_chain, b_chain| {
                    sum.* = nvptx.mmaSync(options, a_chain, b_chain, sum.*);
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
        const element: Element = @fromBackingInt(@intCast(config.variant));
        @export(&Peak(element).kernel, .{ .name = configurations.kernelName(element) });
    }
}
