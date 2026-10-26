//! The kernels copy FP8 operands to shared memory once and then issue a chain of `wgmma` on them
//! in each warpgroup, with one commit group of `mmas_per_group` instructions in flight.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const configurations = @import("mma_rate_sm90_configurations.zig");
const wgmma = @import("wgmma.zig");

const Rate = configurations.Rate;
const k_block = configurations.k_block;

extern var shared_storage: [0]u8 align(1024) addrspace(.shared);

fn Kernel(comptime rate: Rate) type {
    return struct {
        const n = rate.n;
        const threads = rate.threads();
        const Chunk = @Vector(4, u32);

        fn kernel(
            operands: [*]addrspace(.global) const Chunk,
            out: [*]addrspace(.global) f32,
            iterations: u32,
        ) callconv(.{ .nvptx_kernel = .{ .required_work_group_size = .{ .x = threads } } }) void {
            const base = std.mem.alignForward(u32, @intCast(@intFromPtr(&shared_storage)), 1024);
            const chunks: [*]addrspace(.shared) Chunk = @ptrFromInt(base);
            const tid = @workItemId(0);
            var index = tid;
            while (index < rate.operandBytes() / @sizeOf(Chunk)) : (index += threads) {
                chunks[index] = operands[index];
            }
            nvptx.fenceProxyAsync(.shared);
            nvptx.barrierSyncAligned(.work_group, null);
            const warpgroup = tid / 128;
            const a: wgmma.Descriptor = .kMajor(base + warpgroup * 64 * k_block, .bytes_128);
            const b: wgmma.Descriptor = .kMajor(base + 64 * rate.warpgroups * k_block, .bytes_128);
            var accumulator: wgmma.Accumulator(n) = @splat(0);
            wgmma.fence();
            var iteration: u32 = 0;
            while (iteration < iterations) : (iteration += 1) {
                const steps = configurations.mmas_per_group;
                wgmma.mmaKSteps(n, steps, .bytes_128, 64, n, &accumulator, a, b, true);
                wgmma.commit();
                wgmma.wait(1);
            }
            wgmma.wait(0);
            wgmma.fenceOperand(&accumulator);
            var total: f32 = 0;
            inline for (0..n / 2) |element| total += accumulator[element];
            out[@workGroupId(0) * threads + tid] = total;
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (configurations.rates) |rate| {
            @export(&Kernel(rate).kernel, .{ .name = configurations.kernelName(rate) });
        }
    }
}
