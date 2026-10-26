const std = @import("std");
const configurations = @import("configurations.zig");

pub const k_block = 128;
pub const mmas_per_group = 4;

pub const Rate = struct {
    n: u32,
    warpgroups: u32,

    pub fn threads(rate: Rate) u32 {
        return 128 * rate.warpgroups;
    }

    pub fn operandBytes(rate: Rate) u32 {
        return (64 * rate.warpgroups + rate.n) * k_block;
    }

    /// The kernel aligns its operands to 1024 bytes, the repeat of the 128-byte swizzle pattern.
    pub fn sharedBytes(rate: Rate) u32 {
        return rate.operandBytes() + 1024;
    }
};

pub const rates = [_]Rate{
    .{ .n = 256, .warpgroups = 2 },
    .{ .n = 256, .warpgroups = 3 },
    .{ .n = 192, .warpgroups = 2 },
    .{ .n = 128, .warpgroups = 2 },
};

pub fn kernelName(comptime rate: Rate) []const u8 {
    return std.fmt.comptimePrint("mma_rate_sm90_n{d}_w{d}", .{ rate.n, rate.warpgroups });
}

const kernels = kernels: {
    var list: [rates.len]configurations.Kernel = undefined;
    for (rates, &list) |rate, *kernel| kernel.* = .{
        .name = kernelName(rate),
        .launch = .{
            .threads = rate.threads(),
            .shared_bytes = rate.sharedBytes(),
            .tile_m = 64 * rate.warpgroups,
            .tile_n = rate.n,
            .tile_k = k_block,
        },
    };
    const final = list;
    break :kernels &final;
};

pub const modules: []const configurations.Module = &.{.{
    .name = "mma_rate_sm90",
    .source = "mma_rate_sm90",
    .arch = 90,
    .targets = &.{"sm_90a"},
    .kernels = kernels,
}};
