const std = @import("std");
const configurations = @import("configurations.zig");

pub const k_block = 128;
pub const stages = 2;
pub const wait_limit_ns = 1_000_000_000;

pub const Check = struct {
    n: u32,
    consumers: u32,
    cluster_ctas: u32,
    registers: ?Registers = null,

    pub fn threads(check: Check) u32 {
        return 128 * (check.consumers + 1);
    }

    pub fn stageBytes(check: Check) u32 {
        return (64 * check.consumers + check.n) * k_block;
    }

    /// The kernel aligns its tiles to 1024 bytes, the repeat of the 128-byte swizzle pattern.
    pub fn sharedBytes(check: Check) u32 {
        return stages * check.stageBytes() + 1024;
    }
};

/// The producer warpgroup gives registers to the consumer warpgroups, as in a GEMM whose consumers
/// hold the accumulators.
pub const Registers = struct { producer: u32, consumer: u32 };

const split: Registers = .{ .producer = 40, .consumer = 232 };

pub const checks = [_]Check{
    .{ .n = 128, .consumers = 1, .cluster_ctas = 1 },
    .{ .n = 256, .consumers = 1, .cluster_ctas = 1, .registers = split },
    .{ .n = 256, .consumers = 2, .cluster_ctas = 1, .registers = split },
    .{ .n = 256, .consumers = 2, .cluster_ctas = 2, .registers = split },
    .{ .n = 128, .consumers = 2, .cluster_ctas = 2, .registers = split },
};

pub fn kernelName(comptime check: Check) []const u8 {
    return std.fmt.comptimePrint(
        "mma_check_sm90_n{d}_c{d}_x{d}",
        .{ check.n, check.consumers, check.cluster_ctas },
    );
}

const kernels = kernels: {
    var list: [checks.len]configurations.Kernel = undefined;
    for (checks, &list) |check, *kernel| kernel.* = .{
        .name = kernelName(check),
        .launch = .{
            .threads = check.threads(),
            .shared_bytes = check.sharedBytes(),
            .tile_m = 64 * check.consumers,
            .tile_n = check.n,
            .tile_k = k_block,
            .stages = stages,
        },
    };
    const final = list;
    break :kernels &final;
};

pub const modules: []const configurations.Module = &.{.{
    .name = "mma_check_sm90",
    .source = "mma_check_sm90",
    .arch = 90,
    .targets = &.{"sm_90a"},
    .kernels = kernels,
}};
