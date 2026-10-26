const configurations = @import("configurations.zig");

pub const Kind = enum { f8f6f4, mxf8f6f4 };

pub const threads = 256;
pub const chains = 8;
pub const words_per_lane = chains * 6 + 2;

pub fn kernelName(comptime kind: Kind) []const u8 {
    return "mma_rate_sm120_" ++ @tagName(kind);
}

pub const modules: []const configurations.Module = &.{.{
    .name = "mma_rate_sm120",
    .source = "mma_rate_sm120",
    .arch = 120,
    .targets = &.{"sm_120f"},
    .kernels = &.{
        .{ .name = kernelName(.f8f6f4), .launch = .{ .threads = threads } },
        .{ .name = kernelName(.mxf8f6f4), .launch = .{ .threads = threads } },
    },
}};
