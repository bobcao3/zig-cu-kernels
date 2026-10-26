const std = @import("std");
const configurations = @import("configurations.zig");

pub const threads = 256;
pub const head_dims = [_]u32{ 64, 128, 256 };

pub fn kernelName(comptime head_dim: u32) []const u8 {
    return std.fmt.comptimePrint("mxfp8_qkv_prep_d{d}", .{head_dim});
}

pub const modules: []const configurations.Module = &.{.{
    .name = "qkv_prep_sm89",
    .source = "qkv_prep",
    .arch = 89,
    .targets = &.{ "sm_89", "sm_90", "sm_100", "sm_120" },
    .kernels = kernels(),
}};

fn kernels() []const configurations.Kernel {
    var list: []const configurations.Kernel = &.{};
    for (head_dims) |head_dim| {
        list = list ++ .{configurations.Kernel{
            .name = kernelName(head_dim),
            .launch = .{ .threads = threads },
        }};
    }
    return list;
}
