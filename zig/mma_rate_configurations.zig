const configurations = @import("configurations.zig");

pub const Element = enum { bf16, e4m3 };

pub const threads = 256;
pub const chains = 8;

pub fn kernelName(comptime element: Element) []const u8 {
    return "mma_rate_" ++ @tagName(element);
}

pub const modules: []const configurations.Module = &.{
    .{
        .name = "mma_rate_sm80",
        .source = "mma_rate",
        .variant = 0,
        .arch = 80,
        .kernels = &.{.{ .name = kernelName(.bf16), .launch = .{ .threads = threads } }},
    },
    .{
        .name = "mma_rate_sm89",
        .source = "mma_rate",
        .variant = 1,
        .arch = 89,
        .kernels = &.{.{ .name = kernelName(.e4m3), .launch = .{ .threads = threads } }},
    },
};
