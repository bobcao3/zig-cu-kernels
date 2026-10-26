const configurations = @import("configurations.zig");

pub const Element = enum { bf16, f32 };

pub const threads = 128;
pub const chunks_per_thread = 1;

pub fn kernelName(comptime element: Element) []const u8 {
    return "axpby_" ++ @tagName(element);
}

pub const modules: []const configurations.Module = &.{.{
    .name = "axpby_sm80",
    .source = "axpby",
    .arch = 80,
    .kernels = &.{
        .{ .name = kernelName(.bf16), .launch = launch },
        .{ .name = kernelName(.f32), .launch = launch },
    },
}};

const launch: configurations.Launch = .{ .threads = threads, .chunks_per_thread = chunks_per_thread };
