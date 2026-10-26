const std = @import("std");
const configurations = @import("configurations.zig");

pub const Selectors = struct {
    a_byte: u2,
    a_thread: u1,
    b_byte: u2,
    b_thread: u2,
};

pub const selectors = [_]Selectors{
    .{ .a_byte = 0, .a_thread = 0, .b_byte = 0, .b_thread = 0 },
    .{ .a_byte = 3, .a_thread = 1, .b_byte = 2, .b_thread = 3 },
};

pub const tile_rows = 64;
pub const tile_columns = 128;
pub const stages = 2;
pub const copy_threads = 128;
pub const wait_limit_ns = 1_000_000_000;

pub const f8f6f4_kernel_name = "mma_check_sm120_f8f6f4";
pub const tma_copy_kernel_name = "mma_check_sm120_tma_copy";

pub fn blockScaleKernelName(comptime index: usize) []const u8 {
    const s = selectors[index];
    return std.fmt.comptimePrint("mma_check_sm120_block_scale_a{d}{d}_b{d}{d}", .{
        s.a_byte,
        s.a_thread,
        s.b_byte,
        s.b_thread,
    });
}

const warp: configurations.Launch = .{ .threads = 32 };
const copy: configurations.Launch = .{
    .threads = copy_threads,
    .tile_m = tile_rows,
    .tile_k = tile_columns,
    .stages = stages,
};

pub const modules: []const configurations.Module = &.{.{
    .name = "mma_check_sm120",
    .source = "mma_check_sm120",
    .arch = 120,
    .targets = &.{"sm_120f"},
    .kernels = &.{
        .{ .name = f8f6f4_kernel_name, .launch = warp },
        .{ .name = blockScaleKernelName(0), .launch = warp },
        .{ .name = blockScaleKernelName(1), .launch = warp },
        .{ .name = tma_copy_kernel_name, .launch = copy },
    },
}};
