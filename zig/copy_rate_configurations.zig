const std = @import("std");
const configurations = @import("configurations.zig");

pub const Kind = enum { registers, bulk, cast };

pub const register_threads = 256;
pub const unrolls = [_]u32{ 1, 2, 4, 8 };
pub const max_stages = 32;

pub fn registerKernelName(comptime unroll: u32) []const u8 {
    return std.fmt.comptimePrint("copy_rate_registers_unroll{d}", .{unroll});
}

pub fn halveKernelName(comptime unroll: u32) []const u8 {
    return std.fmt.comptimePrint("copy_rate_halve_unroll{d}", .{unroll});
}

pub fn readKernelName(comptime unroll: u32) []const u8 {
    return std.fmt.comptimePrint("copy_rate_read_unroll{d}", .{unroll});
}

pub fn castKernelName(comptime unroll: u32) []const u8 {
    return std.fmt.comptimePrint("copy_rate_cast_unroll{d}", .{unroll});
}

pub fn castRoundedKernelName(comptime unroll: u32) []const u8 {
    return std.fmt.comptimePrint("copy_rate_cast_rounded_unroll{d}", .{unroll});
}

pub const bulk_kernel_name = "copy_rate_bulk";

pub const modules: []const configurations.Module = &.{
    .{
        .name = "copy_rate_sm80",
        .source = "copy_rate",
        .variant = @backingInt(Kind.registers),
        .arch = 80,
        .kernels = kernels: {
            var list: [3 * unrolls.len]configurations.Kernel = undefined;
            for (unrolls, 0..) |unroll, index| {
                const names = .{ registerKernelName(unroll), halveKernelName(unroll), readKernelName(unroll) };
                for (names, 0..) |name, kind| list[kind * unrolls.len + index] = .{
                    .name = name,
                    .launch = .{ .threads = register_threads },
                };
            }
            const final = list;
            break :kernels &final;
        },
    },
    .{
        .name = "copy_rate_sm89",
        .source = "copy_rate",
        .variant = @backingInt(Kind.cast),
        .arch = 89,
        .kernels = kernels: {
            var list: [2 * unrolls.len]configurations.Kernel = undefined;
            for (unrolls, 0..) |unroll, index| {
                list[index] = .{ .name = castKernelName(unroll), .launch = .{ .threads = register_threads } };
                list[unrolls.len + index] = .{
                    .name = castRoundedKernelName(unroll),
                    .launch = .{ .threads = register_threads },
                };
            }
            const final = list;
            break :kernels &final;
        },
    },
    .{
        .name = "copy_rate_sm90",
        .source = "copy_rate",
        .variant = @backingInt(Kind.bulk),
        .arch = 90,
        .kernels = &.{.{ .name = bulk_kernel_name, .launch = .{ .threads = 32 } }},
    },
};
