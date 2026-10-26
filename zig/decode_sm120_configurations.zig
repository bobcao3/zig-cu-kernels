const std = @import("std");
const configurations = @import("configurations.zig");

pub const Recipe = @import("mm_sm120_configurations.zig").Recipe;

pub const weight_rows = 128;
pub const block_k = 128;
pub const consumer_warps = 8;
pub const threads = (consumer_warps + 1) * 32;
pub const swizzle_alignment = 1024;
pub const shared_limit = 99 * 1024;
pub const max_stages = 6;
pub const wait_limit_ns = 1_000_000_000;
pub const token_tiles = [_]u32{ 16, 32, 64, 128 };

pub fn stageBytes(tokens: u32) u32 {
    return (weight_rows + tokens) * block_k;
}

pub fn stages(tokens: u32) u32 {
    return @min(max_stages, (shared_limit - swizzle_alignment) / stageBytes(tokens));
}

pub fn sharedBytes(tokens: u32) u32 {
    return stages(tokens) * stageBytes(tokens) + swizzle_alignment;
}

pub fn kernelName(comptime recipe: Recipe, comptime tokens: u32) []const u8 {
    return std.fmt.comptimePrint("{s}_mm_abt_decode_sm120_t{d}", .{ @tagName(recipe), tokens });
}

const kernels = kernels: {
    var list: []const configurations.Kernel = &.{};
    for (std.enums.values(Recipe)) |recipe| {
        for (token_tiles) |tokens| {
            list = list ++ .{configurations.Kernel{
                .name = kernelName(recipe, tokens),
                .launch = .{
                    .threads = threads,
                    .shared_bytes = sharedBytes(tokens),
                    .tile_m = weight_rows,
                    .tile_n = tokens,
                    .tile_k = block_k,
                    .stages = stages(tokens),
                },
            }};
        }
    }
    break :kernels list;
};

pub const modules: []const configurations.Module = &.{.{
    .name = "decode_sm120",
    .source = "decode_sm120",
    .arch = 120,
    .targets = &.{"sm_120f"},
    .kernels = kernels,
}};
