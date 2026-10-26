const std = @import("std");
const configurations = @import("configurations.zig");

pub const Recipe = enum { mxfp8, fp8_b128, fp8_rowwise };

pub const recipes = [_]Recipe{ .mxfp8, .fp8_b128, .fp8_rowwise };
pub const token_tiles = [_]u32{ 16, 32, 64, 128 };
pub const epilogue_warps = 4;
pub const threads = (epilogue_warps + 2) * 32;
pub const tile_rows = 128;
pub const k_block = 128;
pub const tile_alignment = 1024;
pub const scale_atom_bytes = 512;
pub const scale_slot_bytes = 2 * scale_atom_bytes;
pub const max_stages = 12;
pub const max_shared_bytes = 232448;

pub fn valueStageBytes(comptime tokens: u32) u32 {
    return tile_rows * k_block + tokens * k_block;
}

pub fn stages(comptime tokens: u32) u32 {
    const per_stage = valueStageBytes(tokens) + scale_slot_bytes;
    return @min(max_stages, (max_shared_bytes - tile_alignment) / per_stage);
}

pub fn sharedBytes(comptime tokens: u32) u32 {
    return tile_alignment + stages(tokens) * (valueStageBytes(tokens) + scale_slot_bytes);
}

pub fn name(comptime recipe: Recipe, comptime tokens: u32) []const u8 {
    return @tagName(recipe) ++ "_decode_mm_abt_sm100_t" ++ std.fmt.comptimePrint("{d}", .{tokens});
}

fn kernels() [recipes.len * token_tiles.len]configurations.Kernel {
    var list: [recipes.len * token_tiles.len]configurations.Kernel = undefined;
    for (recipes, 0..) |recipe, r| {
        for (token_tiles, 0..) |tokens, t| {
            list[r * token_tiles.len + t] = .{ .name = name(recipe, tokens), .launch = .{
                .threads = threads,
                .shared_bytes = sharedBytes(tokens),
                .tile_m = tile_rows,
                .tile_n = tokens,
                .tile_k = k_block,
                .stages = stages(tokens),
            } };
        }
    }
    return list;
}

const kernel_list = kernels();

pub const modules: []const configurations.Module = &.{.{
    .name = "decode_sm100",
    .source = "decode_sm100",
    .arch = 100,
    .targets = &.{"sm_100f"},
    .kernels = &kernel_list,
}};
