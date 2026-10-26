const configurations = @import("configurations.zig");

pub const Recipe = enum { mxfp8, fp8_b128, fp8_rowwise };
pub const Operation = enum { mm_abt, mm_atb, mm_abt_low_rank, mm_glu_swiglu, mm_glu_geglu_tanh };
pub const glu_operations = [_]Operation{ .mm_glu_swiglu, .mm_glu_geglu_tanh };

pub const epilogue_warps = 4;
pub const threads = (epilogue_warps + 2) * 32;
pub const stages = 6;
pub const tile = 256;
pub const half = tile / 2;
pub const k_block = 128;
pub const rank_block = 64;
pub const chunk_columns = 64;
pub const tile_alignment = 1024;
pub const value_stage_bytes = 2 * half * k_block;
pub const scale_atom_bytes = 512;
pub const scale_stage_bytes = 3 * scale_atom_bytes;
pub const shared_bytes = tile_alignment + stages * (value_stage_bytes + scale_stage_bytes);
pub const output_bytes = half * chunk_columns * 2;
pub const weight_span_bytes = 2 * 32 * 4;
pub const weight_spans = 88;
pub const weight_table_bytes = weight_spans * weight_span_bytes;
pub const band_rows = 8;
pub const probe_words = 12;
pub const probe_recipes = [_]Recipe{ .mxfp8, .fp8_rowwise };

pub fn name(
    comptime recipe: Recipe,
    comptime operation: Operation,
    comptime probe: bool,
) []const u8 {
    const suffix = if (probe) "_probe" else "";
    return @tagName(recipe) ++ "_" ++ @tagName(operation) ++ "_sm100" ++ suffix;
}

fn kernel(
    comptime recipe: Recipe,
    comptime operation: Operation,
    comptime probe: bool,
) configurations.Kernel {
    const extra_bytes = if (recipe == .fp8_b128) weight_table_bytes else output_bytes;
    return .{ .name = name(recipe, operation, probe), .launch = .{
        .threads = threads,
        .shared_bytes = shared_bytes + extra_bytes,
        .tile_m = tile,
        .tile_n = tile,
        .tile_k = k_block,
        .stages = stages,
    } };
}

pub const modules: []const configurations.Module = &.{.{
    .name = "mm_sm100",
    .source = "mm_sm100",
    .arch = 100,
    .targets = &.{"sm_100f"},
    .kernels = &.{
        kernel(.mxfp8, .mm_abt, false),
        kernel(.fp8_b128, .mm_abt, false),
        kernel(.fp8_rowwise, .mm_abt, false),
        kernel(.mxfp8, .mm_abt, true),
        kernel(.fp8_rowwise, .mm_abt, true),
        kernel(.mxfp8, .mm_atb, false),
        kernel(.mxfp8, .mm_abt_low_rank, false),
        kernel(.mxfp8, .mm_glu_swiglu, false),
        kernel(.mxfp8, .mm_glu_geglu_tanh, false),
    },
}};
