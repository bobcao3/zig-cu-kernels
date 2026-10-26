const std = @import("std");
const configurations = @import("configurations.zig");

pub const weight_rows = 128;
pub const k_block = 128;
pub const threads = 384;
pub const wait_limit_ns = 1_000_000_000;
pub const max_stages = 12;
pub const scale_atom_rows = 128;
/// The dynamic shared memory of a CTA on SM90 is at most 227 KiB.
pub const shared_limit = 227 * 1024;

pub const Recipe = enum { fp8_b128_e8m0, fp8_b128_fp32, fp8_rowwise };
pub const Epilogue = enum { store, bias };

pub const Variant = struct {
    recipe: Recipe,
    epilogue: Epilogue,
    tokens: u32,

    pub fn blockScaled(variant: Variant) bool {
        return variant.recipe != .fp8_rowwise;
    }

    pub fn stageBytes(variant: Variant) u32 {
        return (weight_rows + variant.tokens) * k_block;
    }

    pub fn scaleBytes(variant: Variant) u32 {
        if (!variant.blockScaled()) return 0;
        const element: u32 = if (variant.recipe == .fp8_b128_e8m0) 1 else 4;
        return @min(scale_atom_rows * 4, 16 * variant.tokens) * element;
    }

    pub fn stages(variant: Variant) u32 {
        const bytes = variant.stageBytes() + variant.scaleBytes();
        return @min(max_stages, (shared_limit - 2048) / bytes);
    }

    /// The kernel aligns its stages to 1024 bytes, the repeat of the 128-byte swizzle pattern.
    pub fn sharedBytes(variant: Variant) u32 {
        return variant.stages() * (variant.stageBytes() + variant.scaleBytes()) + 1024;
    }
};

pub const token_tiles = [_]u32{ 8, 16, 32, 64, 128 };

const recipe_epilogues = [_]struct { Recipe, Epilogue }{
    .{ .fp8_b128_e8m0, .store },
    .{ .fp8_b128_fp32, .store },
    .{ .fp8_rowwise, .store },
    .{ .fp8_rowwise, .bias },
};

pub const variants = variants: {
    var list: [recipe_epilogues.len * token_tiles.len]Variant = undefined;
    for (recipe_epilogues, 0..) |pair, index| {
        for (token_tiles, 0..) |tokens, within| list[index * token_tiles.len + within] = .{
            .recipe = pair[0],
            .epilogue = pair[1],
            .tokens = tokens,
        };
    }
    break :variants list;
};

pub fn kernelName(comptime variant: Variant) []const u8 {
    const suffix = if (variant.epilogue == .bias) "_bias" else "";
    return std.fmt.comptimePrint(
        "{s}_decode_sm90_t{d}{s}",
        .{ @tagName(variant.recipe), variant.tokens, suffix },
    );
}

const kernels = kernels: {
    var list: [variants.len]configurations.Kernel = undefined;
    for (variants, &list) |variant, *kernel| kernel.* = .{
        .name = kernelName(variant),
        .launch = .{
            .threads = threads,
            .shared_bytes = variant.sharedBytes(),
            .tile_m = weight_rows,
            .tile_n = variant.tokens,
            .tile_k = k_block,
            .stages = variant.stages(),
        },
    };
    const final = list;
    break :kernels &final;
};

pub const modules: []const configurations.Module = &.{.{
    .name = "decode_sm90",
    .source = "decode_sm90",
    .arch = 90,
    .targets = &.{"sm_90a"},
    .kernels = kernels,
}};
