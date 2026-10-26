const std = @import("std");
const configurations = @import("configurations.zig");

pub const Recipe = enum { mxfp8, fp8_b128_e8m0, fp8_b128_fp32, fp8_rowwise };

pub const Epilogue = enum { store, low_rank, swiglu, geglu_tanh, qkv_prep, split };
pub const qkv_head_dim = 128;

pub const Tile = struct {
    m: u32,
    n: u32,
    k: u32 = 128,
    stages: u32,
    warps_m: u32 = 2,
    warps_n: u32 = 4,
    producer_warp: bool = true,

    pub fn threads(tile: Tile) u32 {
        return (consumer_warps + @as(u32, @intFromBool(tile.producer_warp))) * 32;
    }

    pub fn stageBytes(tile: Tile) u32 {
        return (tile.m + tile.n) * tile.k;
    }

    pub fn sharedBytes(tile: Tile) u32 {
        return tile.stages * tile.stageBytes() + swizzle_alignment;
    }

    fn launch(tile: Tile) configurations.Launch {
        return .{
            .threads = tile.threads(),
            .shared_bytes = tile.sharedBytes(),
            .tile_m = tile.m,
            .tile_n = tile.n,
            .tile_k = tile.k,
            .stages = tile.stages,
            .warp_m = tile.m / tile.warps_m,
            .warp_n = tile.n / tile.warps_n,
        };
    }
};

pub const square: Tile = .{ .m = 128, .n = 128, .stages = 3 };
pub const wide: Tile = .{ .m = 128, .n = 256, .stages = 2, .producer_warp = false };

pub const consumer_warps = 8;
pub const swizzle_alignment = 1024;
pub const wait_limit_ns = 1_000_000_000;

pub const tiles: []const Tile = &.{ square, wide };

pub fn kernelName(comptime recipe: Recipe, comptime tile: Tile) []const u8 {
    return std.fmt.comptimePrint("{s}_mm_abt_sm120_{d}x{d}x{d}", .{
        @tagName(recipe),
        tile.m,
        tile.n,
        tile.k,
    });
}

pub fn splitKernelName(comptime recipe: Recipe, comptime tile: Tile) []const u8 {
    return std.fmt.comptimePrint("{s}_mm_abt_split_sm120_{d}x{d}x{d}", .{
        @tagName(recipe),
        tile.m,
        tile.n,
        tile.k,
    });
}

pub fn hasSplit(comptime tile: Tile) bool {
    return std.meta.eql(tile, wide);
}

pub fn lowRankKernelName(comptime tile: Tile) []const u8 {
    return std.fmt.comptimePrint("mxfp8_mm_abt_low_rank_sm120_{d}x{d}x{d}", .{
        tile.m,
        tile.n,
        tile.k,
    });
}

/// The fused GLU epilogue needs warps that hold 32 columns of `lo` and 32 of `hi`.
pub fn hasGlu(comptime tile: Tile) bool {
    return tile.n / tile.warps_n == 64;
}

pub fn qkvKernelName(comptime tile: Tile) []const u8 {
    return std.fmt.comptimePrint("mxfp8_mm_qkv_prep_d{d}_sm120_{d}x{d}x{d}", .{
        qkv_head_dim,
        tile.m,
        tile.n,
        tile.k,
    });
}

pub fn gluKernelName(
    comptime recipe: Recipe,
    comptime activation: Epilogue,
    comptime tile: Tile,
) []const u8 {
    return std.fmt.comptimePrint("{s}_mm_glu_{s}_sm120_{d}x{d}x{d}", .{
        @tagName(recipe),
        @tagName(activation),
        tile.m,
        tile.n,
        tile.k,
    });
}

const kernels = kernels: {
    var list: []const configurations.Kernel = &.{};
    for (std.enums.values(Recipe)) |recipe| {
        for (tiles) |tile| {
            list = list ++ .{configurations.Kernel{
                .name = kernelName(recipe, tile),
                .launch = tile.launch(),
            }};
            if (hasSplit(tile)) {
                list = list ++ .{configurations.Kernel{
                    .name = splitKernelName(recipe, tile),
                    .launch = tile.launch(),
                }};
            }
            if (recipe == .mxfp8) {
                list = list ++ .{configurations.Kernel{
                    .name = lowRankKernelName(tile),
                    .launch = tile.launch(),
                }};
            }
            if (recipe == .mxfp8 and hasGlu(tile)) {
                list = list ++ .{configurations.Kernel{
                    .name = qkvKernelName(tile),
                    .launch = tile.launch(),
                }};
            }
            if (hasGlu(tile)) {
                for ([_]Epilogue{ .swiglu, .geglu_tanh }) |activation| {
                    list = list ++ .{configurations.Kernel{
                        .name = gluKernelName(recipe, activation, tile),
                        .launch = tile.launch(),
                    }};
                }
            }
        }
    }
    break :kernels list;
};

pub const modules: []const configurations.Module = &.{.{
    .name = "mm_sm120",
    .source = "mm_sm120",
    .arch = 120,
    .targets = &.{"sm_120f"},
    .kernels = kernels,
}};
