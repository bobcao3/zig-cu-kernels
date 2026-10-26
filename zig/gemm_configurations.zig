const std = @import("std");
const configurations = @import("configurations.zig");

pub const Kind = enum { store_bf16, store_f32, gated };

pub const Tiling = struct {
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    stages: u32,
    warp_m: u32,
    warp_n: u32,

    pub fn threads(tiling: Tiling) u32 {
        return tiling.tile_m / tiling.warp_m * (tiling.tile_n / tiling.warp_n) * 32;
    }

    pub fn sharedBytes(tiling: Tiling) u32 {
        const pipeline = tiling.stages * (tiling.tile_m + tiling.tile_n) * tiling.tile_k * 2;
        const staged = tiling.tile_m * (tiling.tile_n + staged_padding) * 2;
        return @max(pipeline, staged);
    }
};

/// The padding puts the rows of C that a warp stages in different shared memory banks.
pub const staged_padding = 8;

pub const tiling_128x256x32: Tiling = .{
    .tile_m = 128,
    .tile_n = 256,
    .tile_k = 32,
    .stages = 4,
    .warp_m = 64,
    .warp_n = 64,
};

/// Per multiply-add, this tiling copies 11% fewer bytes from the L2 cache than `tiling_128x256x32`.
pub const tiling_192x192x32: Tiling = .{
    .tile_m = 192,
    .tile_n = 192,
    .tile_k = 32,
    .stages = 4,
    .warp_m = 96,
    .warp_n = 48,
};

pub const tiling_128x128x32: Tiling = .{
    .tile_m = 128,
    .tile_n = 128,
    .tile_k = 32,
    .stages = 3,
    .warp_m = 64,
    .warp_n = 64,
};

/// A tile of this tiling holds up to 256 rows of A, so that a product of up to 256 rows copies each
/// row of B from the L2 cache once.
pub const tiling_256x128x32: Tiling = .{
    .tile_m = 256,
    .tile_n = 128,
    .tile_k = 32,
    .stages = 3,
    .warp_m = 64,
    .warp_n = 64,
};

/// The small tiles keep every streaming multiprocessor (SM) busy in a product with few tiles.
pub const tiling_64x64x64: Tiling = .{
    .tile_m = 64,
    .tile_n = 64,
    .tile_k = 64,
    .stages = 3,
    .warp_m = 32,
    .warp_n = 32,
};

/// Smaller warp tiles leave registers for the second set of accumulators of `store_f32`.
pub const tiling_128x128x64: Tiling = .{
    .tile_m = 128,
    .tile_n = 128,
    .tile_k = 64,
    .stages = 3,
    .warp_m = 64,
    .warp_n = 32,
};

pub const run_k = 256;

pub const max_splits = 8;

/// A few-rows kernel multiplies at most `max_m` rows of A. The `warps` warps of a work group take
/// turns at the values of k, `warp_k` values in each turn.
pub const FewRows = struct {
    max_m: u32,
    warps: u32,
    warp_k: u32,

    pub fn threads(shape: FewRows) u32 {
        return shape.warps * 32;
    }

    pub fn sharedBytes(shape: FewRows) u32 {
        return shape.warps * shape.max_m * partial_stride * 4;
    }
};

pub const few_rows_b_rows = 16;

/// The 4 extra FP32 values of a row of warp sums let the warps store them without bank conflicts.
pub const partial_stride = few_rows_b_rows + 4;

pub const few_rows: []const FewRows = &.{
    .{ .max_m = 8, .warps = 8, .warp_k = 64 },
    .{ .max_m = 32, .warps = 8, .warp_k = 64 },
};

pub fn fewRowsName(comptime kind: Kind, comptime shape: FewRows) []const u8 {
    return std.fmt.comptimePrint("gemm_{s}_m{d}", .{ @tagName(kind), shape.max_m });
}

pub fn tilingsOf(comptime kind: Kind) []const Tiling {
    return switch (kind) {
        .store_bf16 => &.{
            tiling_128x256x32,
            tiling_192x192x32,
            tiling_128x128x32,
            tiling_256x128x32,
            tiling_64x64x64,
        },
        .gated => &.{
            tiling_128x256x32,
            tiling_192x192x32,
            tiling_128x128x32,
            tiling_256x128x32,
            tiling_64x64x64,
        },
        .store_f32 => &.{ tiling_128x128x64, tiling_64x64x64 },
    };
}

pub fn kernelName(comptime kind: Kind, comptime tiling: Tiling) []const u8 {
    return std.fmt.comptimePrint("gemm_{s}_{d}x{d}x{d}", .{
        @tagName(kind),
        tiling.tile_m,
        tiling.tile_n,
        tiling.tile_k,
    });
}

fn launch(comptime tiling: Tiling) configurations.Launch {
    return .{
        .threads = tiling.threads(),
        .shared_bytes = tiling.sharedBytes(),
        .tile_m = tiling.tile_m,
        .tile_n = tiling.tile_n,
        .tile_k = tiling.tile_k,
        .stages = tiling.stages,
        .warp_m = tiling.warp_m,
        .warp_n = tiling.warp_n,
        .max_splits = max_splits,
    };
}

pub const modules: []const configurations.Module = &.{.{
    .name = "gemm_sm80",
    .source = "gemm_sm80",
    .arch = 80,
    .kernels = kernels: {
        var list: []const configurations.Kernel = &.{};
        for (std.enums.values(Kind)) |kind| {
            for (tilingsOf(kind)) |tiling| {
                list = list ++ .{configurations.Kernel{
                    .name = kernelName(kind, tiling),
                    .launch = launch(tiling),
                }};
            }
            for (few_rows) |shape| {
                list = list ++ .{configurations.Kernel{
                    .name = fewRowsName(kind, shape),
                    .launch = .{
                        .threads = shape.threads(),
                        .shared_bytes = shape.sharedBytes(),
                        .tile_m = shape.max_m,
                        .tile_n = few_rows_b_rows,
                        .tile_k = shape.warp_k,
                        .max_splits = 1,
                    },
                }};
            }
        }
        break :kernels list;
    },
}};
