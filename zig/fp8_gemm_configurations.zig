const std = @import("std");
const configurations = @import("configurations.zig");

pub const ScaleFormat = @import("fp8.zig").ScaleFormat;

/// `rowwise` kernels multiply the sum of row `r` and column `c` by the FP32 scales of row `r` of A
/// and row `c` of B. The kernels read the scales at element strides, and a zero stride gives
/// tensorwise scales. `block128` kernels multiply each chain by the scales of its scale group.
/// `mxfp8` kernels multiply each chain of 32 values by the E8M0 scales of its row of A and of B.
pub const Scaling = enum { rowwise, block128, mxfp8 };
/// `chains` kernels add the `mma` sum of each chain of at most 64 values of k to FP32 sums. `fast`
/// kernels let the tensor cores accumulate all k of a warp.
pub const Accumulation = enum { chains, fast };
pub const Variant = struct {
    scaling: Scaling,
    accumulation: Accumulation = .chains,
};
pub const variants: []const Variant = &.{
    .{ .scaling = .rowwise },
    .{ .scaling = .rowwise, .accumulation = .fast },
    .{ .scaling = .block128 },
    .{ .scaling = .mxfp8 },
};
pub const Kind = enum { store, gated, glu };
pub const GluActivation = enum(u32) { swiglu, geglu_tanh };
pub const Element = enum { bf16, f32 };
pub const Quantizer = enum { rows, tiles, tensorwise, amax };

/// A `block128` scale covers `scale_group` values of an activation row, or a weight tile of
/// `scale_group` rows by `scale_group` values.
pub const scale_group = 128;
/// A scale atom holds the four E8M0 scales of 128 values of a row in one word.
pub const mxfp8_atom_k = 128;

/// The `k_warps` warps of a warp tile split the `tile_k` values of each k tile.
pub const Tiling = struct {
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    stages: u32,
    warp_m: u32,
    warp_n: u32,
    k_warps: u32,

    pub fn threads(tiling: Tiling) u32 {
        const warp_tiles = tiling.tile_m / tiling.warp_m * (tiling.tile_n / tiling.warp_n);
        return warp_tiles * tiling.k_warps * 32;
    }

    pub fn sharedBytes(tiling: Tiling, scaling: Scaling) u32 {
        const operands = (tiling.tile_m + tiling.tile_n) * tiling.tile_k;
        const scales = switch (scaling) {
            .rowwise => 0,
            .block128 => @max(1, tiling.tile_k / scale_group) * (tiling.tile_m + 4) * @sizeOf(f32),
            .mxfp8 => tiling.tile_k / mxfp8_atom_k * (tiling.tile_m + tiling.tile_n) * @sizeOf(u32),
        };
        const warp_scales = switch (scaling) {
            .mxfp8 => tiling.threads() / 32 * (tiling.tile_k / 32 / tiling.k_warps) *
                (tiling.warp_m + tiling.warp_n) * @sizeOf(f32),
            else => 0,
        };
        return tiling.stages * (operands + scales) + warp_scales;
    }
};

pub const tiling_128x256x128: Tiling = .{
    .tile_m = 128,
    .tile_n = 256,
    .tile_k = 128,
    .stages = 2,
    .warp_m = 64,
    .warp_n = 64,
    .k_warps = 1,
};

pub const tiling_64x128x256: Tiling = .{
    .tile_m = 64,
    .tile_n = 128,
    .tile_k = 256,
    .stages = 2,
    .warp_m = 32,
    .warp_n = 64,
    .k_warps = 2,
};

/// The narrow tiles of this tiling suit products with few rows.
pub const tiling_32x64x256: Tiling = .{
    .tile_m = 32,
    .tile_n = 64,
    .tile_k = 256,
    .stages = 4,
    .warp_m = 32,
    .warp_n = 32,
    .k_warps = 4,
};

pub const tiling_128x128x128: Tiling = .{
    .tile_m = 128,
    .tile_n = 128,
    .tile_k = 128,
    .stages = 3,
    .warp_m = 64,
    .warp_n = 64,
    .k_warps = 2,
};

pub const tiling_128x64x256: Tiling = .{
    .tile_m = 128,
    .tile_n = 64,
    .tile_k = 256,
    .stages = 2,
    .warp_m = 64,
    .warp_n = 64,
    .k_warps = 4,
};

pub const tilings: []const Tiling = &.{
    tiling_128x256x128,
    tiling_64x128x256,
    tiling_32x64x256,
    tiling_128x128x128,
    tiling_128x64x256,
};

/// The scale atoms of `mxfp8` kernels leave room for fewer operand stages in the shared memory
/// of a work group (99 KiB on sm_89).
pub const mxfp8_tilings: []const Tiling = &.{
    .{ .tile_m = 128, .tile_n = 128, .tile_k = 128, .stages = 2, .warp_m = 64, .warp_n = 64, .k_warps = 1 },
    .{ .tile_m = 64, .tile_n = 128, .tile_k = 128, .stages = 3, .warp_m = 32, .warp_n = 64, .k_warps = 2 },
    .{ .tile_m = 32, .tile_n = 64, .tile_k = 256, .stages = 3, .warp_m = 32, .warp_n = 32, .k_warps = 4 },
};

pub fn tilingsOf(comptime variant: Variant) []const Tiling {
    return if (variant.scaling == .mxfp8) mxfp8_tilings else tilings;
}

pub const quantize_threads = 256;

pub fn gemmName(
    comptime variant: Variant,
    comptime kind: Kind,
    comptime tiling: Tiling,
) []const u8 {
    if (variant.scaling == .mxfp8) {
        const operation = if (kind == .glu) "glu" else "abt";
        return std.fmt.comptimePrint("mxfp8_mm_{s}_sm89_{d}x{d}", .{ operation, tiling.tile_m, tiling.tile_n });
    }
    return std.fmt.comptimePrint("fp8_gemm_{s}{s}_{s}_{d}x{d}", .{
        @tagName(variant.scaling),
        if (variant.accumulation == .fast) "_fast" else "",
        @tagName(kind),
        tiling.tile_m,
        tiling.tile_n,
    });
}

pub fn lowRankName(comptime tiling: Tiling) []const u8 {
    return std.fmt.comptimePrint("mxfp8_mm_abt_low_rank_sm89_{d}x{d}", .{ tiling.tile_m, tiling.tile_n });
}

pub fn quantizerName(
    comptime quantizer: Quantizer,
    comptime element: Element,
    comptime format: ScaleFormat,
) []const u8 {
    const prefix = if (quantizer == .amax) "fp8_" else "fp8_quantize_";
    const suffix = if (format == .e8m0) "_e8m0" else "";
    return prefix ++ @tagName(quantizer) ++ "_" ++ @tagName(element) ++ suffix;
}

pub const modules: []const configurations.Module = &.{
    .{
        .name = "fp8_gemm_sm89",
        .source = "fp8_gemm_sm89",
        .arch = 89,
        .kernels = gemmKernels(),
    },
    .{
        .name = "fp8_quantize_sm89",
        .source = "fp8_quantize",
        .arch = 89,
        .kernels = quantizerKernels(),
    },
};

fn gemmKernels() []const configurations.Kernel {
    var kernels: []const configurations.Kernel = &.{};
    for (variants) |variant| {
        for (tilingsOf(variant)) |tiling| {
            const glu: []const Kind = if (hasGlu(variant, tiling)) &.{.glu} else &.{};
            const low_rank: []const []const u8 = if (variant.scaling == .mxfp8) &.{lowRankName(tiling)} else &.{};
            var names: []const []const u8 = low_rank;
            for (kinds(variant) ++ glu) |kind| names = names ++ .{gemmName(variant, kind, tiling)};
            for (names) |name| {
                kernels = kernels ++ .{configurations.Kernel{
                    .name = name,
                    .launch = .{
                        .threads = tiling.threads(),
                        .shared_bytes = tiling.sharedBytes(variant.scaling),
                        .tile_m = tiling.tile_m,
                        .tile_n = tiling.tile_n,
                        .tile_k = tiling.tile_k,
                        .stages = tiling.stages,
                        .warp_m = tiling.warp_m,
                        .warp_n = tiling.warp_n,
                    },
                }};
            }
        }
    }
    return kernels;
}

pub fn kinds(comptime variant: Variant) []const Kind {
    return if (variant.scaling == .mxfp8) &.{.store} else &.{ .store, .gated };
}

pub fn hasGlu(comptime variant: Variant, comptime tiling: Tiling) bool {
    return switch (variant.scaling) {
        .block128 => std.meta.eql(tiling, tiling_128x256x128),
        .mxfp8 => std.meta.eql(tiling, mxfp8_tilings[0]),
        .rowwise => variant.accumulation == .chains and std.meta.eql(tiling, tiling_128x256x128),
    };
}

fn quantizerKernels() []const configurations.Kernel {
    var kernels: []const configurations.Kernel = &.{};
    for ([_]Quantizer{ .rows, .tiles, .tensorwise, .amax }) |quantizer| {
        const formats: []const ScaleFormat = switch (quantizer) {
            .rows, .tiles => &.{ .f32, .e8m0 },
            .tensorwise, .amax => &.{.f32},
        };
        for ([_]Element{ .bf16, .f32 }) |element| {
            for (formats) |format| {
                kernels = kernels ++ .{configurations.Kernel{
                    .name = quantizerName(quantizer, element, format),
                    .launch = .{
                        .threads = quantize_threads,
                        .tile_m = if (quantizer == .tiles) scale_group else 0,
                    },
                }};
            }
        }
    }
    return kernels;
}
