const std = @import("std");
const configurations = @import("configurations.zig");

pub const k_block = 128;
pub const cluster_ctas = 2;
pub const wait_limit_ns = 1_000_000_000;
pub const probe_tiles = 32;
pub const timeline_ctas = 4;
pub const timeline_blocks = 512;

pub const Recipe = enum { fp8_b128_e8m0, fp8_b128_fp32, fp8_rowwise };
pub const Epilogue = enum { store, bias, swiglu, geglu_tanh };

pub const Variant = struct {
    recipe: Recipe,
    epilogue: Epilogue,
    n: u32,
    stages: u32,
    /// A promoted kernel adds the product of each 128-deep K block to FP32 sums, and a fast kernel
    /// sums all of K in the `wgmma` accumulator.
    promoted: bool = true,
    /// An overlapped promoted kernel multiplies the next K block while it adds the previous one.
    overlapped: bool = false,
    /// A deferred promoted kernel decodes the scales of a K block after its `wgmma` wait.
    deferred: bool = true,
    /// An early kernel tests the full barrier of the next stage right after it issues the `wgmma`
    /// of a K block, and lane `i` of each consumer warp releases the stage to CTA `i` of the cluster.
    early: bool = false,
    /// A hoisted kernel computes the scale addresses of its rows and columns once per tile and
    /// advances the `wgmma` descriptors with 32-bit adds.
    hoisted: bool = false,
    /// The scale loads of an eager kernel run this many K blocks ahead of the block that uses them.
    scales_ahead: u32 = 1,
    /// The producer of a kernel with shared scales copies the scales of each K block into its stage.
    shared_scales: bool = false,
    /// A kernel with shared scales and scales first reads the scales of a K block from its stage
    /// before it issues the `wgmma` of the block.
    scales_first: bool = false,
    /// A kernel with a split tail splits the K blocks of the cluster tiles of its last round into
    /// ranges, and the warpgroup that finishes a tile last adds the FP32 sums of all ranges.
    split_tail: bool = false,
    /// In an ordered kernel, consumer 1 issues the `wgmma` of a K block only after consumer 0 has
    /// issued it, through named barrier 2.
    ordered: bool = false,
    /// A timeline kernel writes the clock at eight points of each of the first `timeline_blocks`
    /// K blocks of the first `timeline_ctas` CTAs to the bias argument.
    timeline: bool = false,
    /// A kernel with a late try-wait tests the full barrier of the next stage after the release
    /// of a K block instead of after its issue.
    late_try_wait: bool = false,
    /// A kernel with a test wait tests the full barrier of the next stage with the non-blocking
    /// `mbarrier.test_wait`. PTX ISA 9.7.15.16.19.
    test_wait: bool = false,
    probe: bool = false,
    name_suffix: []const u8 = "",
    band_pairs: u32 = 8,
    consumers: u32 = 2,

    pub fn tileM(variant: Variant) u32 {
        return 64 * variant.consumers;
    }

    pub fn threads(variant: Variant) u32 {
        return 128 * (variant.consumers + 1);
    }

    /// The register file holds 65536 registers, and `setmaxnreg` counts in multiples of 8.
    pub fn registers(variant: Variant) Registers {
        return switch (variant.consumers) {
            2 => .{ .producer = 40, .consumer = 232 },
            3 => .{ .producer = 24, .consumer = 160 },
            else => unreachable,
        };
    }

    /// A GLU kernel multiplies the `lo` rows of `w1` for half of the tile columns and the `hi`
    /// rows for the other half.
    pub fn glu(variant: Variant) bool {
        return variant.epilogue == .swiglu or variant.epilogue == .geglu_tanh;
    }

    pub fn blockScaled(variant: Variant) bool {
        return variant.recipe != .fp8_rowwise;
    }

    pub fn stageBytes(variant: Variant) u32 {
        return (variant.tileM() + variant.n) * k_block;
    }

    /// One scale atom of `A` (128 rows by 4 K blocks) and the two column scales of `B` per stage.
    pub fn scaleStageBytes(variant: Variant) u32 {
        if (!variant.shared_scales) return 0;
        const scale_bytes: u32 = if (variant.recipe == .fp8_b128_e8m0) 1 else 4;
        return 512 * scale_bytes + 16;
    }

    /// The kernel aligns its tiles to 1024 bytes, the repeat of the 128-byte swizzle pattern.
    pub fn sharedBytes(variant: Variant) u32 {
        return variant.stages * (variant.stageBytes() + variant.scaleStageBytes()) + 1024;
    }
};

pub const Registers = struct { producer: u32, consumer: u32 };

pub const variants = [_]Variant{
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .test_wait = true },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .bias, .n = 192, .stages = 5, .deferred = false },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .test_wait = true },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .bias, .n = 192, .stages = 5, .deferred = false },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .name_suffix = "_try" },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .name_suffix = "_try" },
    .{ .recipe = .fp8_rowwise, .epilogue = .store, .n = 192, .stages = 5, .early = true },
    .{ .recipe = .fp8_rowwise, .epilogue = .bias, .n = 192, .stages = 5 },
    .{ .recipe = .fp8_rowwise, .epilogue = .store, .n = 256, .stages = 4, .promoted = false },
    .{ .recipe = .fp8_rowwise, .epilogue = .store, .n = 128, .stages = 5, .deferred = false, .consumers = 3 },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .scales_first = true, .name_suffix = "_lds" },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .scales_first = true, .name_suffix = "_lds" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .split_tail = true, .test_wait = true, .name_suffix = "_split" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .split_tail = true, .name_suffix = "_split_try" },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .split_tail = true, .test_wait = true, .name_suffix = "_split" },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .split_tail = true, .name_suffix = "_split_try" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 128, .stages = 6, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .scales_first = true, .name_suffix = "_n128" },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 128, .stages = 6, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .scales_first = true, .name_suffix = "_n128" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .ordered = true, .name_suffix = "_ordered" },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .ordered = true, .name_suffix = "_ordered" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .late_try_wait = true, .name_suffix = "_late" },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .late_try_wait = true, .name_suffix = "_late" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .late_try_wait = true, .timeline = true, .name_suffix = "_late_timeline" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .test_wait = true, .timeline = true, .name_suffix = "_timeline" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .ordered = true, .timeline = true, .name_suffix = "_ordered_timeline" },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .probe = true },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .probe = true },
    .{ .recipe = .fp8_b128_e8m0, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .scales_first = true, .name_suffix = "_lds", .probe = true },
    .{ .recipe = .fp8_b128_fp32, .epilogue = .store, .n = 192, .stages = 5, .deferred = false, .early = true, .hoisted = true, .shared_scales = true, .scales_first = true, .name_suffix = "_lds", .probe = true },
} ++ gluVariants();

fn gluVariants() [6]Variant {
    var list: [6]Variant = undefined;
    for (std.enums.values(Recipe), 0..) |recipe, index| {
        list[2 * index] = .{ .recipe = recipe, .epilogue = .swiglu, .n = 256, .stages = 4 };
        list[2 * index + 1] = .{ .recipe = recipe, .epilogue = .geglu_tanh, .n = 256, .stages = 4 };
    }
    return list;
}

pub fn kernelName(comptime variant: Variant) []const u8 {
    if (variant.glu()) return std.fmt.comptimePrint(
        "{s}_mm_glu_{s}_sm90",
        .{ @tagName(variant.recipe), @tagName(variant.epilogue) },
    );
    const suffix = switch (variant.epilogue) {
        .store => "",
        .bias => "_bias",
        .swiglu, .geglu_tanh => unreachable,
    };
    const accumulation = if (variant.promoted) "" else "_fast";
    const probe = if (variant.probe) "_probe" else "";
    const band = if (variant.band_pairs == 8) "" else std.fmt.comptimePrint("_band{d}", .{variant.band_pairs});
    const consumers = if (variant.consumers == 2) "" else std.fmt.comptimePrint("_m{d}", .{variant.tileM()});
    return std.fmt.comptimePrint(
        "{s}_mm_abt_sm90{s}{s}{s}{s}{s}{s}",
        .{ @tagName(variant.recipe), suffix, accumulation, consumers, variant.name_suffix, probe, band },
    );
}

const kernels = kernels: {
    var list: [variants.len]configurations.Kernel = undefined;
    for (variants, &list) |variant, *kernel| kernel.* = .{
        .name = kernelName(variant),
        .launch = .{
            .threads = variant.threads(),
            .shared_bytes = variant.sharedBytes(),
            .tile_m = variant.tileM(),
            .tile_n = variant.n,
            .tile_k = k_block,
            .stages = variant.stages,
        },
    };
    const final = list;
    break :kernels &final;
};

pub const modules: []const configurations.Module = &.{.{
    .name = "mm_sm90",
    .source = "mm_sm90",
    .arch = 90,
    .targets = &.{"sm_90a"},
    .kernels = kernels,
}};
