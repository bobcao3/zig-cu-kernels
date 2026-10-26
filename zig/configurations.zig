//! `build.zig` compiles `modules` and writes each kernel and its `Launch` to `kernels.manifest`.

const std = @import("std");

pub const axpby = @import("axpby_configurations.zig");
pub const gemm = @import("gemm_configurations.zig");
pub const fp8_gemm = @import("fp8_gemm_configurations.zig");
pub const rms_norm = @import("rms_norm_configurations.zig");
pub const mma_rate = @import("mma_rate_configurations.zig");
pub const copy_rate = @import("copy_rate_configurations.zig");
pub const launch_test = @import("launch_test_configurations.zig");
pub const mma_check_sm120 = @import("mma_check_sm120_configurations.zig");
pub const mma_rate_sm120 = @import("mma_rate_sm120_configurations.zig");
pub const fp8_pack = @import("fp8_pack_configurations.zig");
pub const mma_check_sm100 = @import("mma_check_sm100_configurations.zig");
pub const mma_rate_sm100 = @import("mma_rate_sm100_configurations.zig");
pub const mma_check_sm90 = @import("mma_check_sm90_configurations.zig");
pub const mma_rate_sm90 = @import("mma_rate_sm90_configurations.zig");
pub const mm_sm90 = @import("mm_sm90_configurations.zig");
pub const mm_sm120 = @import("mm_sm120_configurations.zig");
pub const mm_sm100 = @import("mm_sm100_configurations.zig");
pub const glu_quant = @import("glu_quant_configurations.zig");
pub const qkv_prep = @import("qkv_prep_configurations.zig");
pub const glu_bwd = @import("glu_bwd_configurations.zig");
pub const decode_sm90 = @import("decode_sm90_configurations.zig");
pub const decode_sm120 = @import("decode_sm120_configurations.zig");
pub const decode_sm100 = @import("decode_sm100_configurations.zig");
pub const decode_reduce = @import("decode_reduce_configurations.zig");

pub const Launch = struct {
    threads: u32,
    shared_bytes: u32 = 0,
    tile_m: u32 = 0,
    tile_n: u32 = 0,
    tile_k: u32 = 0,
    stages: u32 = 0,
    warp_m: u32 = 0,
    warp_n: u32 = 0,
    max_width: u32 = 0,
    chunks_per_thread: u32 = 0,
    max_splits: u32 = 0,
};

pub const Kernel = struct {
    name: []const u8,
    launch: Launch,
};

pub const Module = struct {
    name: []const u8,
    source: []const u8,
    variant: u32 = 0,
    arch: u32,
    /// When a module names its targets, `build.zig` compiles one PTX file per target with the
    /// features of that target, for example those of `sm_90a` or of the family `sm_100f`.
    targets: ?[]const []const u8 = null,
    kernels: []const Kernel,
};

pub const modules: []const Module = axpby.modules ++ gemm.modules ++ fp8_gemm.modules ++
    rms_norm.modules ++ mma_rate.modules ++ copy_rate.modules ++ launch_test.modules ++
    mma_check_sm120.modules ++ mma_rate_sm120.modules ++ fp8_pack.modules ++
    mma_check_sm100.modules ++ mma_rate_sm100.modules ++ mma_check_sm90.modules ++
    mma_rate_sm90.modules ++ mm_sm90.modules ++ mm_sm120.modules ++ mm_sm100.modules ++
    glu_quant.modules ++ qkv_prep.modules ++ decode_sm90.modules ++
    decode_sm120.modules ++ decode_sm100.modules ++ decode_reduce.modules ++ glu_bwd.modules;

pub fn ptxTarget(comptime arch: u32) []const u8 {
    return std.fmt.comptimePrint("sm_{d}", .{arch});
}

/// ptxas compiles the cubin of `target` from PTX for the returned CPU. The cubins of `sm_90`,
/// `sm_100` and `sm_120` come from PTX for their own architecture, which has the bulk prefetch
/// (`sm_90`) and the packed FP32 instructions (`sm_100`) that older PTX lacks.
pub fn ptxCpu(comptime module: Module, target: []const u8) []const u8 {
    if (module.targets != null) return target;
    inline for (.{ 90, 100, 120 }) |own| {
        const own_target = comptime ptxTarget(own);
        if (std.mem.eql(u8, target, own_target) and module.arch < own) return own_target;
    }
    return comptime ptxTarget(module.arch);
}

/// `build.zig` installs the PTX for this CPU as `ptx/<module name>.ptx`.
pub fn ownCpu(comptime module: Module) []const u8 {
    return if (module.targets) |targets| targets[0] else comptime ptxTarget(module.arch);
}

/// The cubin of a module holds machine code for these targets. ptxas compiles the PTX of an older
/// target for each newer one, so the `sm_80` PTX also yields the `sm_86` and `sm_89` machine code.
pub fn cubinTargets(comptime module: Module) []const []const u8 {
    return module.targets orelse archTargets(module.arch) ++ newerTargets(module.arch);
}

/// A build without `-Dsm` assembles these targets of `cubinTargets`.
pub fn defaultTargets(comptime module: Module) []const []const u8 {
    return module.targets orelse archTargets(module.arch);
}

fn archTargets(comptime arch: u32) []const []const u8 {
    return switch (arch) {
        80 => &.{ "sm_80", "sm_86", "sm_89" },
        89 => &.{"sm_89"},
        90 => &.{ "sm_90", "sm_100", "sm_120" },
        else => @compileError("unsupported architecture"),
    };
}

fn newerTargets(comptime arch: u32) []const []const u8 {
    return switch (arch) {
        80, 89 => &.{ "sm_90", "sm_100", "sm_120" },
        90 => &.{},
        else => @compileError("unsupported architecture"),
    };
}

comptime {
    @setEvalBranchQuota(4_000_000);
    var names: []const []const u8 = &.{};
    for (modules) |module| {
        for (module.kernels) |kernel| {
            for (names) |name| {
                if (std.mem.eql(u8, name, kernel.name)) @compileError("two kernels are named " ++ name);
            }
            names = names ++ .{kernel.name};
        }
    }
}

test "a module that names its targets compiles the PTX of each target" {
    const module: Module = .{
        .name = "m",
        .source = "m",
        .arch = 100,
        .targets = &.{ "sm_100f", "sm_120f" },
        .kernels = &.{},
    };
    try std.testing.expectEqual(module.targets.?, comptime cubinTargets(module));
    try std.testing.expectEqual(module.targets.?, comptime defaultTargets(module));
    try std.testing.expectEqualStrings("sm_100f", ownCpu(module));
    try std.testing.expectEqualStrings("sm_120f", ptxCpu(module, "sm_120f"));
    const older: Module = .{ .name = "o", .source = "o", .arch = 80, .kernels = &.{} };
    try std.testing.expectEqualStrings("sm_80", ptxCpu(older, "sm_89"));
    try std.testing.expectEqualStrings("sm_100", ptxCpu(older, "sm_100"));
}
