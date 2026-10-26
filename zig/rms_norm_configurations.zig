//! An instance normalizes each row `h` to `h * r` with `r = rsqrt(mean(h * h) + epsilon)`. With the
//! weight `w`, `Weight.bf16` gives `w * round(h * r)` (LLaMA), where `round` rounds to the dtype of
//! `x`. `Weight.f32` gives `w * (h * r)`, and zero-centered weights give `h * r * (1 + w)` (Gemma).
//! The add writes `h = round(x + a)` to `residual_out`. For the values of `Add` in order, `a` is
//! `residual`, `bf16(gate * branch)`, the BF16 norm `n` of `branch` or `bf16(gate * n)`.

const std = @import("std");
const configurations = @import("configurations.zig");

pub const Element = enum { bf16, f32 };
pub const Add = enum { none, residual, gated, dual, dual_gated };
pub const Weight = enum { none, bf16, f32, zero_centered_bf16, zero_centered_f32 };
pub const Modulation = enum { none, scale, scale_shift };
pub const Output = enum {
    bf16,
    f32,
    fp8_tensorwise,
    fp8_rowwise,
    fp8_blockwise,
    fp8_blockwise_e8m0,
};

pub const Options = struct {
    x: Element = .bf16,
    add: Add = .none,
    weight: Weight = .none,
    modulation: Modulation = .none,
    output: Output = .bf16,
};

pub const Size = struct {
    chunks_per_thread: u32,
    /// The most work items of a work group. Fewer work items leave each one more registers.
    threads: u32 = max_threads,
    /// The most registers of a work item on `sm_100` and later, or null for the limit that
    /// `threads` sets. ptxas spills the `sm_90` kernels under the same limit.
    max_registers: ?u32 = null,

    pub fn maxWidth(size: Size) u32 {
        return size.threads * size.chunks_per_thread * values_per_chunk;
    }
};

/// The register engine loads each row with `ld.global`. The bulk engine stages the rows in shared
/// memory with bulk copies and needs `sm_90`.
pub const Engine = enum { registers, bulk };

pub const values_per_chunk = 8;
pub const scale_group = 128;
pub const max_threads = 1024;
pub const max_stages = 32;

/// One chunk per work item serves only the rows of fewer than 32 chunks, and with at most 256 work
/// items, each work item has the registers of the gated and dual kernels.
const sizes = [_]Size{
    .{ .chunks_per_thread = 1, .threads = 256 },
    .{ .chunks_per_thread = 2 },
    .{ .chunks_per_thread = 4 },
};

/// Eight chunks per work item spread the per-row work of an FP8 output over twice the values of four
/// chunks, and six chunks split rows of 384, 768, 1536 or 3072 values among work items without
/// padding. With 96 registers per work item, five work groups of 128 work items fit on a streaming
/// multiprocessor.
const fp8_sizes = sizes ++ [_]Size{
    .{ .chunks_per_thread = 6, .threads = 512, .max_registers = 96 },
    .{ .chunks_per_thread = 8, .threads = 512, .max_registers = 96 },
};

/// With an addend, a work item of four chunks keeps a copy of the row code without bounds checks
/// only with at most 512 work items; the 1024 work items of the wider rows leave too few registers.
const fp8_addend_sizes = sizes[0..2].* ++ [_]Size{.{ .chunks_per_thread = 4, .threads = 512 }} ++
    sizes[2..].*;

/// An FP32 row and its addend in four chunks per work item would not fit in the registers.
pub fn sizesOf(comptime options: Options) []const Size {
    if (isFp8(options.output)) return if (options.add == .none) &fp8_sizes else &fp8_addend_sizes;
    return if (options.x == .f32 and options.add != .none) sizes[0..2] else &sizes;
}

pub const instances = [_]Options{
    .{},
    .{ .weight = .bf16 },
    .{ .weight = .f32 },
    .{ .weight = .zero_centered_bf16 },
    .{ .weight = .zero_centered_f32 },
    .{ .x = .f32, .output = .f32 },
    .{ .x = .f32, .weight = .f32, .output = .f32 },
    .{ .x = .f32, .weight = .bf16 },
    .{ .x = .f32, .weight = .bf16, .output = .f32 },
    .{ .x = .f32, .weight = .zero_centered_f32, .output = .f32 },
    .{ .add = .residual },
    .{ .add = .residual, .weight = .bf16 },
    .{ .add = .residual, .weight = .zero_centered_bf16 },
    .{ .x = .f32, .add = .residual, .weight = .bf16 },
    .{ .x = .f32, .add = .residual, .weight = .bf16, .output = .f32 },
    .{ .weight = .bf16, .output = .fp8_tensorwise },
    .{ .weight = .bf16, .output = .fp8_rowwise },
    .{ .weight = .bf16, .output = .fp8_blockwise },
    .{ .weight = .bf16, .output = .fp8_blockwise_e8m0 },
    .{ .add = .residual, .weight = .bf16, .output = .fp8_tensorwise },
    .{ .add = .residual, .weight = .bf16, .output = .fp8_rowwise },
    .{ .add = .residual, .weight = .bf16, .output = .fp8_blockwise },
    .{ .add = .residual, .weight = .bf16, .output = .fp8_blockwise_e8m0 },
    .{ .modulation = .scale },
    .{ .modulation = .scale_shift },
    .{ .add = .gated, .modulation = .scale },
    .{ .add = .gated, .modulation = .scale_shift },
    .{ .weight = .zero_centered_f32, .modulation = .scale },
    .{ .weight = .zero_centered_f32, .modulation = .scale_shift },
    .{ .add = .gated, .weight = .zero_centered_f32, .modulation = .scale },
    .{ .add = .gated, .weight = .zero_centered_f32, .modulation = .scale_shift },
    .{ .add = .gated, .modulation = .scale_shift, .output = .fp8_rowwise },
    .{ .add = .gated, .modulation = .scale_shift, .output = .fp8_blockwise },
    .{ .add = .gated, .modulation = .scale_shift, .output = .fp8_blockwise_e8m0 },
    .{ .add = .dual, .weight = .bf16 },
    .{ .add = .dual, .weight = .bf16, .modulation = .scale },
    .{ .add = .dual_gated, .weight = .bf16 },
    .{ .add = .dual_gated, .weight = .bf16, .modulation = .scale },
    .{ .add = .dual, .weight = .zero_centered_bf16 },
    .{ .add = .dual, .weight = .bf16, .output = .fp8_tensorwise },
    .{ .add = .dual, .weight = .bf16, .output = .fp8_rowwise },
    .{ .add = .dual, .weight = .bf16, .output = .fp8_blockwise },
    .{ .add = .dual, .weight = .bf16, .output = .fp8_blockwise_e8m0 },
};

pub fn instanceName(comptime options: Options) []const u8 {
    const add = switch (options.add) {
        .none => "",
        .residual => "_add",
        .gated => "_gated",
        .dual => "_dual",
        .dual_gated => "_dual_gated",
    };
    const weight = switch (options.weight) {
        .none => "",
        .bf16 => "_weight_bf16",
        .f32 => "_weight_f32",
        .zero_centered_bf16 => "_zero_centered_bf16",
        .zero_centered_f32 => "_zero_centered_f32",
    };
    const modulation = switch (options.modulation) {
        .none => "",
        .scale => "_scale",
        .scale_shift => "_scale_shift",
    };
    return "rms_norm_" ++ @tagName(options.x) ++ add ++ weight ++ modulation ++ "_to_" ++
        @tagName(options.output);
}

fn engineSuffix(comptime engine: Engine) []const u8 {
    return switch (engine) {
        .registers => "",
        .bulk => "_bulk",
    };
}

pub fn kernelName(comptime options: Options, comptime size: Size, comptime engine: Engine) []const u8 {
    const suffix = engineSuffix(engine);
    return std.fmt.comptimePrint("{s}_c{d}_max{d}{s}", .{
        instanceName(options),
        size.chunks_per_thread,
        size.maxWidth(),
        suffix,
    });
}

pub fn arch(comptime options: Options) u32 {
    return if (isFp8(options.output)) 89 else 80;
}

pub fn isFp8(output: Output) bool {
    return switch (output) {
        .bf16, .f32 => false,
        .fp8_tensorwise, .fp8_rowwise, .fp8_blockwise, .fp8_blockwise_e8m0 => true,
    };
}

pub const modules: []const configurations.Module = modules: {
    @setEvalBranchQuota(400_000);
    var list: [std.enums.values(Engine).len * instances.len]configurations.Module = undefined;
    for (std.enums.values(Engine)) |engine| {
        for (instances, 0..) |options, index| {
            const options_sizes = sizesOf(options);
            var kernels: [options_sizes.len]configurations.Kernel = undefined;
            for (&kernels, options_sizes) |*kernel, size| kernel.* = .{
                .name = kernelName(options, size, engine),
                .launch = .{
                    .threads = size.threads,
                    .max_width = size.maxWidth(),
                    .chunks_per_thread = size.chunks_per_thread,
                },
            };
            const final = kernels;
            const module_arch = if (engine == .bulk) 90 else arch(options);
            const suffix = engineSuffix(engine);
            list[@backingInt(engine) * instances.len + index] = .{
                .name = std.fmt.comptimePrint("{s}{s}_sm{d}", .{ instanceName(options), suffix, module_arch }),
                .source = "rms_norm",
                .variant = @backingInt(engine) * instances.len + index,
                .arch = module_arch,
                .kernels = &final,
            };
        }
    }
    const final = list;
    break :modules &final;
};
