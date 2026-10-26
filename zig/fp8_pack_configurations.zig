const std = @import("std");
const configurations = @import("configurations.zig");

pub const ScaleFormat = @import("fp8.zig").ScaleFormat;

pub const Element = enum { bf16, f32 };

pub const Quantizer = enum { mxfp8, groups, weight, rowwise, rowwise_held, columnwise, dual };

pub const threads = 256;
pub const chunk_values = 16;
pub const tile = 128;
pub const row_chunks = 4;

pub fn kernelName(
    comptime quantizer: Quantizer,
    comptime element: Element,
    comptime format: ScaleFormat,
) []const u8 {
    const prefix = switch (quantizer) {
        .mxfp8 => "mxfp8_quant_",
        .groups => "fp8_b128_quant_",
        .weight => "fp8_b128_quant_weight_",
        .rowwise => "fp8_rowwise_quant_",
        .rowwise_held => "fp8_rowwise_quant_held_",
        .columnwise => "mxfp8_quant_columnwise_",
        .dual => "mxfp8_quant_dual_",
    };
    const suffix = switch (quantizer) {
        .mxfp8, .rowwise, .rowwise_held, .columnwise, .dual => "",
        .groups, .weight => "_" ++ @tagName(format),
    };
    return prefix ++ @tagName(element) ++ suffix;
}

pub fn formatsOf(comptime quantizer: Quantizer) []const ScaleFormat {
    return switch (quantizer) {
        .mxfp8, .columnwise, .dual => &.{.e8m0},
        .rowwise, .rowwise_held => &.{.f32},
        .groups, .weight => &.{ .e8m0, .f32 },
    };
}

pub const modules: []const configurations.Module = &.{.{
    .name = "fp8_pack_sm89",
    .source = "fp8_pack",
    .arch = 89,
    .targets = &.{ "sm_89", "sm_90", "sm_100", "sm_120" },
    .kernels = kernels(),
}};

fn kernels() []const configurations.Kernel {
    var list: []const configurations.Kernel = &.{};
    for (std.enums.values(Quantizer)) |quantizer| {
        for (std.enums.values(Element)) |element| {
            for (formatsOf(quantizer)) |format| {
                list = list ++ .{configurations.Kernel{
                    .name = kernelName(quantizer, element, format),
                    .launch = .{
                        .threads = threads,
                        .tile_m = if (quantizer == .weight) tile else 0,
                    },
                }};
            }
        }
    }
    return list;
}
