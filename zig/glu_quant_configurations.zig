const std = @import("std");
const configurations = @import("configurations.zig");

pub const ScaleFormat = @import("fp8.zig").ScaleFormat;

pub const Activation = enum { swiglu, geglu_tanh };

/// `rowwise` writes BF16 `h` for `fp8_rowwise_quant_bf16`, and `rowwise_one_pass` writes the pack.
pub const Pack = enum { mxfp8, b128_e8m0, b128_f32, rowwise, rowwise_one_pass };

pub const threads = 256;
pub const chunk_values = 16;
pub const row_chunks = 6;

pub fn kernelName(comptime pack: Pack, comptime activation: Activation) []const u8 {
    const prefix = switch (pack) {
        .mxfp8 => "mxfp8_glu_quant_",
        .b128_e8m0 => "fp8_b128_glu_quant_e8m0_",
        .b128_f32 => "fp8_b128_glu_quant_f32_",
        .rowwise => "fp8_rowwise_glu_bf16_",
        .rowwise_one_pass => "fp8_rowwise_glu_quant_",
    };
    return prefix ++ @tagName(activation);
}

/// ptxas spills registers of the one-pass rowwise GEGLU kernel, so GEGLU takes the two-kernel path.
pub fn hasKernel(comptime pack: Pack, comptime activation: Activation) bool {
    return pack != .rowwise_one_pass or activation == .swiglu;
}

pub const modules: []const configurations.Module = &.{.{
    .name = "glu_quant_sm89",
    .source = "glu_quant",
    .arch = 89,
    .targets = &.{ "sm_89", "sm_90", "sm_100", "sm_120" },
    .kernels = kernels(),
}};

fn kernels() []const configurations.Kernel {
    var list: []const configurations.Kernel = &.{};
    for (std.enums.values(Pack)) |pack| {
        for (std.enums.values(Activation)) |activation| {
            if (!hasKernel(pack, activation)) continue;
            list = list ++ .{configurations.Kernel{
                .name = kernelName(pack, activation),
                .launch = .{ .threads = threads },
            }};
        }
    }
    return list;
}
