const std = @import("std");
const configurations = @import("configurations.zig");

pub const Activation = @import("glu_quant_configurations.zig").Activation;

pub const threads = 128;

/// `operands` writes the three MXFP8 packs of the gradient products, and `bf16` writes the BF16
/// gradient with an optional BF16 `h` and rowwise pack of the gradient.
pub const Outputs = enum { operands, bf16 };

pub fn kernelName(comptime outputs: Outputs, comptime activation: Activation) []const u8 {
    const prefix = switch (outputs) {
        .operands => "mxfp8_glu_backward_",
        .bf16 => "mxfp8_glu_backward_bf16_",
    };
    return prefix ++ @tagName(activation);
}

pub const modules: []const configurations.Module = &.{.{
    .name = "glu_bwd_sm89",
    .source = "glu_bwd",
    .arch = 89,
    .targets = &.{ "sm_89", "sm_90", "sm_100", "sm_120" },
    .kernels = kernels(),
}};

fn kernels() []const configurations.Kernel {
    var list: []const configurations.Kernel = &.{};
    for (std.enums.values(Outputs)) |outputs| {
        for (std.enums.values(Activation)) |activation| {
            list = list ++ .{configurations.Kernel{
                .name = kernelName(outputs, activation),
                .launch = .{ .threads = threads },
            }};
        }
    }
    return list;
}
