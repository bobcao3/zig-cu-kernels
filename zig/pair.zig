//! Pairs of FP32 values. On `sm_100` and up each operation is one `f32x2` instruction, which
//! ptxas turns into one packed instruction on `sm_100` and `sm_103` (PTX ISA 9.7.3.1).

const std = @import("std");
const nvptx = std.nvptx;

pub const Pair = @Vector(2, f32);

const packed_instructions = nvptx.gpu.sm >= 100;

pub inline fn splat(x: f32) Pair {
    return @splat(x);
}

pub inline fn add(a: Pair, b: Pair) Pair {
    if (packed_instructions) return nvptx.math.addF32x2(.nearest_even, false, a, b);
    return .{ a[0] + b[0], a[1] + b[1] };
}

pub inline fn mul(a: Pair, b: Pair) Pair {
    if (packed_instructions) return nvptx.math.mulF32x2(.nearest_even, false, a, b);
    return .{ a[0] * b[0], a[1] * b[1] };
}

/// Returns `a * b + c`, rounded once.
pub inline fn fma(a: Pair, b: Pair, c: Pair) Pair {
    return fmaRounded(.nearest_even, a, b, c);
}

/// Returns `a * b + c`, rounded once with `rounding`.
pub inline fn fmaRounded(comptime rounding: nvptx.math.Rounding, a: Pair, b: Pair, c: Pair) Pair {
    if (packed_instructions) return nvptx.math.fmaF32x2(rounding, false, a, b, c);
    return .{
        nvptx.math.fmaF32(rounding, .{}, a[0], b[0], c[0]),
        nvptx.math.fmaF32(rounding, .{}, a[1], b[1], c[1]),
    };
}
