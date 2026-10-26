//! The value functions of the GEMM epilogues, which round to BF16 where the same PyTorch
//! operations on BF16 tensors round.

const std = @import("std");
const bf16 = @import("bf16.zig");
const cuda_math = @import("cuda_math.zig");

const Chunk = bf16.Chunk;

pub const Activation = enum(u32) { none, silu, gelu_tanh, gelu };

pub inline fn activate(comptime activation: Activation, x: f32) f32 {
    return switch (activation) {
        .none => x,
        .silu => cuda_math.silu(x),
        .gelu_tanh => cuda_math.geluTanh(x),
        .gelu => cuda_math.gelu(x),
    };
}

pub inline fn activateChunk(comptime activation: Activation, chunk: Chunk) Chunk {
    if (activation == .none) return chunk;
    var values = bf16.unpackChunk(chunk);
    inline for (&values) |*value| value.* = activate(activation, value.*);
    return bf16.packChunk(values);
}

pub inline fn gate(comptime activation: Activation, g: f32, u: f32) f32 {
    return bf16.round(activate(activation, g)) * u;
}

pub inline fn gateChunk(comptime activation: Activation, gates: Chunk, ups: Chunk) Chunk {
    const g = bf16.unpackChunk(gates);
    const u = bf16.unpackChunk(ups);
    var values: [8]f32 = undefined;
    inline for (&values, g, u) |*value, g_value, u_value| value.* = gate(activation, g_value, u_value);
    return bf16.packChunk(values);
}

pub inline fn addChunk(chunk: Chunk, other: Chunk) Chunk {
    const a = bf16.unpackChunk(chunk);
    const b = bf16.unpackChunk(other);
    var values: [8]f32 = undefined;
    inline for (&values, a, b) |*value, a_value, b_value| value.* = a_value + b_value;
    return bf16.packChunk(values);
}
