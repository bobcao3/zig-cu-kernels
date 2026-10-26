const std = @import("std");
const nvptx = std.nvptx;

/// Eight BF16 values fill the 16 bytes of one vector load, `cp.async` copy or `ldmatrix` row.
pub const Chunk = @Vector(4, u32);

pub inline fn toF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// The conversion rounds `x` to the nearest BF16 value, ties to even, as `cvt.rn.bf16.f32` does.
pub inline fn fromF32(x: f32) u16 {
    return nvptx.bf16FromF32(x);
}

pub inline fn round(x: f32) f32 {
    return toF32(fromF32(x));
}

pub inline fn unpack(pair: u32) @Vector(2, f32) {
    return nvptx.Bf16x2.toF32(@fromBackingInt(pair));
}

pub inline fn pack(x0: f32, x1: f32) u32 {
    return @backingInt(nvptx.Bf16x2.fromF32(x0, x1));
}

pub inline fn unpackChunk(chunk: Chunk) [8]f32 {
    var values: [8]f32 = undefined;
    inline for (0..4) |word| {
        const pair = unpack(chunk[word]);
        values[2 * word] = pair[0];
        values[2 * word + 1] = pair[1];
    }
    return values;
}

pub inline fn packChunk(values: [8]f32) Chunk {
    var chunk: Chunk = undefined;
    inline for (0..4) |word| chunk[word] = pack(values[2 * word], values[2 * word + 1]);
    return chunk;
}

/// Mixed-precision `fma.rn.f32.bf16` exists on sm_100 and up (PTX ISA 9.7.5.3).
pub const mixed_fma = nvptx.gpu.sm >= 100;

/// Returns the product of the BF16 values in half `half` of `a` and of `b`, plus `c`, rounded once
/// to FP32. The product of two BF16 values is exact. Needs `mixed_fma`.
pub inline fn fmaHalves(comptime half: u1, a: u32, b: u32, c: f32) f32 {
    nvptx.require("fma.rn.f32.bf16", .{ .ptx = 86, .sm = .{ .base = 100 }, .threads = .thread });
    const operands = if (half == 0) "a0, b0" else "a1, b1";
    return asm ("{\n.reg .b16 a0, a1, b0, b1;\nmov.b32 {a0, a1}, %[a];\nmov.b32 {b0, b1}, %[b];\n" ++
            "fma.rn.f32.bf16 %[d], " ++ operands ++ ", %[c];\n}"
        : [d] "=f" (-> f32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "f" (c),
    );
}
