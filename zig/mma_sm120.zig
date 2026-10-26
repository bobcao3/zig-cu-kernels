//! The FP8 `mma.sync` forms of compute capability 12.x (PTX ISA 9.7.16.5.14, target notes).

const std = @import("std");
const nvptx = std.nvptx;

pub const A = @Vector(4, u32);
pub const B = @Vector(2, u32);
pub const Accumulator = @Vector(4, f32);
pub const BlockScaleOptions = nvptx.matrix.BlockScaleOptions;

const e4m3: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k32,
    .accumulator = .f32,
    .element = .e4m3,
};

pub inline fn mma(a: A, b: B, c: Accumulator) Accumulator {
    nvptx.require("mma.sync.aligned.kind::f8f6f4", .{
        .ptx = 88,
        .sm = .{ .family = &.{ 120, 121 } },
        .threads = .warp,
    });
    return asm volatile ("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 " ++
            "%[d], %[a], %[b], %[c];"
        : [d] "=f" (-> Accumulator),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "f" (c),
    );
}

/// `options` selects the lanes and E8M0 bytes that scale A and B (PTX ISA 9.7.16.3).
pub inline fn mmaBlockScaled(
    comptime options: BlockScaleOptions,
    a: A,
    b: B,
    c: Accumulator,
    scale_a: u32,
    scale_b: u32,
) Accumulator {
    return nvptx.matrix.mmaSyncBlockScale(e4m3, options, a, b, c, scale_a, scale_b);
}
