//! Warpgroup matrix multiply-accumulate of FP8 operands in shared memory (PTX ISA 9.7.17.5). FP8
//! `wgmma` reads both operands K-major because it takes no transpose operands.

const std = @import("std");
const nvptx = std.nvptx;

pub const fence = nvptx.wgmma.fenceSyncAligned;
pub const commit = nvptx.wgmma.commitGroupSyncAligned;
pub const wait = nvptx.wgmma.waitGroupSyncAligned;

pub const Swizzle = enum(u2) {
    none = 0,
    bytes_128 = 1,
    bytes_64 = 2,
    bytes_32 = 3,

    pub fn fromBytes(comptime bytes: u32) Swizzle {
        return switch (bytes) {
            0 => .none,
            128 => .bytes_128,
            64 => .bytes_64,
            32 => .bytes_32,
            else => @compileError(std.fmt.comptimePrint("no swizzle of {d} bytes", .{bytes})),
        };
    }

    pub fn width(swizzle: Swizzle) u32 {
        return switch (swizzle) {
            .none => 0,
            .bytes_128 => 128,
            .bytes_64 => 64,
            .bytes_32 => 32,
        };
    }
};

/// The shared memory matrix descriptor of PTX ISA 9.7.17.5.1.2.2.
pub const Descriptor = packed struct(u64) {
    start: u14,
    reserved0: u2 = 0,
    leading_byte_offset: u14,
    reserved1: u2 = 0,
    stride_byte_offset: u14,
    reserved2: u3 = 0,
    base_offset: u3 = 0,
    reserved3: u10 = 0,
    swizzle: Swizzle,

    pub inline fn init(
        start: u32,
        leading_byte_offset: u32,
        stride_byte_offset: u32,
        swizzle: Swizzle,
    ) Descriptor {
        const fields = @as(u64, encode(leading_byte_offset)) << 16 |
            @as(u64, encode(stride_byte_offset)) << 32 |
            @as(u64, @backingInt(swizzle)) << 62;
        return @bitCast(fields | encode(start));
    }

    /// Assumes that the tile starts where the swizzle pattern repeats, at a multiple of 8 rows, so
    /// that the base offset is 0.
    pub inline fn kMajor(start: u32, comptime swizzle: Swizzle) Descriptor {
        if (swizzle == .none) @compileError("a K-major descriptor without swizzle needs init");
        const fields = comptime Descriptor.init(0, 16, 8 * swizzle.width(), swizzle).bits();
        return @bitCast(fields | encode(start));
    }

    /// Assumes that the new start address stays below 2^18 bytes, the range of the start field.
    pub inline fn advance(descriptor: Descriptor, bytes: u32) Descriptor {
        return @bitCast(descriptor.bits() + (bytes >> 4));
    }

    /// Makes the same assumption as `advance`, under which the sum cannot carry out of the low
    /// 32 bits, and adds in 32 bits.
    pub inline fn advanceLow(descriptor: Descriptor, bytes: u32) Descriptor {
        const low: u32 = @truncate(descriptor.bits());
        const high: u32 = @truncate(descriptor.bits() >> 32);
        return @bitCast(@as(u64, high) << 32 | (low + (bytes >> 4)));
    }

    pub inline fn bits(descriptor: Descriptor) u64 {
        return @bitCast(descriptor);
    }

    inline fn encode(bytes: u32) u14 {
        return @truncate((bytes & 0x3ffff) >> 4);
    }
};

/// Returns the byte offset of the K step `k_step` of 32 bytes in a tile of `rows` rows that the
/// tensor copy stores as consecutive boxes, each `swizzle` bytes wide.
pub fn kStepOffset(comptime swizzle: Swizzle, comptime rows: u32, comptime k_step: u32) u32 {
    const steps_per_box = swizzle.width() / 32;
    return k_step / steps_per_box * rows * swizzle.width() + k_step % steps_per_box * 32;
}

pub const Fp8 = enum { e4m3, e5m2 };

pub fn Accumulator(comptime n: u32) type {
    return @Vector(n / 2, f32);
}

pub inline fn fenceOperand(accumulator: anytype) void {
    var value = accumulator.*;
    asm volatile (""
        : [value] "+f" (value),
    );
    accumulator.* = value;
}

/// The product replaces the accumulator when `accumulate` is false.
pub inline fn mma(
    comptime n: u32,
    comptime a_type: Fp8,
    comptime b_type: Fp8,
    accumulator: *Accumulator(n),
    a: Descriptor,
    b: Descriptor,
    accumulate: bool,
) void {
    nvptx.require("wgmma.mma_async", .{
        .ptx = 80,
        .sm = .{ .arch_specific = 90 },
        .threads = .warpgroup,
    });
    if (n < 8 or n > 256 or n % 8 != 0) {
        @compileError(std.fmt.comptimePrint("wgmma has no FP8 shape with n = {d}", .{n}));
    }
    const text = std.fmt.comptimePrint(
        "wgmma.mma_async.sync.aligned.m64n{d}k32.f32.{t}.{t} " ++
            "%[d], %[a], %[b], %[accumulate], 1, 1;",
        .{ n, a_type, b_type },
    );
    var value = accumulator.*;
    asm volatile (text
        : [d] "+f" (value),
        : [a] "l" (a.bits()),
          [b] "l" (b.bits()),
          [accumulate] "b" (accumulate),
        : .{ .memory = true });
    accumulator.* = value;
}

pub const Advance = enum { wide, low };

pub inline fn mmaKSteps(
    comptime n: u32,
    comptime k_steps: u32,
    comptime swizzle: Swizzle,
    comptime a_rows: u32,
    comptime b_rows: u32,
    accumulator: *Accumulator(n),
    a: Descriptor,
    b: Descriptor,
    accumulate: bool,
) void {
    mmaKStepsAdvancing(.wide, n, k_steps, swizzle, a_rows, b_rows, accumulator, a, b, accumulate);
}

pub inline fn mmaKStepsAdvancing(
    comptime advance: Advance,
    comptime n: u32,
    comptime k_steps: u32,
    comptime swizzle: Swizzle,
    comptime a_rows: u32,
    comptime b_rows: u32,
    accumulator: *Accumulator(n),
    a: Descriptor,
    b: Descriptor,
    accumulate: bool,
) void {
    inline for (0..k_steps) |k_step| {
        const a_offset = comptime kStepOffset(swizzle, a_rows, k_step);
        const b_offset = comptime kStepOffset(swizzle, b_rows, k_step);
        mma(
            n,
            .e4m3,
            .e4m3,
            accumulator,
            if (advance == .low) a.advanceLow(a_offset) else a.advance(a_offset),
            if (advance == .low) b.advanceLow(b_offset) else b.advance(b_offset),
            if (k_step == 0) accumulate else true,
        );
    }
}

test "the descriptor fields sit at the bits of the descriptor format" {
    const descriptor: Descriptor = .init(0x1_2340, 16, 1024, .bytes_128);
    try std.testing.expectEqual(@as(u64, 0x1234), descriptor.bits() & 0x3fff);
    try std.testing.expectEqual(@as(u64, 1), descriptor.bits() >> 16 & 0x3fff);
    try std.testing.expectEqual(@as(u64, 64), descriptor.bits() >> 32 & 0x3fff);
    try std.testing.expectEqual(@as(u64, 1), descriptor.bits() >> 62);
    var with_base = Descriptor.kMajor(0, .bytes_64);
    with_base.base_offset = 5;
    try std.testing.expectEqual(@as(u64, 5), with_base.bits() >> 49 & 7);
    try std.testing.expectEqual(@as(u64, 2), with_base.bits() >> 62);
    try std.testing.expectEqual(@as(u64, 32), with_base.bits() >> 32 & 0x3fff);
}

test "a K step advances the start address in units of 16 bytes" {
    const base: Descriptor = .kMajor(0x8000, .bytes_128);
    const step = base.advance(kStepOffset(.bytes_128, 128, 3));
    try std.testing.expectEqual(@as(u64, (0x8000 + 96) >> 4), step.bits() & 0x3fff);
    try std.testing.expectEqual(base.bits() >> 14, step.bits() >> 14);
    try std.testing.expectEqual(@as(u32, 64 * 64 + 32), kStepOffset(.bytes_64, 64, 3));
    try std.testing.expectEqual(step, base.advanceLow(kStepOffset(.bytes_128, 128, 3)));
}
