//! The FP8 kernels divide the amax of the values that share a scale by 448 to derive the scale.

const std = @import("std");
const nvptx = std.nvptx;
const warp = @import("warp.zig");
const bf16 = @import("bf16.zig");
const pair = @import("pair.zig");

pub const e4m3_max = 448;

pub const ScaleFormat = enum { f32, e8m0 };

pub const Codes = @Vector(2, u32);

const mantissa_bits = 23;
const mantissa_mask: u32 = (1 << mantissa_bits) - 1;
const sign_mask: u32 = 1 << 31;
const infinity_bits: u32 = 0x7f80_0000;
const canonical_nan_bits: u32 = 0x7fff_ffff;
const smallest_e8m0_bits: u32 = 1 << (mantissa_bits - 1);
const e8m0_one = 127;
const e8m0_max = 254;
const e8m0_nan = 255;

pub fn Scale(comptime format: ScaleFormat) type {
    return switch (format) {
        .f32 => f32,
        .e8m0 => u8,
    };
}

/// Unlike `@max`, returns NaN for a NaN operand, as `torch.amax` does (PTX ISA 9.7.3.12).
pub inline fn maxNan(a: f32, b: f32) f32 {
    return asm ("max.NaN.f32 %[d], %[a], %[b];"
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

pub inline fn maxNanOverLanes(x: f32, comptime lanes: u32) f32 {
    var result = x;
    comptime var mask = lanes / 2;
    inline while (mask > 0) : (mask /= 2) result = maxNan(result, warp.shuffleXor(result, mask));
    return result;
}

pub inline fn amaxOf(values: anytype) f32 {
    var amax: f32 = 0;
    inline for (values) |value| amax = maxNan(amax, @abs(value));
    return amax;
}

pub inline fn scaleOf(comptime format: ScaleFormat, amax: f32) Scale(format) {
    const quotient = nvptx.math.divF32(.nearest_even, false, amax, e4m3_max);
    return switch (format) {
        .f32 => if (quotient == 0) 1 else quotient,
        .e8m0 => e8m0Of(quotient),
    };
}

/// Returns the byte of the least E8M0 value at or above a positive finite `quotient`, at most 254.
pub inline fn e8m0Of(quotient: f32) u8 {
    const magnitude = @as(u32, @bitCast(quotient)) & ~sign_mask;
    const exponent_rounded_up = @min((magnitude + mantissa_mask) >> mantissa_bits, e8m0_max);
    var byte: u32 = if (magnitude <= smallest_e8m0_bits) 0 else exponent_rounded_up;
    byte = if (magnitude == 0) e8m0_one else byte;
    byte = if (magnitude >= infinity_bits) e8m0_nan else byte;
    return @intCast(byte);
}

/// Returns `e8m0Of(amax / 448)` for an `amax` with a BF16 value, without the division. With
/// `amax = M * 2^(E - 127)`, the quotient is `(M / 1.75) * 2^(E - 135)`, so the least power of two at
/// or above it is `2^(E - 135)` when `M <= 1.75` and `2^(E - 134)` otherwise. No BF16 significand
/// lies close enough to 1.75 for the rounding of the quotient to change that. A comparison with
/// `e8m0Of` of the FP32 quotient over every BF16 value found no difference.
pub inline fn e8m0OfBf16Amax(amax: f32) u8 {
    const magnitude = @as(u32, @bitCast(amax)) & ~sign_mask;
    const exponent = magnitude >> mantissa_bits;
    const rounded = exponent + @intFromBool(magnitude & mantissa_mask > 0x60_0000);
    var byte: u32 = if (rounded > 8) rounded - 8 else 0;
    byte = if (magnitude == 0) e8m0_one else byte;
    byte = if (magnitude >= infinity_bits) e8m0_nan else byte;
    return @intCast(byte);
}

pub inline fn e8m0ToF32(byte: u8) f32 {
    var bits: u32 = if (byte == 0) smallest_e8m0_bits else @as(u32, byte) << mantissa_bits;
    bits = if (byte == e8m0_nan) canonical_nan_bits else bits;
    return @bitCast(bits);
}

/// The product `x * 2^(127 - byte)` rounds the exact quotient `x / 2^(byte - 127)`, and
/// `cvt.rn.satfinite.e4m3x2.f32` clamps magnitudes above 448 to 448 (PTX ISA 9.7.10.24).
pub inline fn quantize(comptime format: ScaleFormat, values: [8]f32, scale: Scale(format)) Codes {
    var quotients: [8]f32 = undefined;
    switch (format) {
        .f32 => inline for (&quotients, values) |*quotient, value| {
            quotient.* = nvptx.math.divF32(.nearest_even, false, value, scale);
        },
        .e8m0 => {
            const factor = e8m0ToF32(if (scale == e8m0_nan) e8m0_nan else e8m0_max - scale);
            inline for (&quotients, values) |*quotient, value| {
                quotient.* = nvptx.math.mulF32(.nearest_even, .{}, value, factor);
            }
        },
    }
    var codes: Codes = undefined;
    inline for (0..2) |word| {
        const four = quotients[word * 4 ..][0..4];
        codes[word] = @backingInt(nvptx.E4m3x4.fromF32(four[0], four[1], four[2], four[3]));
    }
    return codes;
}

/// Returns the largest magnitude of the BF16 values in the words of `words`, or NaN when one of
/// them is NaN. `max.NaN.xorsign.abs.bf16x2` returns the larger magnitude of each pair of halves
/// with the XOR of their signs, or a canonical NaN (PTX ISA 9.7.4.8), so only the sign of the
/// result needs clearing.
/// Returns the larger magnitudes of the BF16 halves of `word` over groups of `lanes` lanes, with
/// garbage sign bits, or canonical NaN halves.
pub inline fn maxMagnitudesOverLanes(word: u32, comptime lanes: u32) u32 {
    var result = word;
    comptime var mask = lanes / 2;
    inline while (mask > 0) : (mask /= 2) {
        result = maxMagnitudes(result, @bitCast(warp.shuffleXor(@bitCast(result), mask)));
    }
    return result;
}

pub inline fn amaxOfBf16(words: anytype) f32 {
    const count = @typeInfo(@TypeOf(words)).vector.len;
    var amax: u32 = words[0];
    inline for (1..count) |index| amax = maxMagnitudes(amax, words[index]);
    amax = maxMagnitudes(amax, amax >> 16 | amax << 16);
    return @bitCast(amax & 0x7fff_0000);
}

inline fn maxMagnitudes(a: u32, b: u32) u32 {
    nvptx.require("max.NaN.xorsign.abs.bf16x2", .{ .ptx = 72, .sm = .{ .base = 86 }, .threads = .thread });
    return asm ("max.NaN.xorsign.abs.bf16x2 %[d], %[a], %[b];"
        : [d] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
    );
}

/// Divides by a scale with the result of `div.rn.f32`. For a normal finite scale, the quotient
/// `q` of the correctly rounded reciprocal and the remainder `x - q * scale`, which one fma
/// computes exactly, give the correctly rounded quotient in one more fma (Markstein's theorem,
/// Handbook of Floating-Point Arithmetic, section 4.7). The remainder rounds toward negative
/// infinity, which changes no exact result but makes an exact cancellation -0, so that the
/// quotient of a zero keeps its sign. A remainder near the subnormal range can be inexact, so a
/// scale below 2^-64 and `x` are both multiplied by 2^100 first, which changes neither the quotient
/// nor its rounding. With `|x| <= 448 * scale`, only a subnormal `x` then yields an inexact
/// remainder, and its quotient below 2^-62 converts to an FP8 zero of the sign of `x` either way.
/// For an infinite or NaN scale, `x` times the reciprocal (0 or NaN) is the quotient.
pub const Divisor = struct {
    negative_scale: f32,
    reciprocal: f32,
    prescale: f32,
    finite: bool,

    const tiny = 0x1p-64;
    const lift = 0x1p100;

    pub inline fn init(scale: f32) Divisor {
        return withReciprocal(scale, reciprocalOf(scale));
    }

    /// Returns the reciprocal that `init(scale)` computes.
    pub inline fn reciprocalOf(scale: f32) f32 {
        const prescale: f32 = if (scale < tiny) lift else 1;
        return nvptx.math.rcpF32(.nearest_even, false, scale * prescale);
    }

    /// Returns `init(scale)` for the reciprocal `reciprocalOf(scale)` computed elsewhere.
    pub inline fn withReciprocal(scale: f32, reciprocal: f32) Divisor {
        const prescale: f32 = if (scale < tiny) lift else 1;
        return .{
            .negative_scale = -(scale * prescale),
            .reciprocal = reciprocal,
            .prescale = prescale,
            .finite = scale <= std.math.floatMax(f32),
        };
    }

    /// Returns the quotients of the pairs of `x`, choosing the method once for all of them.
    pub inline fn divideAll(divisor: Divisor, comptime count: usize, x: [count]pair.Pair) [count]pair.Pair {
        var quotients: [count]pair.Pair = undefined;
        const reciprocal = pair.splat(divisor.reciprocal);
        if (!divisor.finite) {
            inline for (&quotients, x) |*quotient, values| quotient.* = pair.mul(values, reciprocal);
        } else if (divisor.prescale != 1) {
            const prescale = pair.splat(divisor.prescale);
            inline for (&quotients, x) |*quotient, values| {
                quotient.* = divisor.correctedQuotient(pair.mul(values, prescale));
            }
        } else {
            inline for (&quotients, x) |*quotient, values| quotient.* = divisor.correctedQuotient(values);
        }
        return quotients;
    }

    /// Returns the quotients of the pairs of `x` for any `x`, also when a quotient exceeds the FP32
    /// range: where the uncorrected quotient is infinite, the correction would give NaN, so the
    /// infinite quotient stays, which `cvt.rn.satfinite` clamps to 448 as it clamps the infinite
    /// result of `div.rn.f32`.
    pub inline fn divideAllUnbounded(divisor: Divisor, comptime count: usize, x: [count]pair.Pair) [count]pair.Pair {
        if (!divisor.finite) return divisor.divideAll(count, x);
        const prescale = pair.splat(divisor.prescale);
        const reciprocal = pair.splat(divisor.reciprocal);
        var quotients: [count]pair.Pair = undefined;
        inline for (&quotients, x) |*quotient, values| {
            const lifted = pair.mul(values, prescale);
            const corrected = divisor.correctedQuotient(lifted);
            const first = pair.mul(lifted, reciprocal);
            inline for (0..2) |lane| {
                quotient[lane] = if (@abs(first[lane]) == std.math.inf(f32)) first[lane] else corrected[lane];
            }
        }
        return quotients;
    }

    /// Whether `divideAll` divides by this divisor with `correctedQuotient` alone.
    inline fn isPlain(divisor: Divisor) bool {
        return divisor.finite and divisor.prescale == 1;
    }

    /// Returns the codes of the quotients of the BF16 values of `words` for a divisor that
    /// `isPlain` accepts.
    inline fn plainCodes(divisor: Divisor, words: bf16.Chunk) Codes {
        var quotients: [4]pair.Pair = undefined;
        inline for (&quotients, 0..) |*quotient, word| {
            quotient.* = divisor.correctedQuotient(bf16.unpack(words[word]));
        }
        return packCodes(quotients);
    }

    inline fn correctedQuotient(divisor: Divisor, x: pair.Pair) pair.Pair {
        const reciprocal = pair.splat(divisor.reciprocal);
        const quotient = pair.mul(x, reciprocal);
        const negative_scale = pair.splat(divisor.negative_scale);
        const remainder = pair.fmaRounded(.toward_negative_infinity, quotient, negative_scale, x);
        return pair.fma(remainder, reciprocal, quotient);
    }
};

/// Quantizes eight BF16 values with a tensorwise scale, which bounds no quotient, as `quantize`
/// quantizes their FP32 values.
pub const TensorwiseQuantizer = struct {
    divisor: Divisor,

    pub inline fn init(scale: f32) TensorwiseQuantizer {
        return .{ .divisor = .init(scale) };
    }

    /// Whether `plainCodes` gives the codes of values of at most the magnitude `amax`. With
    /// `amax` times the reciprocal finite, every uncorrected quotient is finite, so the correction
    /// gives no NaN, and a corrected quotient that rounds to infinity converts to 448 as every
    /// quotient above 448 does.
    pub inline fn isPlainUpTo(quantizer: TensorwiseQuantizer, amax: f32) bool {
        const divisor = quantizer.divisor;
        return divisor.isPlain() and amax * divisor.reciprocal <= std.math.floatMax(f32);
    }

    /// Returns `codes(words)` for values that `isPlainUpTo` accepts.
    pub inline fn plainCodes(quantizer: TensorwiseQuantizer, words: bf16.Chunk) Codes {
        return quantizer.divisor.plainCodes(words);
    }

    pub inline fn codes(quantizer: TensorwiseQuantizer, words: bf16.Chunk) Codes {
        var values: [4]pair.Pair = undefined;
        inline for (&values, 0..) |*value, word| value.* = bf16.unpack(words[word]);
        return packCodes(quantizer.divisor.divideAllUnbounded(4, values));
    }
};

/// Quantizes eight BF16 values with one scale, as `quantize` quantizes their FP32 values.
pub fn Quantizer(comptime format: ScaleFormat) type {
    return struct {
        divisor: if (format == .f32) Divisor else void,
        /// The FP32 factor, or with `bf16.mixed_fma` the BF16 factor in both halves of a word.
        factor: if (format == .e8m0) u32 else void,

        const Self = @This();

        /// The part of a quantizer that a work item computes from the scale and that other work
        /// items of the scale group can take with a shuffle: the reciprocal of an FP32 scale or the
        /// factor of an E8M0 scale.
        pub inline fn sharedPart(scale: Scale(format)) u32 {
            return switch (format) {
                .f32 => @bitCast(Divisor.reciprocalOf(scale)),
                .e8m0 => init(scale).factor,
            };
        }

        /// Returns `init(scale)` from the result of `sharedPart(scale)`.
        pub inline fn fromShared(scale: Scale(format), shared: u32) Self {
            return switch (format) {
                .f32 => .{ .divisor = .withReciprocal(scale, @bitCast(shared)), .factor = {} },
                .e8m0 => .{ .divisor = {}, .factor = shared },
            };
        }

        pub inline fn init(scale: Scale(format)) Self {
            return switch (format) {
                .f32 => .{ .divisor = .init(scale), .factor = {} },
                .e8m0 => .{ .divisor = {}, .factor = if (bf16.mixed_fma) bf16FactorOf(scale) else @bitCast(
                    e8m0ToF32(if (scale == e8m0_nan) e8m0_nan else e8m0_max - scale),
                ) },
            };
        }

        /// Whether `plainCodes` gives the codes of this quantizer.
        pub inline fn isPlain(quantizer: Self) bool {
            return switch (format) {
                .f32 => quantizer.divisor.isPlain(),
                .e8m0 => true,
            };
        }

        /// Returns `codes(words)` for a quantizer that `isPlain` accepts, without the branches that
        /// choose the division method.
        pub inline fn plainCodes(quantizer: Self, words: bf16.Chunk) Codes {
            return switch (format) {
                .f32 => quantizer.divisor.plainCodes(words),
                .e8m0 => quantizer.codes(words),
            };
        }

        pub inline fn codes(quantizer: Self, words: bf16.Chunk) Codes {
            if (format == .e8m0 and bf16.mixed_fma) return quantizer.powerOfTwoCodes(words);
            var values: [4]pair.Pair = undefined;
            inline for (&values, 0..) |*value, word| value.* = bf16.unpack(words[word]);
            var quotients: [4]pair.Pair = undefined;
            switch (format) {
                .f32 => quotients = quantizer.divisor.divideAll(4, values),
                .e8m0 => inline for (&quotients, values) |*quotient, value| {
                    quotient.* = pair.mul(value, pair.splat(@bitCast(quantizer.factor)));
                },
            }
            return packCodes(quotients);
        }

        /// An E8M0 factor is a power of two, which BF16 represents exactly, so one mixed-precision
        /// fma per value gives the exact product with the BF16 value. The addend -0 leaves every
        /// product unchanged, a zero product included.
        inline fn powerOfTwoCodes(quantizer: Self, words: bf16.Chunk) Codes {
            const factor = quantizer.factor;
            var quotients: [4]pair.Pair = undefined;
            inline for (&quotients, 0..) |*quotient, word| quotient.* = .{
                bf16.fmaHalves(0, words[word], factor, -0.0),
                bf16.fmaHalves(1, words[word], factor, -0.0),
            };
            return packCodes(quotients);
        }

        /// The BF16 bits of `2^(127 - byte)` in both halves of a word: the factor that divides by the
        /// E8M0 scale `byte`, as `e8m0ToF32` gives it in FP32.
        inline fn bf16FactorOf(byte: u8) u32 {
            const exponent = @as(u32, e8m0_max) -% byte;
            var bits: u32 = if (exponent == 0) 0x0040 else exponent << 7;
            bits = if (byte == e8m0_nan) 0x7fff else bits;
            return bits * 0x0001_0001;
        }
    };
}

inline fn packCodes(quotients: [4]pair.Pair) Codes {
    var result: Codes = undefined;
    inline for (0..2) |half| {
        const low = quotients[2 * half];
        const high = quotients[2 * half + 1];
        result[half] = @backingInt(nvptx.E4m3x4.fromF32(low[0], low[1], high[0], high[1]));
    }
    return result;
}
