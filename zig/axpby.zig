//! The kernels round `beta * y` to FP32 with `mul.rn.f32`, add `alpha * x` with one rounding by
//! `fma.rn.f32` and round a BF16 result with `cvt.rn` (PTX ISA 9.7.10.24). The code optimizer does
//! not fuse a `mul` that names its rounding (PTX ISA 9.7.3.5). `out` may be exactly `x` or `y`.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const bf16 = @import("bf16.zig");
const configurations = @import("axpby_configurations.zig");

const Chunk = bf16.Chunk;
const Element = configurations.Element;
const threads = configurations.threads;
const chunks_per_thread = configurations.chunks_per_thread;

fn Axpby(comptime element: Element) type {
    return struct {
        const Scalar = switch (element) {
            .bf16 => u16,
            .f32 => f32,
        };
        const values_per_chunk = @sizeOf(Chunk) / @sizeOf(Scalar);

        /// Assumes 16-byte aligned `out`, `x` when `alpha != 0` and `y` when `beta != 0` (PTX ISA
        /// 6.4.1), and a grid of at least one work group that covers every chunk.
        fn kernel(
            x: [*]addrspace(.global) const Chunk,
            y: [*]addrspace(.global) const Chunk,
            out: [*]addrspace(.global) Chunk,
            count: u64,
            alpha: f32,
            beta: f32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = threads },
        } }) void {
            const chunks = count / values_per_chunk;
            const first = @as(u64, @workGroupId(0)) * threads * chunks_per_thread + @workItemId(0);
            var xs: [chunks_per_thread]Chunk = @splat(@splat(0));
            var ys: [chunks_per_thread]Chunk = @splat(@splat(0));
            inline for (&xs, &ys, 0..) |*x_chunk, *y_chunk, i| {
                const index = first + i * threads;
                if (index < chunks) {
                    if (alpha != 0) x_chunk.* = x[index];
                    if (beta != 0) y_chunk.* = y[index];
                }
            }
            inline for (xs, ys, 0..) |x_chunk, y_chunk, i| {
                const index = first + i * threads;
                if (index < chunks) out[index] = combineChunk(x_chunk, y_chunk, alpha, beta);
            }
            if (@workGroupId(0) + 1 == nvptx.nctaid(0)) {
                const tail = chunks * values_per_chunk + @workItemId(0);
                if (tail < count) {
                    const x_values: [*]addrspace(.global) const Scalar = @ptrCast(x);
                    const y_values: [*]addrspace(.global) const Scalar = @ptrCast(y);
                    const out_values: [*]addrspace(.global) Scalar = @ptrCast(out);
                    const x_value = if (alpha != 0) toF32(x_values[tail]) else 0;
                    const y_value = if (beta != 0) toF32(y_values[tail]) else 0;
                    out_values[tail] = fromF32(combine(x_value, y_value, alpha, beta));
                }
            }
        }

        inline fn toF32(value: Scalar) f32 {
            return if (element == .bf16) bf16.toF32(value) else value;
        }

        inline fn fromF32(value: f32) Scalar {
            return if (element == .bf16) bf16.fromF32(value) else value;
        }

        inline fn combine(x: f32, y: f32, alpha: f32, beta: f32) f32 {
            return nvptx.math.fmaF32(.nearest_even, .{}, alpha, x, beta * y);
        }

        inline fn combineChunk(x: Chunk, y: Chunk, alpha: f32, beta: f32) Chunk {
            if (element == .f32) {
                var result: Chunk = undefined;
                inline for (0..4) |i| {
                    result[i] = @bitCast(combine(@bitCast(x[i]), @bitCast(y[i]), alpha, beta));
                }
                return result;
            }
            const x_values = bf16.unpackChunk(x);
            const y_values = bf16.unpackChunk(y);
            var result: [8]f32 = undefined;
            inline for (&result, x_values, y_values) |*value, x_value, y_value| {
                value.* = combine(x_value, y_value, alpha, beta);
            }
            return bf16.packChunk(result);
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Element)) |element| {
            @export(&Axpby(element).kernel, .{ .name = configurations.kernelName(element) });
        }
    }
}
