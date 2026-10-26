//! The kernels quantize BF16 or FP32 values to E4M3 with the scales of `fp8.zig`. Input row `r`
//! starts at element `r * input_stride`, a multiple of 16 bytes. The codes form a dense matrix.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const bf16 = @import("bf16.zig");
const fp8 = @import("fp8.zig");
const configurations = @import("fp8_gemm_configurations.zig");

const Chunk = bf16.Chunk;
const Codes = fp8.Codes;
const ScaleFormat = fp8.ScaleFormat;
const Element = configurations.Element;
const scale_group = configurations.scale_group;
const threads = configurations.quantize_threads;
const warps = threads / 32;
const scale_group_chunks = scale_group / 8;
const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = threads },
} };

inline fn load(
    comptime element: Element,
    input: [*]addrspace(.global) const Chunk,
    first: u64,
) [8]f32 {
    switch (element) {
        .bf16 => return bf16.unpackChunk(input[first / 8]),
        .f32 => {
            const low = input[first / 4];
            const high = input[first / 4 + 1];
            var values: [8]f32 = undefined;
            inline for (0..4) |i| {
                values[i] = @bitCast(low[i]);
                values[4 + i] = @bitCast(high[i]);
            }
            return values;
        },
    }
}

/// Scale group `g` of row `r` gets the scale `scales[g * scale_stride + r]`.
fn QuantizeRows(comptime element: Element, comptime format: ScaleFormat) type {
    return struct {
        fn kernel(
            input: [*]addrspace(.global) const Chunk,
            input_stride: u32,
            codes: [*]addrspace(.global) Codes,
            scales: [*]addrspace(.global) fp8.Scale(format),
            scale_stride: u32,
            rows: u32,
            width: u32,
        ) callconv(kernel_convention) void {
            const chunks = width / 8;
            const i = @workGroupId(0) * threads + @workItemId(0);
            const valid = i < rows * chunks;
            const row = @min(i / chunks, rows - 1);
            const chunk = i % chunks;
            const values = load(element, input, @as(u64, row) * input_stride + chunk * 8);
            const amax = fp8.maxNanOverLanes(fp8.amaxOf(values), scale_group_chunks);
            const scale = fp8.scaleOf(format, amax);
            if (!valid) return;
            codes[@as(u64, row) * chunks + chunk] = fp8.quantize(format, values, scale);
            if (chunk % scale_group_chunks == 0) {
                scales[@as(u64, chunk / scale_group_chunks) * scale_stride + row] = scale;
            }
        }
    };
}

/// Work group `i * width / scale_group + j` quantizes the rows below `rows` of the `scale_group` by
/// `scale_group` tile `(i, j)` and writes the scale of the tile to `scales[i * scale_stride + j]`.
fn QuantizeTiles(comptime element: Element, comptime format: ScaleFormat) type {
    return struct {
        const passes = scale_group * scale_group / 8 / threads;
        const rows_per_pass = threads / scale_group_chunks;
        var partials: [warps]f32 addrspace(.shared) = undefined;

        fn kernel(
            input: [*]addrspace(.global) const Chunk,
            input_stride: u32,
            codes: [*]addrspace(.global) Codes,
            scales: [*]addrspace(.global) fp8.Scale(format),
            scale_stride: u32,
            rows: u32,
            width: u32,
        ) callconv(kernel_convention) void {
            const tid = @workItemId(0);
            const tiles_k = width / scale_group;
            const tile_row = @workGroupId(0) / tiles_k;
            const tile_column = @workGroupId(0) % tiles_k;
            const first_row = tile_row * scale_group + tid / scale_group_chunks;
            const chunk = tile_column * scale_group_chunks + tid % scale_group_chunks;
            var values: [passes][8]f32 = undefined;
            var amax: f32 = 0;
            inline for (&values, 0..) |*pass_values, pass| {
                const row = first_row + pass * rows_per_pass;
                if (row < rows) {
                    pass_values.* = load(element, input, @as(u64, row) * input_stride + chunk * 8);
                    amax = fp8.maxNan(amax, fp8.amaxOf(pass_values.*));
                }
            }
            amax = fp8.maxNanOverLanes(amax, 32);
            if (tid % 32 == 0) partials[tid / 32] = amax;
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (0..warps) |w| amax = fp8.maxNan(amax, partials[w]);
            const scale = fp8.scaleOf(format, amax);
            inline for (values, 0..) |pass_values, pass| {
                const row = first_row + pass * rows_per_pass;
                if (row < rows) {
                    codes[@as(u64, row) * (width / 8) + chunk] =
                        fp8.quantize(format, pass_values, scale);
                }
            }
            if (tid == 0) scales[@as(u64, tile_row) * scale_stride + tile_column] = scale;
        }
    };
}

/// With a zero `amax` address, the kernel reads the scale from `scale`. Otherwise the kernel
/// computes the scale from the largest magnitude that `Amax` wrote to `amax` and writes the scale
/// to `scale`. In that case the work groups take the input in reverse order, because `Amax` reads
/// the end of the input last and the L2 cache may still hold that end.
fn QuantizeTensorwise(comptime element: Element) type {
    return struct {
        fn kernel(
            input: [*]addrspace(.global) const Chunk,
            input_stride: u32,
            codes: [*]addrspace(.global) Codes,
            rows: u32,
            width: u32,
            scale: [*]addrspace(.global) f32,
            amax: u64,
        ) callconv(kernel_convention) void {
            const chunks = width / 8;
            const block = if (amax == 0) @workGroupId(0) else nvptx.nctaid(0) - 1 - @workGroupId(0);
            const i = block * threads + @workItemId(0);
            const value_scale = if (amax == 0) scale[0] else dynamic: {
                const bits: *addrspace(.global) const u32 = @ptrFromInt(amax);
                const computed = fp8.scaleOf(.f32, @bitCast(bits.*));
                if (i == 0) scale[0] = computed;
                break :dynamic computed;
            };
            if (i >= rows * chunks) return;
            const row = i / chunks;
            const chunk = i % chunks;
            const values = load(element, input, @as(u64, row) * input_stride + chunk * 8);
            codes[i] = fp8.quantize(.f32, values, value_scale);
        }
    };
}

/// Raises the word at `amax` to the bits of the largest magnitude, compared as unsigned integers.
fn Amax(comptime element: Element) type {
    return struct {
        var partials: [warps]f32 addrspace(.shared) = undefined;

        fn kernel(
            input: [*]addrspace(.global) const Chunk,
            input_stride: u32,
            rows: u32,
            width: u32,
            amax: *addrspace(.global) u32,
        ) callconv(kernel_convention) void {
            const chunks = width / 8;
            const tid = @workItemId(0);
            const step = @as(u64, nvptx.nctaid(0)) * threads;
            var i = @as(u64, @workGroupId(0)) * threads + tid;
            var result: f32 = 0;
            while (i < @as(u64, rows) * chunks) : (i += step) {
                const index: u32 = @intCast(i);
                const first = @as(u64, index / chunks) * input_stride + index % chunks * 8;
                result = fp8.maxNan(result, fp8.amaxOf(load(element, input, first)));
            }
            result = fp8.maxNanOverLanes(result, 32);
            if (tid % 32 == 0) partials[tid / 32] = result;
            nvptx.barrierSyncAligned(.work_group, null);
            if (tid == 0) {
                inline for (0..warps) |w| result = fp8.maxNan(result, partials[w]);
                nvptx.red(.gpu, .monotonic, .max, amax, @as(u32, @bitCast(result)));
            }
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Element)) |element| {
            for (std.enums.values(ScaleFormat)) |format| {
                @export(&QuantizeRows(element, format).kernel, .{
                    .name = configurations.quantizerName(.rows, element, format),
                });
                @export(&QuantizeTiles(element, format).kernel, .{
                    .name = configurations.quantizerName(.tiles, element, format),
                });
            }
            @export(&QuantizeTensorwise(element).kernel, .{
                .name = configurations.quantizerName(.tensorwise, element, .f32),
            });
            @export(&Amax(element).kernel, .{
                .name = configurations.quantizerName(.amax, element, .f32),
            });
        }
    }
}
