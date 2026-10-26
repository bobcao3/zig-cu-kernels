//! The kernels compute the GLU product `h` of a BF16 matrix `Z = [lo | hi]` and write the FP8 pack
//! of `h` by the rules of `tests/pack_references.py::glu_h_quant`, or the BF16 `h` itself for the
//! rowwise pack. A work item loads 16 consecutive values of each half with 16-byte loads.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const math = nvptx.math;
const bf16 = @import("bf16.zig");
const fp8 = @import("fp8.zig");
const fp8_pack = @import("fp8_pack.zig");
const configurations = @import("glu_quant_configurations.zig");

const Activation = configurations.Activation;
const Pack = configurations.Pack;
const ScaleFormat = configurations.ScaleFormat;
const Words = @Vector(4, u32);
const Values = [chunk_values]f32;
const chunk_values = configurations.chunk_values;
const threads = configurations.threads;
const warps = threads / 32;
const row_chunks = configurations.row_chunks;
const atom_rows = 128;
const atom_blocks = 4;
const atom_bytes = 512;
const negative_log2e: f32 = @bitCast(@as(u32, 0xbfb8_aa3b));
const gelu_linear: f32 = @bitCast(@as(u32, 0x4013_5761));
const gelu_cubic: f32 = @bitCast(@as(u32, 0x3dd2_d3e8));
const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = threads },
} };

inline fn load(input: [*]addrspace(.global) const Words, first: u64) Values {
    var values: Values = undefined;
    inline for (0..2) |half| {
        const eight = bf16.unpackChunk(input[first / 8 + half]);
        inline for (0..8) |i| values[half * 8 + i] = eight[i];
    }
    return values;
}

/// The sigmoid of the clamped gate `g` is `rcp.approx.ftz(1 + ex2.approx.ftz(-log2(e)·g))` as in
/// `triton_ops/fast_math.py` (PTX ISA 9.7.3.21, 9.7.3.13).
inline fn swiglu(up: f32, gate: f32, limit: f32) f32 {
    const g = @min(gate, limit);
    const u = @max(@min(up, limit), -limit);
    const exponent = math.mulF32(.nearest_even, .{ .ftz = true }, g, negative_log2e);
    const power = math.ex2ApproxF32(true, exponent);
    const sigmoid = math.rcpApproxF32(true, math.addF32(.nearest_even, .{ .ftz = true }, power, 1));
    const silu = bf16.round(math.mulF32(.nearest_even, .{}, g, sigmoid));
    return bf16.round(math.mulF32(.nearest_even, .{}, u, silu));
}

/// The tanh GELU computes `tanh(|t|)` as `1 - 2 / (2^(2·log2(e)·|t|) + 1)` with
/// `t = sqrt(2/π)·(gate + 0.044715·gate³)`, the formula of `triton_ops/geglu_tanh.py`.
inline fn gegluTanh(up: f32, gate: f32) f32 {
    const square = math.mulF32(.nearest_even, .{}, gate, gate);
    const inner = math.fmaF32(.nearest_even, .{}, gelu_cubic, square, gelu_linear);
    const power = math.ex2ApproxF32(true, math.mulF32(.nearest_even, .{}, @abs(gate), inner));
    const reciprocal = math.rcpApproxF32(true, math.addF32(.nearest_even, .{}, power, 1));
    const magnitude = math.fmaF32(.nearest_even, .{}, reciprocal, -2, 1);
    const sign: f32 = if (gate < 0) -0.5 else 0.5;
    const half = math.fmaF32(.nearest_even, .{}, sign, magnitude, 0.5);
    const gelu = bf16.round(math.mulF32(.nearest_even, .{}, gate, half));
    return bf16.round(math.mulF32(.nearest_even, .{}, gelu, up));
}

/// Row `r` of `z` holds the `padded` columns of `lo` and then the `padded` columns of `hi`.
const Glu = struct {
    z: [*]addrspace(.global) const Words,
    padded: u32,
    hidden: u32,
    gate_first: bool,
    limit: f32,

    inline fn products(glu: Glu, comptime activation: Activation, row: u32, column: u32) Values {
        const row_first = @as(u64, row) * 2 * glu.padded;
        const lo = load(glu.z, row_first + column);
        const hi = load(glu.z, row_first + glu.padded + column);
        const up = if (glu.gate_first) hi else lo;
        const gate = if (glu.gate_first) lo else hi;
        var h: Values = undefined;
        inline for (&h, up, gate, 0..) |*value, u, g, j| {
            const product = switch (activation) {
                .swiglu => swiglu(u, g, glu.limit),
                .geglu_tanh => gegluTanh(u, g),
            };
            value.* = if (column + j < glu.hidden) product else 0;
        }
        return h;
    }
};

/// The output has `code_width` columns, of which those past `hidden` hold zeros, and `block`
/// values of a row share a scale. The scale atoms cover `scale_width` columns and a multiple of
/// 128 rows. A block outside the product gets the scale of zero values, or 0 with `fill_padding`.
fn BlockGlu(
    comptime activation: Activation,
    comptime block: u32,
    comptime format: ScaleFormat,
    comptime fill_padding: bool,
) type {
    return struct {
        const lanes = block / chunk_values;

        fn kernel(
            z: [*]addrspace(.global) const Words,
            codes: [*]addrspace(.global) Words,
            scales: [*]addrspace(.global) fp8.Scale(format),
            rows: u32,
            padded: u32,
            hidden: u32,
            code_width: u32,
            scale_width: u32,
            gate_first: u32,
            limit: f32,
        ) callconv(kernel_convention) void {
            const scale_rows = std.mem.alignForward(u32, rows, atom_rows);
            const chunks_per_row = scale_width / chunk_values;
            const i = @workGroupId(0) * threads + @workItemId(0);
            const row = i / chunks_per_row;
            const column = i % chunks_per_row * chunk_values;
            const inside = row < rows and column < hidden;
            var values: Values = @splat(0);
            const glu: Glu = .{
                .z = z,
                .padded = padded,
                .hidden = hidden,
                .gate_first = gate_first != 0,
                .limit = limit,
            };
            if (inside) values = glu.products(activation, row, column);
            const amax = fp8.maxNanOverLanes(fp8.amaxOf(values), lanes);
            const scaling: fp8_pack.Scaling(format) = .init(amax);
            if (row < rows and column < code_width) {
                const index = (@as(u64, row) * code_width + column) / chunk_values;
                codes[index] = fp8_pack.scaledCodes(values, scaling.factor);
            }
            if (row >= scale_rows or i % lanes != 0) return;
            const blocks_per_row = scale_width / block;
            const block_index = column / block;
            const atoms_per_row = blocks_per_row / atom_blocks;
            const atom = (row / atom_rows) * atoms_per_row + block_index / atom_blocks;
            const offset = atom * atom_bytes + (row % 32) * 16 +
                (row % atom_rows) / 32 * atom_blocks + block_index % atom_blocks;
            const padding = !(row < rows and column < code_width);
            scales[offset] = if (fill_padding and padding) 0 else scaling.scale;
        }
    };
}

fn RowGlu(comptime activation: Activation) type {
    return struct {
        fn kernel(
            z: [*]addrspace(.global) const Words,
            h: [*]addrspace(.global) Words,
            rows: u32,
            padded: u32,
            hidden: u32,
            gate_first: u32,
            limit: f32,
        ) callconv(kernel_convention) void {
            const chunks_per_row = hidden / chunk_values;
            const i = @workGroupId(0) * threads + @workItemId(0);
            const row = i / chunks_per_row;
            if (row >= rows) return;
            const column = i % chunks_per_row * chunk_values;
            const glu: Glu = .{
                .z = z,
                .padded = padded,
                .hidden = hidden,
                .gate_first = gate_first != 0,
                .limit = limit,
            };
            const values = glu.products(activation, row, column);
            const first = (@as(u64, row) * hidden + column) / 8;
            h[first] = bf16.packChunk(values[0..8].*);
            h[first + 1] = bf16.packChunk(values[8..16].*);
        }
    };
}

/// Work group `r` keeps row `r` of `h` in registers and quantizes it by the rule of
/// `fp8_rowwise_quant_bf16`. Assumes `hidden <= row_chunks * threads * 16`.
fn RowGluQuant(comptime activation: Activation) type {
    return struct {
        var partials: [warps]f32 addrspace(.shared) = undefined;

        fn kernel(
            z: [*]addrspace(.global) const Words,
            codes: [*]addrspace(.global) Words,
            scales: [*]addrspace(.global) f32,
            padded: u32,
            hidden: u32,
            gate_first: u32,
            limit: f32,
        ) callconv(kernel_convention) void {
            @setEvalBranchQuota(10_000);
            const row = @workGroupId(0);
            const tid = @workItemId(0);
            const chunks = hidden / chunk_values;
            const glu: Glu = .{
                .z = z,
                .padded = padded,
                .hidden = hidden,
                .gate_first = gate_first != 0,
                .limit = limit,
            };
            var h: [row_chunks][2]Words = undefined;
            var amax: f32 = 0;
            inline for (&h, 0..) |*packed_h, j| {
                const chunk = tid + @as(u32, j * threads);
                if (chunk < chunks) {
                    const values = glu.products(activation, row, chunk * chunk_values);
                    amax = fp8.maxNan(amax, fp8.amaxOf(values));
                    packed_h[0] = bf16.packChunk(values[0..8].*);
                    packed_h[1] = bf16.packChunk(values[8..16].*);
                }
            }
            amax = fp8.maxNanOverLanes(amax, 32);
            if (tid % 32 == 0) partials[tid / 32] = amax;
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (0..warps) |w| amax = fp8.maxNan(amax, partials[w]);
            const scale = fp8_pack.rowScale(.bf16, amax);
            const divisor: fp8.Divisor = .init(scale);
            inline for (h, 0..) |packed_h, j| {
                const chunk = tid + @as(u32, j * threads);
                if (chunk < chunks) {
                    var values: Values = undefined;
                    inline for (0..2) |half| {
                        const eight = bf16.unpackChunk(packed_h[half]);
                        inline for (0..8) |i| values[half * 8 + i] = eight[i];
                    }
                    codes[@as(u64, row) * chunks + chunk] = fp8_pack.dividedCodes(divisor, values);
                }
            }
            if (tid == 0) scales[row] = scale;
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Activation)) |activation| {
            @export(&BlockGlu(activation, 32, .e8m0, false).kernel, .{
                .name = configurations.kernelName(.mxfp8, activation),
            });
            @export(&BlockGlu(activation, 128, .e8m0, true).kernel, .{
                .name = configurations.kernelName(.b128_e8m0, activation),
            });
            @export(&BlockGlu(activation, 128, .f32, true).kernel, .{
                .name = configurations.kernelName(.b128_f32, activation),
            });
            @export(&RowGlu(activation).kernel, .{
                .name = configurations.kernelName(.rowwise, activation),
            });
            if (configurations.hasKernel(.rowwise_one_pass, activation)) {
                @export(&RowGluQuant(activation).kernel, .{
                    .name = configurations.kernelName(.rowwise_one_pass, activation),
                });
            }
        }
    }
}
