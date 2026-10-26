//! The kernels compute the backward of the GLU product `h` of a BF16 matrix `Z = [first | second]`
//! from the BF16 gradient `dh`, by the rules of `tests/fp8_mm_references.py::glu_backward`. They
//! write the rowwise and the columnwise MXFP8 packs of `dZ` and the columnwise MXFP8 pack of the
//! recomputed `h`, the operands of the data and weight gradient products. A work item handles two
//! adjacent columns of one half over 32 rows.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const math = nvptx.math;
const bf16 = @import("bf16.zig");
const fp8 = @import("fp8.zig");
const fp8_pack = @import("fp8_pack.zig");
const configurations = @import("glu_bwd_configurations.zig");

const Activation = configurations.Activation;
const Scaling = fp8_pack.Scaling(.e8m0);
const threads = configurations.threads;
const block_rows = 32;
const block_lanes = 16;
const atom_rows = 128;
const atom_blocks = 4;
const atom_bytes = 512;
const flush: math.Modifiers = .{ .ftz = true };
const negative_log2e: f32 = @bitCast(@as(u32, 0xbfb8_aa3b));
const gelu_linear: f32 = @bitCast(@as(u32, 0x4013_5761));
const gelu_cubic: f32 = @bitCast(@as(u32, 0x3dd2_d3e8));
const gelu_slope_constant: f32 = @bitCast(@as(u32, 0x3ecc_422a));
const gelu_slope_quadratic: f32 = @bitCast(@as(u32, 0x3d5b_33b6));
const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = threads },
} };

const Gradients = struct {
    up: f32,
    gate: f32,
    h: f32,
};

inline fn mul(a: f32, b: f32) f32 {
    return math.mulF32(.nearest_even, .{}, a, b);
}

inline fn fma(a: f32, b: f32, c: f32) f32 {
    return math.fmaF32(.nearest_even, .{}, a, b, c);
}

/// The sigmoid is `rcp.approx.ftz(1 + ex2.approx.ftz(-log2(e)·g))` (PTX ISA 9.7.3.21,
/// 9.7.3.13), and the FP32 steps of the derivative flush subnormal results to zero. The gradient of
/// a clamped operand is its BF16 gradient times zero.
inline fn swigluBackward(dh: f32, up_raw: f32, gate_raw: f32, limit: f32) Gradients {
    const gate = @min(gate_raw, limit);
    const up = @max(@min(up_raw, limit), -limit);
    const exponent = math.mulF32(.nearest_even, flush, gate, negative_log2e);
    const power = math.ex2ApproxF32(true, exponent);
    const sigmoid = math.rcpApproxF32(true, math.addF32(.nearest_even, flush, power, 1));
    const silu = math.mulF32(.nearest_even, flush, gate, sigmoid);
    const complement = math.subF32(.nearest_even, flush, 1, sigmoid);
    const slope = math.fmaF32(.nearest_even, flush, silu, complement, sigmoid);
    const up_gradient = math.mulF32(.nearest_even, flush, dh, silu);
    const scaled = math.mulF32(.nearest_even, flush, dh, up);
    const gate_gradient = math.mulF32(.nearest_even, flush, scaled, slope);
    const up_mask: f32 = if (up == up_raw) 1 else 0;
    const gate_mask: f32 = if (gate == gate_raw) 1 else 0;
    return .{
        .up = mul(bf16.round(up_gradient), up_mask),
        .gate = mul(bf16.round(gate_gradient), gate_mask),
        .h = bf16.round(math.mulF32(.nearest_even, flush, up, silu)),
    };
}

/// The tanh GELU follows `triton_ops/geglu_tanh.py`: `tanh(|t|)` is
/// `1 - 2 / (2^(2·log2(e)·|t|) + 1)` with `t = sqrt(2/π)·(gate + 0.044715·gate³)`, and
/// `d gelu / d gate = 0.5·(1 + tanh) + gate·(1 - tanh²)·0.5·dt/dgate`.
inline fn gegluBackward(dh: f32, up: f32, gate: f32) Gradients {
    const square = mul(gate, gate);
    const power = math.ex2ApproxF32(true, mul(@abs(gate), fma(gelu_cubic, square, gelu_linear)));
    const reciprocal = math.rcpApproxF32(true, math.addF32(.nearest_even, .{}, power, 1));
    const magnitude = fma(reciprocal, -2, 1);
    const sign: f32 = if (gate < 0) -0.5 else 0.5;
    const half = fma(sign, magnitude, 0.5);
    const half_slope = fma(gelu_slope_quadratic, square, gelu_slope_constant);
    const complement = fma(-magnitude, magnitude, 1);
    const derivative = fma(mul(gate, complement), half_slope, half);
    const gelu = bf16.round(mul(gate, half));
    return .{
        .up = bf16.round(mul(dh, gelu)),
        .gate = bf16.round(mul(derivative, bf16.round(mul(dh, up)))),
        .h = bf16.round(mul(up, gelu)),
    };
}

inline fn atomOffset(atom: u64, row: u32, block_index: u32) u64 {
    return atom * atom_bytes + (row % 32) * 16 + (row % atom_rows) / 32 * atom_blocks +
        block_index % atom_blocks;
}

inline fn pairCodes(values: [2]f32, factors: [2]f32) u16 {
    const first = mul(values[0], factors[0]);
    return @backingInt(nvptx.E4m3x2.fromF32(first, mul(values[1], factors[1])));
}

/// A columnwise pack of `width` columns gets the codes and scales of one column pair over 32 rows.
const ColumnPack = struct {
    codes: [*]addrspace(.global) u16,
    scales: [*]addrspace(.global) u8,
    width: u32,

    inline fn write(
        pack: ColumnPack,
        words: *const [block_rows]u32,
        rows: u32,
        row_block: u32,
        column: u32,
    ) void {
        var amax: [2]f32 = .{ 0, 0 };
        inline for (words) |word| {
            const pair_values = bf16.unpack(word);
            inline for (&amax, 0..) |*a, j| a.* = fp8.maxNan(a.*, @abs(pair_values[j]));
        }
        const first: Scaling = .init(amax[0]);
        const second: Scaling = .init(amax[1]);
        const first_row = row_block * block_rows;
        inline for (words, 0..) |word, r| {
            const row = first_row + @as(u32, r);
            if (row < rows) {
                const pair_values = bf16.unpack(word);
                pack.codes[(@as(u64, row) * pack.width + column) / 2] = pairCodes(
                    .{ pair_values[0], pair_values[1] },
                    .{ first.factor, second.factor },
                );
            }
        }
        const atoms_per_row = pack.width / atom_rows;
        const atom = @as(u64, row_block / atom_blocks) * atoms_per_row + column / atom_rows;
        pack.scales[atomOffset(atom, column, row_block)] = first.scale;
        pack.scales[atomOffset(atom, column + 1, row_block)] = second.scale;
    }
};

inline fn writeRow(
    codes: [*]addrspace(.global) u16,
    scales: [*]addrspace(.global) u8,
    values: [2]f32,
    rows: u32,
    row: u32,
    width: u32,
    column: u32,
) void {
    const amax = fp8.maxNanOverLanes(fp8.amaxOf(values), block_lanes);
    const scaling: Scaling = .init(amax);
    if (row < rows) {
        codes[(@as(u64, row) * width + column) / 2] =
            pairCodes(values, .{ scaling.factor, scaling.factor });
    }
    if (column % (2 * block_lanes) == 0) {
        const block_index = column / (2 * block_lanes);
        const atom = @as(u64, row / atom_rows) * (width / atom_rows) + block_index / atom_blocks;
        scales[atomOffset(atom, row, block_index)] = scaling.scale;
    }
}

const RowPair = struct {
    first: [2]f32 = .{ 0, 0 },
    second: [2]f32 = .{ 0, 0 },
    h: [2]f32 = .{ 0, 0 },
};

/// Row `r` of `z` holds the `hidden` columns of `first` and then the `hidden` columns of `second`.
const Glu = struct {
    dh: [*]addrspace(.global) const u32,
    z: [*]addrspace(.global) const u32,
    hidden: u32,
    gate_first: bool,
    limit: f32,

    inline fn pair(glu: Glu, comptime activation: Activation, row: u32, column: u32) RowPair {
        const z_row = @as(u64, row) * 2 * glu.hidden + column;
        const gradient = bf16.unpack(glu.dh[(@as(u64, row) * glu.hidden + column) / 2]);
        const lo = bf16.unpack(glu.z[z_row / 2]);
        const hi = bf16.unpack(glu.z[(z_row + glu.hidden) / 2]);
        var result: RowPair = .{};
        inline for (0..2) |j| {
            const up = if (glu.gate_first) hi[j] else lo[j];
            const gate = if (glu.gate_first) lo[j] else hi[j];
            const g = switch (activation) {
                .swiglu => swigluBackward(gradient[j], up, gate, glu.limit),
                .geglu_tanh => gegluBackward(gradient[j], up, gate),
            };
            result.first[j] = if (glu.gate_first) g.gate else g.up;
            result.second[j] = if (glu.gate_first) g.up else g.gate;
            result.h[j] = g.h;
        }
        return result;
    }
};

fn GluBackward(comptime activation: Activation) type {
    return struct {
        fn operands(
            dh: [*]addrspace(.global) const u32,
            z: [*]addrspace(.global) const u32,
            dz_row_codes: [*]addrspace(.global) u16,
            dz_row_scales: [*]addrspace(.global) u8,
            dz_column_codes: [*]addrspace(.global) u16,
            dz_column_scales: [*]addrspace(.global) u8,
            h_codes: [*]addrspace(.global) u16,
            h_scales: [*]addrspace(.global) u8,
            rows: u32,
            hidden: u32,
            padded: u32,
            gate_first: u32,
            limit: f32,
        ) callconv(kernel_convention) void {
            @setEvalBranchQuota(20_000);
            const pairs_per_row = padded / 2;
            const i = @workGroupId(0) * threads + @workItemId(0);
            const row_block = i / pairs_per_row;
            const column = i % pairs_per_row * 2;
            const first_row = row_block * block_rows;
            if (first_row >= std.mem.alignForward(u32, rows, atom_rows)) return;
            const width = 2 * hidden;
            const inside_columns = column < hidden;
            const glu: Glu = .{
                .dh = dh,
                .z = z,
                .hidden = hidden,
                .gate_first = gate_first != 0,
                .limit = limit,
            };
            var first: [block_rows]u32 = @splat(0);
            var second: [block_rows]u32 = @splat(0);
            var h: [block_rows]u32 = @splat(0);
            inline for (0..block_rows) |r| {
                const row = first_row + @as(u32, r);
                const p: RowPair = if (row < rows and inside_columns)
                    glu.pair(activation, row, column)
                else
                    .{};
                first[r] = bf16.pack(p.first[0], p.first[1]);
                second[r] = bf16.pack(p.second[0], p.second[1]);
                h[r] = bf16.pack(p.h[0], p.h[1]);
                if (inside_columns) {
                    const codes = dz_row_codes;
                    const scales = dz_row_scales;
                    writeRow(codes, scales, p.first, rows, row, width, column);
                    writeRow(codes, scales, p.second, rows, row, width, hidden + column);
                }
            }
            const h_pack: ColumnPack = .{ .codes = h_codes, .scales = h_scales, .width = padded };
            h_pack.write(&h, rows, row_block, column);
            if (!inside_columns) return;
            const dz_pack: ColumnPack = .{
                .codes = dz_column_codes,
                .scales = dz_column_scales,
                .width = width,
            };
            dz_pack.write(&first, rows, row_block, column);
            dz_pack.write(&second, rows, row_block, hidden + column);
        }

        /// Writes the BF16 `dZ`, and the BF16 `h` and the rowwise MXFP8 pack of `dZ` where their
        /// addresses are not zero. A work item handles one column pair of 32 rows.
        fn bf16Outputs(
            dh: [*]addrspace(.global) const u32,
            z: [*]addrspace(.global) const u32,
            dz: [*]addrspace(.global) u32,
            dz_row_codes: ?[*]addrspace(.global) u16,
            dz_row_scales: ?[*]addrspace(.global) u8,
            h: ?[*]addrspace(.global) u32,
            rows: u32,
            hidden: u32,
            gate_first: u32,
            limit: f32,
        ) callconv(kernel_convention) void {
            @setEvalBranchQuota(20_000);
            const pairs_per_row = hidden / 2;
            const i = @workGroupId(0) * threads + @workItemId(0);
            const row_block = i / pairs_per_row;
            const column = i % pairs_per_row * 2;
            const first_row = row_block * block_rows;
            if (first_row >= std.mem.alignForward(u32, rows, atom_rows)) return;
            const width = 2 * hidden;
            const glu: Glu = .{
                .dh = dh,
                .z = z,
                .hidden = hidden,
                .gate_first = gate_first != 0,
                .limit = limit,
            };
            inline for (0..block_rows) |r| {
                const row = first_row + @as(u32, r);
                const p: RowPair = if (row < rows) glu.pair(activation, row, column) else .{};
                if (row < rows) {
                    const index = (@as(u64, row) * width + column) / 2;
                    dz[index] = bf16.pack(p.first[0], p.first[1]);
                    dz[index + hidden / 2] = bf16.pack(p.second[0], p.second[1]);
                    if (h) |h_words| {
                        h_words[(@as(u64, row) * hidden + column) / 2] = bf16.pack(p.h[0], p.h[1]);
                    }
                }
                if (dz_row_codes) |codes| {
                    writeRow(codes, dz_row_scales.?, p.first, rows, row, width, column);
                    writeRow(codes, dz_row_scales.?, p.second, rows, row, width, hidden + column);
                }
            }
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Activation)) |activation| {
            @export(&GluBackward(activation).operands, .{
                .name = configurations.kernelName(.operands, activation),
            });
            @export(&GluBackward(activation).bf16Outputs, .{
                .name = configurations.kernelName(.bf16, activation),
            });
        }
    }
}
