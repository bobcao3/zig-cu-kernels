//! The kernels write the FP8 packs of `zig_cu_kernels.packs` by the rules of the reference
//! quantizers in `tests/pack_references.py`. A work item loads 16 consecutive values of a row with
//! 16-byte loads and stores their 16 E4M3 codes with one 16-byte store.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const bf16 = @import("bf16.zig");
const fp8 = @import("fp8.zig");
const pair = @import("pair.zig");
const configurations = @import("fp8_pack_configurations.zig");

const Element = configurations.Element;
const ScaleFormat = configurations.ScaleFormat;
const Words = @Vector(4, u32);
const Values = [chunk_values]f32;
const chunk_values = configurations.chunk_values;
const threads = configurations.threads;
const warps = threads / 32;
const tile = configurations.tile;
const row_chunks = configurations.row_chunks;
const atom_rows = 128;
const mxfp8_block = 32;
const atom_blocks = 4;
const atom_bytes = 512;
const e8m0_one = 127;
const e8m0_max = 254;
const inv_448: f32 = @bitCast(@as(u32, 0x3b12_4925));
const min_group_scale: f32 = @bitCast(@as(u32, 0x0da2_4260));
const kernel_convention: std.builtin.CallingConvention = .{ .nvptx_kernel = .{
    .required_work_group_size = .{ .x = threads },
} };

inline fn load(
    comptime element: Element,
    input: [*]addrspace(.global) const Words,
    first: u64,
) Values {
    var values: Values = undefined;
    switch (element) {
        .bf16 => inline for (0..2) |half| {
            const eight = bf16.unpackChunk(input[first / 8 + half]);
            inline for (0..8) |i| values[half * 8 + i] = eight[i];
        },
        .f32 => inline for (0..4) |quarter| {
            const words = input[first / 4 + quarter];
            inline for (0..4) |i| values[quarter * 4 + i] = @bitCast(words[i]);
        },
    }
    return values;
}

/// `cvt.rn.satfinite.e4m3x2.f32` clamps magnitudes above 448 to 448 (PTX ISA 9.7.10.24).
pub inline fn codesOf(quotients: Values) Words {
    var codes: Words = undefined;
    inline for (0..4) |word| {
        const four = quotients[word * 4 ..][0..4];
        codes[word] = @backingInt(nvptx.E4m3x4.fromF32(four[0], four[1], four[2], four[3]));
    }
    return codes;
}

pub inline fn scaledCodes(values: Values, factor: f32) Words {
    var quotients: Values = undefined;
    inline for (&quotients, values) |*quotient, value| {
        quotient.* = nvptx.math.mulF32(.nearest_even, .{}, value, factor);
    }
    return codesOf(quotients);
}

/// Returns `packs.e8m0_ceil(amax * fp32(1 / 448))`, or 127 for a zero `amax`.
pub inline fn e8m0Of(amax: f32) u8 {
    const bits: u32 = @bitCast(nvptx.math.mulF32(.nearest_even, .{}, amax, inv_448));
    const exponent = (bits >> 23) & 0xff;
    const byte = @min(exponent + @intFromBool(bits & 0x7f_ffff != 0), e8m0_max);
    return if (amax == 0) e8m0_one else @intCast(byte);
}

pub fn Scaling(comptime format: ScaleFormat) type {
    return struct {
        scale: fp8.Scale(format),
        factor: f32,

        pub inline fn init(amax: f32) @This() {
            switch (format) {
                .e8m0 => {
                    const byte = e8m0Of(amax);
                    return .{ .scale = byte, .factor = fp8.e8m0ToF32(e8m0_max - byte) };
                },
                .f32 => {
                    const product = nvptx.math.mulF32(.nearest_even, .{}, amax, inv_448);
                    const scale = fp8.maxNan(product, min_group_scale);
                    return .{
                        .scale = scale,
                        .factor = nvptx.math.rcpF32(.nearest_even, false, scale),
                    };
                },
            }
        }
    };
}

/// Each matrix of the input holds `rows` rows of `width` values, and `block` values of a row
/// share a scale. The scale atoms cover `scale_width` columns, and a block outside the input gets
/// the scale of zero values, or 0 with `fill_padding`.
fn BlockQuantizer(
    comptime element: Element,
    comptime block: u32,
    comptime format: ScaleFormat,
    comptime fill_padding: bool,
) type {
    return struct {
        const lanes = block / chunk_values;

        fn kernel(
            input: [*]addrspace(.global) const Words,
            codes: [*]addrspace(.global) Words,
            scales: [*]addrspace(.global) fp8.Scale(format),
            rows: u32,
            width: u32,
            code_rows: u32,
            code_width: u32,
            scale_width: u32,
        ) callconv(kernel_convention) void {
            const scale_rows = std.mem.alignForward(u32, rows, atom_rows);
            const chunks_per_row = scale_width / chunk_values;
            const matrix: u64 = @workGroupId(1);
            const i = @workGroupId(0) * threads + @workItemId(0);
            const row = i / chunks_per_row;
            const column = i % chunks_per_row * chunk_values;
            const inside = row < rows and column < width;
            var values: Values = @splat(0);
            if (inside) {
                values = load(element, input, (matrix * rows + row) * width + column);
            }
            const amax = fp8.maxNanOverLanes(fp8.amaxOf(values), lanes);
            const scaling: Scaling(format) = .init(amax);
            if (row < code_rows and column < code_width) {
                codes[((matrix * code_rows + row) * code_width + column) / chunk_values] =
                    scaledCodes(values, scaling.factor);
            }
            if (row >= scale_rows or i % lanes != 0) return;
            const blocks_per_row = scale_width / block;
            const block_index = column / block;
            const atom = (row / atom_rows) * (blocks_per_row / atom_blocks) +
                block_index / atom_blocks;
            const offset = atom * atom_bytes + (row % 32) * 16 +
                (row % atom_rows) / 32 * atom_blocks + block_index % atom_blocks;
            scales[matrix * scale_rows * blocks_per_row + offset] =
                if (fill_padding and !inside) 0 else scaling.scale;
        }
    };
}

/// Work group `(j, i)` quantizes tile `(i, j)` and writes its scale to
/// `scales[i * scale_width + j]`, or writes 0 there when column `128 * j` lies outside the matrix.
fn TileQuantizer(comptime element: Element, comptime format: ScaleFormat) type {
    return struct {
        const chunks_per_row = tile / chunk_values;
        const rows_per_pass = threads / chunks_per_row;
        const passes = tile / rows_per_pass;
        var partials: [warps]f32 addrspace(.shared) = undefined;

        fn kernel(
            input: [*]addrspace(.global) const Words,
            codes: [*]addrspace(.global) Words,
            scales: [*]addrspace(.global) fp8.Scale(format),
            width: u32,
            scale_width: u32,
        ) callconv(kernel_convention) void {
            const tid = @workItemId(0);
            const tile_column = @workGroupId(0);
            const tile_row = @workGroupId(1);
            const scale_index = @as(u64, tile_row) * scale_width + tile_column;
            if (tile_column * tile >= width) {
                if (tid == 0) scales[scale_index] = 0;
                return;
            }
            const first_row = tile_row * tile + tid / chunks_per_row;
            const column = tile_column * tile + tid % chunks_per_row * chunk_values;
            var values: [passes]Values = undefined;
            var amax: f32 = 0;
            inline for (&values, 0..) |*pass_values, pass| {
                const row = first_row + pass * rows_per_pass;
                pass_values.* = load(element, input, @as(u64, row) * width + column);
                amax = fp8.maxNan(amax, fp8.amaxOf(pass_values.*));
            }
            amax = fp8.maxNanOverLanes(amax, 32);
            if (tid % 32 == 0) partials[tid / 32] = amax;
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (0..warps) |w| amax = fp8.maxNan(amax, partials[w]);
            const scaling: Scaling(format) = .init(amax);
            inline for (values, 0..) |pass_values, pass| {
                const row = first_row + pass * rows_per_pass;
                codes[(@as(u64, row) * width + column) / chunk_values] =
                    scaledCodes(pass_values, scaling.factor);
            }
            if (tid == 0) scales[scale_index] = scaling.scale;
        }
    };
}

/// Returns the scale `round(max(amax, round(1e-12)) * fp32(1 / 448))` of a row, where `round`
/// rounds to the element type.
pub inline fn rowScale(comptime element: Element, amax: f32) f32 {
    const amax_min: f32 = @bitCast(@as(u32, switch (element) {
        .bf16 => 0x2b8d_0000,
        .f32 => 0x2b8c_bccc,
    }));
    const scale = nvptx.math.mulF32(.nearest_even, .{}, fp8.maxNan(amax, amax_min), inv_448);
    return if (element == .bf16) bf16.toF32(@truncate(bf16.pack(scale, scale))) else scale;
}

/// Divides each value by the scale of `divisor` with correct rounding.
pub inline fn dividedCodes(divisor: fp8.Divisor, values: Values) Words {
    var pairs: [chunk_values / 2]pair.Pair = undefined;
    inline for (&pairs, 0..) |*p, index| p.* = .{ values[2 * index], values[2 * index + 1] };
    const quotients = divisor.divideAllUnbounded(chunk_values / 2, pairs);
    var flat: Values = undefined;
    inline for (quotients, 0..) |q, index| {
        flat[2 * index] = q[0];
        flat[2 * index + 1] = q[1];
    }
    return codesOf(flat);
}

/// A warp quantizes a row with the scale of `rowScale`.
fn RowQuantizer(comptime element: Element) type {
    return struct {
        fn kernel(
            input: [*]addrspace(.global) const Words,
            codes: [*]addrspace(.global) Words,
            scales: [*]addrspace(.global) f32,
            rows: u32,
            width: u32,
        ) callconv(kernel_convention) void {
            @setEvalBranchQuota(10_000);
            const row = @workGroupId(0) * warps + @workItemId(0) / 32;
            const lane = @workItemId(0) % 32;
            if (row >= rows) return;
            const chunks = width / chunk_values;
            const first = @as(u64, row) * width;
            var amax: f32 = 0;
            var chunk = lane;
            while (chunk < chunks) : (chunk += 32) {
                amax = fp8.maxNan(
                    amax,
                    fp8.amaxOf(load(element, input, first + chunk * chunk_values)),
                );
            }
            amax = fp8.maxNanOverLanes(amax, 32);
            const scale = rowScale(element, amax);
            const divisor: fp8.Divisor = .init(scale);
            chunk = lane;
            while (chunk < chunks) : (chunk += 32) {
                const values = load(element, input, first + chunk * chunk_values);
                codes[first / chunk_values + chunk] = dividedCodes(divisor, values);
            }
            if (lane == 0) scales[row] = scale;
        }
    };
}

/// A work group quantizes a row of at most `threads * row_chunks` chunks with the scale of
/// `rowScale`, and keeps the values of the row in registers between the amax and the division.
fn HeldRowQuantizer(comptime element: Element) type {
    return struct {
        var partials: [warps]f32 addrspace(.shared) = undefined;

        fn kernel(
            input: [*]addrspace(.global) const Words,
            codes: [*]addrspace(.global) Words,
            scales: [*]addrspace(.global) f32,
            width: u32,
        ) callconv(kernel_convention) void {
            @setEvalBranchQuota(10_000);
            const tid = @workItemId(0);
            const row = @workGroupId(0);
            const chunks = width / chunk_values;
            const first = @as(u64, row) * width;
            var values: [row_chunks]Values = @splat(@splat(0));
            var amax: f32 = 0;
            inline for (&values, 0..) |*chunk_values_held, k| {
                const chunk = tid + k * threads;
                if (chunk < chunks) {
                    chunk_values_held.* = load(element, input, first + chunk * chunk_values);
                    amax = fp8.maxNan(amax, fp8.amaxOf(chunk_values_held.*));
                }
            }
            amax = fp8.maxNanOverLanes(amax, 32);
            if (tid % 32 == 0) partials[tid / 32] = amax;
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (0..warps) |w| amax = fp8.maxNan(amax, partials[w]);
            const scale = rowScale(element, amax);
            const divisor: fp8.Divisor = .init(scale);
            inline for (values, 0..) |chunk_values_held, k| {
                const chunk = tid + k * threads;
                if (chunk < chunks) {
                    codes[first / chunk_values + chunk] = dividedCodes(divisor, chunk_values_held);
                }
            }
            if (tid == 0) scales[row] = scale;
        }
    };
}

fn PairWord(comptime element: Element) type {
    return switch (element) {
        .bf16 => u32,
        .f32 => @Vector(2, u32),
    };
}

inline fn unpackPair(comptime element: Element, word: PairWord(element)) [2]f32 {
    return switch (element) {
        .bf16 => .{ @bitCast(word << 16), @bitCast(word & 0xffff_0000) },
        .f32 => .{ @bitCast(word[0]), @bitCast(word[1]) },
    };
}

inline fn pairCodes(first: f32, second: f32, first_factor: f32, second_factor: f32) u16 {
    const mul = nvptx.math.mulF32;
    return @backingInt(nvptx.E4m3x2.fromF32(
        mul(.nearest_even, .{}, first, first_factor),
        mul(.nearest_even, .{}, second, second_factor),
    ));
}

inline fn atomOffset(atom: u64, row: u32, block_index: u32) u64 {
    return atom * atom_bytes + (row % 32) * 16 + (row % atom_rows) / 32 * atom_blocks +
        block_index % atom_blocks;
}

/// A work item quantizes two adjacent columns of 32 rows into the columnwise MXFP8 pack, where 32
/// rows of a column share a scale, and with `rowwise` also into the rowwise MXFP8 pack. Both packs
/// keep the row-major layout of the input with rows padded to `code_width` values.
fn ColumnQuantizer(comptime element: Element, comptime rowwise: bool) type {
    return struct {
        const Word = PairWord(element);
        const lanes = mxfp8_block / 2;

        fn kernel(
            input: [*]addrspace(.global) const Word,
            row_codes: [*]addrspace(.global) u16,
            row_scales: [*]addrspace(.global) u8,
            column_codes: [*]addrspace(.global) u16,
            column_scales: [*]addrspace(.global) u8,
            rows: u32,
            width: u32,
            code_width: u32,
        ) callconv(kernel_convention) void {
            @setEvalBranchQuota(10_000);
            const pairs_per_row = code_width / 2;
            const i = @workGroupId(0) * threads + @workItemId(0);
            const row_block = i / pairs_per_row;
            const column = i % pairs_per_row * 2;
            const first_row = row_block * mxfp8_block;
            if (first_row >= std.mem.alignForward(u32, rows, atom_rows)) return;
            var values: [mxfp8_block][2]f32 = @splat(.{ 0, 0 });
            var column_amax: [2]f32 = .{ 0, 0 };
            inline for (&values, 0..) |*pair_values, r| {
                const row = first_row + @as(u32, r);
                if (row < rows and column < width) {
                    pair_values.* =
                        unpackPair(element, input[(@as(u64, row) * width + column) / 2]);
                }
                inline for (&column_amax, pair_values.*) |*amax, value| {
                    amax.* = fp8.maxNan(amax.*, @abs(value));
                }
            }
            const first: Scaling(.e8m0) = .init(column_amax[0]);
            const second: Scaling(.e8m0) = .init(column_amax[1]);
            const atoms_per_row = code_width / atom_rows;
            inline for (values, 0..) |pair_values, r| {
                const row = first_row + @as(u32, r);
                const index = (@as(u64, row) * code_width + column) / 2;
                if (row < rows) {
                    column_codes[index] =
                        pairCodes(pair_values[0], pair_values[1], first.factor, second.factor);
                }
                if (rowwise) {
                    const amax = fp8.maxNanOverLanes(fp8.amaxOf(pair_values), lanes);
                    const scaling: Scaling(.e8m0) = .init(amax);
                    if (row < rows) {
                        row_codes[index] = pairCodes(
                            pair_values[0],
                            pair_values[1],
                            scaling.factor,
                            scaling.factor,
                        );
                    }
                    if (column % mxfp8_block == 0) {
                        const block_index = column / mxfp8_block;
                        const atom = @as(u64, row / atom_rows) * atoms_per_row +
                            block_index / atom_blocks;
                        row_scales[atomOffset(atom, row, block_index)] = scaling.scale;
                    }
                }
            }
            const atom = @as(u64, row_block / atom_blocks) * atoms_per_row + column / atom_rows;
            column_scales[atomOffset(atom, column, row_block)] = first.scale;
            column_scales[atomOffset(atom, column + 1, row_block)] = second.scale;
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        for (std.enums.values(Element)) |element| {
            @export(&BlockQuantizer(element, 32, .e8m0, false).kernel, .{
                .name = configurations.kernelName(.mxfp8, element, .e8m0),
            });
            for (std.enums.values(ScaleFormat)) |format| {
                @export(&BlockQuantizer(element, tile, format, true).kernel, .{
                    .name = configurations.kernelName(.groups, element, format),
                });
                @export(&TileQuantizer(element, format).kernel, .{
                    .name = configurations.kernelName(.weight, element, format),
                });
            }
            @export(&RowQuantizer(element).kernel, .{
                .name = configurations.kernelName(.rowwise, element, .f32),
            });
            @export(&HeldRowQuantizer(element).kernel, .{
                .name = configurations.kernelName(.rowwise_held, element, .f32),
            });
            @export(&ColumnQuantizer(element, false).kernel, .{
                .name = configurations.kernelName(.columnwise, element, .e8m0),
            });
            @export(&ColumnQuantizer(element, true).kernel, .{
                .name = configurations.kernelName(.dual, element, .e8m0),
            });
        }
    }
}
