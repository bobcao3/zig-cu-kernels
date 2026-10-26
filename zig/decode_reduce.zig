//! Work group `(w, g)` adds the FP32 partials of the weight tile `w` for the token rows `4·g` to
//! `4·g + 3` in the order of the CTAs that wrote them, and then runs the epilogue on the sums.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const bf16 = @import("bf16.zig");
const configurations = @import("decode_reduce_configurations.zig");

const Recipe = configurations.Recipe;
const Epilogue = configurations.Epilogue;
const threads = configurations.threads;
const tile_columns = configurations.tile_columns;
const rows_per_work_group = configurations.rows_per_work_group;
const columns_per_thread = configurations.columns_per_thread;
const threads_per_row = tile_columns / columns_per_thread;
const block_depth = 128;
const partials_per_batch = 8;

const Quad = @Vector(columns_per_thread, f32);

const Schedule = struct {
    units: u32,
    ctas: u32,

    fn start(s: Schedule, c: u32) u32 {
        return c * s.units / s.ctas;
    }

    fn owner(s: Schedule, unit: u32) u32 {
        return ((unit + 1) * s.ctas - 1) / s.units;
    }
};

fn Reduce(comptime recipe: Recipe, comptime epilogue: Epilogue) type {
    return struct {
        /// Assumes the partials layout `[ctas, 2, token_tile, 128]` and `n % 8 == 0`, and a grid
        /// of `(padded_n / 128, ceil(m / 4))` work groups.
        fn kernel(
            partials: [*]addrspace(.global) const Quad,
            y: [*]addrspace(.global) [2]u32,
            bias: [*]addrspace(.global) const [2]u32,
            residual: [*]addrspace(.global) const [2]u32,
            scale_a: [*]addrspace(.global) const f32,
            scale_w: [*]addrspace(.global) const f32,
            m: u32,
            n: u32,
            padded_n: u32,
            k: u32,
            ctas: u32,
            token_tile: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = threads },
        } }) void {
            const tile = @workGroupId(0);
            const blocks = k / block_depth;
            const schedule: Schedule = .{
                .units = padded_n / tile_columns * blocks,
                .ctas = ctas,
            };
            const first = schedule.owner(tile * blocks);
            const last = schedule.owner((tile + 1) * blocks - 1);
            if (first == last) return;
            if (comptime nvptx.gpu.sm >= 90) nvptx.griddepcontrol(.wait);

            const quad = @workItemId(0) % threads_per_row;
            const column: u64 = tile * tile_columns + quad * columns_per_thread;
            const row: u64 = @workGroupId(1) * rows_per_work_group + @workItemId(0) / threads_per_row;
            if (column >= n or row >= m) return;
            const first_slot = 2 * first + @intFromBool(schedule.start(first) < tile * blocks);
            var sum = partials[partialIndex(first_slot, token_tile, row, quad)];
            var c = first + 1;
            while (c <= last) : (c += partials_per_batch) {
                var loaded: [partials_per_batch]Quad = undefined;
                inline for (&loaded, 0..) |*partial, i| {
                    if (c + i <= last) partial.* = partials[partialIndex(2 * (c + i), token_tile, row, quad)];
                }
                inline for (loaded, 0..) |partial, i| {
                    if (c + i <= last) sum = addRounded(sum, partial);
                }
            }
            if (recipe == .fp8_rowwise) {
                var column_scales: Quad = undefined;
                inline for (0..columns_per_thread) |e| column_scales[e] = scale_w[column + e];
                sum = mulRounded(mulRounded(sum, @splat(scale_a[row])), column_scales);
            }
            if (epilogue == .residual) {
                sum = addRounded(sum, unpack(residual[(row * padded_n + column) / columns_per_thread]));
            }
            if (epilogue == .bias) sum = addRounded(sum, unpack(bias[column / columns_per_thread]));
            y[(row * n + column) / columns_per_thread] = .{
                bf16.pack(sum[0], sum[1]),
                bf16.pack(sum[2], sum[3]),
            };
        }

        fn partialIndex(slot: u64, token_tile: u64, row: u64, quad: u64) u64 {
            return (slot * token_tile + row) * threads_per_row + quad;
        }

        inline fn unpack(pairs: [2]u32) Quad {
            const low = bf16.unpack(pairs[0]);
            const high = bf16.unpack(pairs[1]);
            return .{ low[0], low[1], high[0], high[1] };
        }

        /// The epilogue rounds each operation on its own, so ptxas does not contract a product
        /// and a sum into one `fma` (PTX ISA 9.7.3.5).
        inline fn addRounded(a: Quad, b: Quad) Quad {
            var result: Quad = undefined;
            inline for (0..columns_per_thread) |e| {
                result[e] = nvptx.math.addF32(.nearest_even, .{}, a[e], b[e]);
            }
            return result;
        }

        inline fn mulRounded(a: Quad, b: Quad) Quad {
            var result: Quad = undefined;
            inline for (0..columns_per_thread) |e| {
                result[e] = nvptx.math.mulF32(.nearest_even, .{}, a[e], b[e]);
            }
            return result;
        }
    };
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (std.enums.values(Recipe)) |recipe| {
            for (std.enums.values(Epilogue)) |epilogue| {
                @export(&Reduce(recipe, epilogue).kernel, .{
                    .name = configurations.kernelName(recipe, epilogue),
                });
            }
        }
    }
}
