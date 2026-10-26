const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const bf16 = @import("bf16.zig");
const fp8 = @import("fp8.zig");
const warp = @import("warp.zig");
const configurations = @import("rms_norm_configurations.zig");
const bulk = @import("bulk.zig");
const pair = @import("pair.zig");

const Pair = pair.Pair;
const Bf16x2 = nvptx.Bf16x2;
const bf16_ones: Bf16x2 = @fromBackingInt(0x3f80_3f80);

const Chunk = bf16.Chunk;
const Options = configurations.Options;
const Size = configurations.Size;
const Element = configurations.Element;
const values_per_chunk = configurations.values_per_chunk;
const HalfChunk = @Vector(2, u32);

const Input = [*]addrspace(.global) const Chunk;
const Output = [*]addrspace(.global) Chunk;
const OptionalInput = [*]allowzero addrspace(.global) const Chunk;
const OptionalOutput = [*]allowzero addrspace(.global) Chunk;

fn bytesOf(comptime element: Element) u32 {
    return switch (element) {
        .bf16 => 2,
        .f32 => 4,
    };
}

fn RmsNorm(comptime options: Options, comptime size: Size) type {
    return struct {
        /// Each slot of a work item holds 16 bytes of `x`, so that the work items of a row load and
        /// store consecutive chunks of `x`.
        const values_per_slot = @sizeOf(Chunk) / bytesOf(options.x);
        const slots = size.chunks_per_thread * values_per_chunk / values_per_slot;
        const chunks_per_scale_group = configurations.scale_group / values_per_slot;
        const pairs_per_slot = values_per_slot / 2;
        /// The values of a slot as FP32 pairs.
        const SlotPairs = [pairs_per_slot]Pair;
        const gated = options.add == .gated or options.add == .dual_gated;
        const normalized_branch = options.add == .dual or options.add == .dual_gated;
        const modulated = gated or options.modulation != .none;
        const quantized = configurations.isFp8(options.output);
        const scale_format: fp8.ScaleFormat =
            if (options.output == .fp8_blockwise_e8m0) .e8m0 else .f32;
        const ScaleOutput = [*]allowzero addrspace(.global) fp8.Scale(scale_format);
        const addend_element: Element = if (options.add == .residual) options.x else .bf16;
        const weight_element: Element = switch (options.weight) {
            .f32, .zero_centered_f32 => .f32,
            .none, .bf16, .zero_centered_bf16 => .bf16,
        };
        /// A work item with at most two chunks loads its values of the weights and modulation rows
        /// together with its values of the row, so that the loads overlap. With four chunks, only
        /// the FP8 outputs without an addend have the registers for the early loads; without them,
        /// each chunk waits for its weights after the work of the chunk before.
        const preloaded = size.chunks_per_thread <= 2 or (quantized and options.add == .none);
        /// A work item of four chunks spends enough instructions per row that removing the bounds
        /// checks of whole rows pays. With fewer chunks, the copy of the body without the checks
        /// takes more registers, so fewer work items fit on a streaming multiprocessor. With an
        /// addend, ptxas spills that copy under the 64 registers that 1024 work items leave.
        const whole_path = size.chunks_per_thread >= 4 and
            (options.add == .none or size.threads <= 512);

        fn Packed(comptime element: Element) type {
            return switch (values_per_slot * bytesOf(element)) {
                8 => HalfChunk,
                16 => Chunk,
                32 => [2]Chunk,
                else => unreachable,
            };
        }

        fn Row(comptime element: Element) type {
            return [slots]Packed(element);
        }

        comptime {
            if (options.output == .f32 and options.modulation != .none) {
                @compileError("the modulation rounds to BF16");
            }
            if (quantized and options.x != .bf16) {
                @compileError("the FP8 outputs take a BF16 x");
            }
        }

        var partials: [2][configurations.max_threads / 32]f32 addrspace(.shared) = undefined;

        /// The weights and modulation rows that multiply a row after its reduction.
        const Factors = struct {
            weight: OptionalInput,
            branch_weight: OptionalInput,
            scale: OptionalInput,
            shift: OptionalInput,
            gains: Row(weight_element) = undefined,
            branch_gains: Row(weight_element) = undefined,
            scales: Row(.bf16) = undefined,
            shifts: Row(.bf16) = undefined,

            inline fn preload(factors: *Factors, position: anytype) void {
                if (!preloaded) return;
                if (options.weight != .none) {
                    factors.gains = loadRow(weight_element, position, factors.weight);
                    if (normalized_branch) {
                        const branch_weight = factors.branch_weight;
                        factors.branch_gains = loadRow(weight_element, position, branch_weight);
                    }
                }
                if (options.modulation != .none) {
                    factors.scales = loadRow(.bf16, position, factors.scale);
                }
                if (options.modulation == .scale_shift) {
                    factors.shifts = loadRow(.bf16, position, factors.shift);
                }
            }

            inline fn gainBits(
                factors: Factors,
                position: anytype,
                comptime slot: u32,
            ) Packed(weight_element) {
                if (options.weight == .none) return std.mem.zeroes(Packed(weight_element));
                return bitsOf(weight_element, factors.gains, factors.weight, position, slot);
            }

            inline fn branchGainBits(
                factors: Factors,
                position: anytype,
                comptime slot: u32,
            ) Packed(weight_element) {
                if (options.weight == .none) return std.mem.zeroes(Packed(weight_element));
                const gains = factors.branch_gains;
                return bitsOf(weight_element, gains, factors.branch_weight, position, slot);
            }

            inline fn scaleBits(factors: Factors, position: anytype, comptime slot: u32) Packed(.bf16) {
                return bitsOf(.bf16, factors.scales, factors.scale, position, slot);
            }

            inline fn shiftBits(factors: Factors, position: anytype, comptime slot: u32) Packed(.bf16) {
                return bitsOf(.bf16, factors.shifts, factors.shift, position, slot);
            }

            inline fn bitsOf(
                comptime element: Element,
                row: Row(element),
                base: OptionalInput,
                position: anytype,
                comptime slot: u32,
            ) Packed(element) {
                return if (preloaded) row[slot] else load(element, base, position.chunk(slot));
            }
        };

        /// The place of a work item in its row. In a whole position every slot of the work item
        /// holds a chunk of the row, so `holds` needs no comparison.
        fn PositionOf(comptime whole: bool) type {
            return struct {
                row: u32,
                lane: u32,
                threads_per_row: u32,
                /// The number of chunks of the row of `x`, or 0 past the last row.
                chunks: u32,
                sample: u32,

                /// Returns the index of the chunk of the row of `x` that `slot` holds.
                inline fn chunk(position: @This(), comptime slot: u32) u32 {
                    return position.lane + slot * position.threads_per_row;
                }

                inline fn holds(position: @This(), comptime slot: u32) bool {
                    return whole or position.chunk(slot) < position.chunks;
                }

                inline fn asWhole(position: @This()) PositionOf(true) {
                    return .{
                        .row = position.row,
                        .lane = position.lane,
                        .threads_per_row = position.threads_per_row,
                        .chunks = position.chunks,
                        .sample = position.sample,
                    };
                }
            };
        }

        const Position = PositionOf(false);

        /// The pointers of the inputs that `processRow` loads, and their strides.
        const Inputs = struct {
            x: Input,
            x_stride: u32,
            other: OptionalInput,
            other_stride: u32,
            weight: OptionalInput,
            branch_weight: OptionalInput,
            scale: OptionalInput,
            scale_stride: u32,
            shift: OptionalInput,
            shift_stride: u32,
        };

        /// The pointers and strides of a launch. A pointer that the options do not use may be 0.
        const Operands = struct {
            other: OptionalInput,
            other_stride: u32,
            residual_out: OptionalOutput,
            residual_out_stride: u32,
            gate: OptionalInput,
            gate_stride: u32,
            y: Output,
            y_stride: u32,
            y_scales: ScaleOutput,
            y_scale_stride: u32,
            rows: u32,
            width: u32,
            epsilon: f32,
        };

        fn positionOf(row: u32, rows: u32, width: u32, rows_per_sample: u32) Position {
            return .{
                .row = row,
                .lane = @workItemId(0),
                .threads_per_row = @workGroupSize(0),
                .chunks = if (row < rows) width / values_per_slot else 0,
                .sample = if (modulated) row / rows_per_sample else 0,
            };
        }

        /// Assumes that `width` is a multiple of 8 and at most `size.maxWidth()`. Assumes that each
        /// row starts at a multiple of 16 bytes, or of 8 bytes for an FP8 `y` (PTX ISA 6.4.1).
        /// Assumes that `@workGroupSize(0)`, the number of work items of a row, is a power of two
        /// up to 32 or a multiple of 32, as `row_layout` in `rms_norm.py` chooses. Assumes that
        /// `@workGroupSize(0)` is a multiple of 16 for a blockwise FP8 output. The strides count
        /// elements. A pointer that the options do not use may be 0.
        fn kernel(
            x: Input,
            x_stride: u32,
            other: OptionalInput,
            other_stride: u32,
            residual_out: OptionalOutput,
            residual_out_stride: u32,
            weight: OptionalInput,
            branch_weight: OptionalInput,
            gate: OptionalInput,
            gate_stride: u32,
            scale: OptionalInput,
            scale_stride: u32,
            shift: OptionalInput,
            shift_stride: u32,
            y: Output,
            y_stride: u32,
            y_scales: ScaleOutput,
            y_scale_stride: u32,
            rows: u32,
            width: u32,
            rows_per_sample: u32,
            epsilon: f32,
        ) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = size.threads },
            .max_registers_per_work_item = if (nvptx.gpu.sm >= 100) size.max_registers else null,
        } }) void {
            @setEvalBranchQuota(20_000);
            const row = @workGroupId(0) * @workGroupSize(1) + @workItemId(1);
            const position = positionOf(row, rows, width, rows_per_sample);
            const inputs: Inputs = .{
                .x = x,
                .x_stride = x_stride,
                .other = other,
                .other_stride = other_stride,
                .weight = weight,
                .branch_weight = branch_weight,
                .scale = scale,
                .scale_stride = scale_stride,
                .shift = shift,
                .shift_stride = shift_stride,
            };
            const operands: Operands = .{
                .other = other,
                .other_stride = other_stride,
                .residual_out = residual_out,
                .residual_out_stride = residual_out_stride,
                .gate = gate,
                .gate_stride = gate_stride,
                .y = y,
                .y_stride = y_stride,
                .y_scales = y_scales,
                .y_scale_stride = y_scale_stride,
                .rows = rows,
                .width = width,
                .epsilon = epsilon,
            };
            // Every work item of a work group takes the same branch, so that the barriers of the
            // row reduction stay aligned (PTX ISA 9.7.13.1).
            const whole_rows = (@workGroupId(0) + 1) * @workGroupSize(1) <= rows;
            if (whole_path and whole_rows and position.threads_per_row * slots * values_per_slot == width) {
                processRow(position.asWhole(), inputs, operands);
            } else {
                processRow(position, inputs, operands);
            }
        }

        inline fn processRow(position: anytype, inputs: Inputs, operands: Operands) void {
            var factors: Factors = .{
                .weight = inputs.weight,
                .branch_weight = inputs.branch_weight,
                .scale = rowOf(.bf16, inputs.scale, position.sample, inputs.scale_stride),
                .shift = rowOf(.bf16, inputs.shift, position.sample, inputs.shift_stride),
            };
            factors.preload(position);
            const row = position.row;
            const h = loadRow(options.x, position, rowOf(options.x, inputs.x, row, inputs.x_stride));
            var addend: Row(addend_element) = undefined;
            if (options.add != .none) {
                const other_row = rowOf(addend_element, inputs.other, row, inputs.other_stride);
                addend = loadRow(addend_element, position, other_row);
            }
            finish(operands, position, factors, h, addend);
        }

        extern var shared_storage: [0]u8 align(128) addrspace(.shared);
        var full: [configurations.max_stages]bulk.Mbarrier addrspace(.shared) = undefined;

        /// The kernel of `kernel` for `sm_90` and up, persistent and with the rows of `x` and of
        /// the addend staged in shared memory by bulk copies. Work group `g` takes the tiles `g`,
        /// `g + nctaid` and so on, a tile being `@workGroupSize(1)` consecutive rows. Each of the
        /// `stages` buffers of the dynamic shared memory holds the `x` rows of a tile followed by
        /// its addend rows. Assumes `1 <= stages <= max_stages` besides the assumptions of
        /// `kernel`, and assumes row strides that are multiples of 16 bytes.
        fn bulkKernel(
            x: Input,
            x_stride: u32,
            other: OptionalInput,
            other_stride: u32,
            residual_out: OptionalOutput,
            residual_out_stride: u32,
            weight: OptionalInput,
            branch_weight: OptionalInput,
            gate: OptionalInput,
            gate_stride: u32,
            scale: OptionalInput,
            scale_stride: u32,
            shift: OptionalInput,
            shift_stride: u32,
            y: Output,
            y_stride: u32,
            y_scales: ScaleOutput,
            y_scale_stride: u32,
            rows: u32,
            width: u32,
            rows_per_sample: u32,
            epsilon: f32,
            stages: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = size.threads },
        } }) void {
            @setEvalBranchQuota(20_000);
            const tile_rows = @workGroupSize(1);
            const tiles = (rows + tile_rows - 1) / tile_rows;
            const first = @workGroupId(0);
            if (first >= tiles) return;
            const step = nvptx.nctaid(0);
            const count = (tiles - first + step - 1) / step;
            const x_row_bytes = width * bytesOf(options.x);
            const addend_row_bytes = if (options.add != .none) width * bytesOf(addend_element) else 0;
            const stage_bytes = tile_rows * (x_row_bytes + addend_row_bytes);
            const buffers: [*]addrspace(.shared) u8 = &shared_storage;
            const issuer = @workItemId(0) == 0 and @workItemId(1) == 0;
            const Loader = struct {
                x: Input,
                x_stride: u32,
                other: OptionalInput,
                other_stride: u32,
                rows: u32,
                x_row_bytes: u32,
                addend_row_bytes: u32,

                fn issue(loader: @This(), tile: u32, buffer: [*]addrspace(.shared) u8, barrier: *addrspace(.shared) bulk.Mbarrier) void {
                    const first_row = tile * @workGroupSize(1);
                    const valid = @min(@workGroupSize(1), loader.rows - first_row);
                    _ = barrier.arriveExpectTx(valid * (loader.x_row_bytes + loader.addend_row_bytes));
                    const x_bytes: [*]addrspace(.global) const u8 = @ptrCast(loader.x);
                    const other_bytes: [*]addrspace(.global) const u8 = @ptrCast(loader.other);
                    const x_pitch: u64 = @as(u64, loader.x_stride) * bytesOf(options.x);
                    const other_pitch: u64 = @as(u64, loader.other_stride) * bytesOf(addend_element);
                    const addends = buffer + @workGroupSize(1) * loader.x_row_bytes;
                    for (0..valid) |r| {
                        const row: u64 = first_row + r;
                        const offset: u32 = @intCast(r);
                        bulk.load(buffer + offset * loader.x_row_bytes, x_bytes + row * x_pitch, loader.x_row_bytes, barrier);
                        if (options.add != .none) {
                            bulk.load(addends + offset * loader.addend_row_bytes, other_bytes + row * other_pitch, loader.addend_row_bytes, barrier);
                        }
                    }
                }
            };
            const loader: Loader = .{
                .x = x,
                .x_stride = x_stride,
                .other = other,
                .other_stride = other_stride,
                .rows = rows,
                .x_row_bytes = x_row_bytes,
                .addend_row_bytes = addend_row_bytes,
            };
            if (issuer) {
                bulk.initBarriers(full[0..stages]);
                var k: u32 = 0;
                while (k < @min(stages, count)) : (k += 1) {
                    loader.issue(first + k * step, buffers + k * stage_bytes, &full[k]);
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            var k: u32 = 0;
            while (k < count) : (k += 1) {
                const tile = first + k * step;
                const stage = k % stages;
                const buffer = buffers + stage * stage_bytes;
                const row = tile * tile_rows + @workItemId(1);
                const position = positionOf(row, rows, width, rows_per_sample);
                var factors: Factors = .{
                    .weight = weight,
                    .branch_weight = branch_weight,
                    .scale = rowOf(.bf16, scale, position.sample, scale_stride),
                    .shift = rowOf(.bf16, shift, position.sample, shift_stride),
                };
                factors.preload(position);
                bulk.wait(&full[stage], k / stages);
                const h = loadSharedRow(options.x, position, buffer + @workItemId(1) * x_row_bytes);
                var addend: Row(addend_element) = undefined;
                if (options.add != .none) {
                    const addends = buffer + tile_rows * x_row_bytes;
                    addend = loadSharedRow(addend_element, position, addends + @workItemId(1) * addend_row_bytes);
                }
                nvptx.barrierSyncAligned(.work_group, null);
                if (issuer and k + stages < count) {
                    loader.issue(tile + stages * step, buffer, &full[stage]);
                }
                finish(.{
                    .other = other,
                    .other_stride = other_stride,
                    .residual_out = residual_out,
                    .residual_out_stride = residual_out_stride,
                    .gate = gate,
                    .gate_stride = gate_stride,
                    .y = y,
                    .y_stride = y_stride,
                    .y_scales = y_scales,
                    .y_scale_stride = y_scale_stride,
                    .rows = rows,
                    .width = width,
                    .epsilon = epsilon,
                }, position, factors, h, addend);
            }
        }

        inline fn loadSharedRow(
            comptime element: Element,
            position: anytype,
            start: [*]addrspace(.shared) const u8,
        ) Row(element) {
            const chunks: [*]addrspace(.shared) const Packed(element) = @ptrCast(@alignCast(start));
            var row: Row(element) = undefined;
            inline for (&row, 0..) |*bits, slot| {
                if (position.holds(slot)) bits.* = chunks[position.chunk(slot)];
            }
            return row;
        }

        /// Adds the addend, normalizes `h` and stores the results of a row.
        inline fn finish(
            operands: Operands,
            position: anytype,
            factors: Factors,
            loaded_h: Row(options.x),
            loaded_addend: Row(addend_element),
        ) void {
            var h = loaded_h;
            const row = position.row;
            const rows = operands.rows;
            const width = operands.width;
            const epsilon = operands.epsilon;
            const y = operands.y;
            const y_stride = operands.y_stride;
            const y_scales = operands.y_scales;
            const y_scale_stride = operands.y_scale_stride;
            if (options.add != .none) {
                var addend = loaded_addend;
                if (normalized_branch) {
                    const branch_r = reciprocalRoot(.bf16, position, addend, 0, width, epsilon);
                    normalizeBranch(position, &addend, branch_r, factors);
                }
                if (gated) {
                    const gate_row = rowOf(.bf16, operands.gate, position.sample, operands.gate_stride);
                    multiplyGate(position, &addend, gate_row);
                }
                const residual_out_row =
                    rowOf(options.x, operands.residual_out, row, operands.residual_out_stride);
                addToResidualStream(position, &h, addend, residual_out_row);
            }
            const r = reciprocalRoot(options.x, position, h, 1, width, epsilon);
            if (options.output == .fp8_rowwise) {
                const row_scale = y_scales + @as(u64, row) * y_scale_stride;
                const codes = fp8Row(y, row, y_stride);
                return quantizeRow(position, h, r, factors, codes, row_scale, row < rows);
            }
            const codes = fp8Row(y, row, y_stride);
            if ((options.output == .fp8_blockwise or options.output == .fp8_blockwise_e8m0) and slots >= 2) {
                return quantizeGroups(position, h, r, factors, codes, y_scales, y_scale_stride);
            }
            if (options.output == .fp8_tensorwise) {
                return quantizeTensorwise(position, h, r, factors, codes, y_scales[0]);
            }
            inline for (h, 0..) |h_bits, slot| {
                const chunk = position.chunk(slot);
                const holds = position.holds(slot);
                switch (options.output) {
                    .bf16 => if (holds) {
                        const y_row = rowOf(.bf16, y, row, y_stride);
                        store(.bf16, y_row, chunk, outputBf16(h_bits, r, factors, position, slot));
                    },
                    .f32 => if (holds) {
                        const y_row = rowOf(.f32, y, row, y_stride);
                        store(.f32, y_row, chunk, pack(.f32, outputF32(h_bits, r, factors, position, slot)));
                    },
                    .fp8_tensorwise, .fp8_rowwise => unreachable,
                    .fp8_blockwise, .fp8_blockwise_e8m0 => {
                        var words: Packed(.bf16) = @splat(0);
                        if (holds) words = outputBf16(h_bits, r, factors, position, slot);
                        const amax = fp8.maxNanOverLanes(fp8.amaxOfBf16(words), chunks_per_scale_group);
                        const group_scale = if (scale_format == .e8m0)
                            fp8.e8m0OfBf16Amax(amax)
                        else
                            fp8.scaleOf(scale_format, amax);
                        if (holds) {
                            const quantizer = fp8.Quantizer(scale_format).init(group_scale);
                            codes[chunk] = quantizer.codes(words);
                            if (chunk % chunks_per_scale_group == 0) {
                                const scale_group_index: u64 = chunk / chunks_per_scale_group;
                                y_scales[scale_group_index * y_scale_stride + row] = group_scale;
                            }
                        }
                    },
                }
            }
        }

        /// The slots of a work item lie in `slots` scale groups, each shared by the 16 work items
        /// `16 * k` to `16 * k + 15` of the row. The work items of a group form blocks of
        /// `scale_block` work items. Work item `j` of a block computes the scale of slot
        /// `j % slots`, every work item takes the scales from the first `slots` work items of its
        /// block, and work item `16 * k + s` stores the scale of slot `s`. The amaxes of two slots
        /// travel as the BF16 halves of one word, which hold them exactly because the values are
        /// BF16.
        inline fn quantizeGroups(
            position: anytype,
            h: Row(options.x),
            r: f32,
            factors: Factors,
            codes: [*]addrspace(.global) fp8.Codes,
            y_scales: ScaleOutput,
            y_scale_stride: u32,
        ) void {
            comptime std.debug.assert(slots % 2 == 0 and slots <= chunks_per_scale_group);
            const Quantizer = fp8.Quantizer(scale_format);
            var words: [slots]Packed(.bf16) = undefined;
            inline for (&words, h, 0..) |*word, h_bits, slot| {
                word.* = @splat(0);
                if (position.holds(slot)) word.* = outputBf16(h_bits, r, factors, position, slot);
            }
            var amaxes: [slots / 2]u32 = undefined;
            inline for (&amaxes, 0..) |*amax, index| {
                const low: u32 = @bitCast(fp8.amaxOfBf16(words[2 * index]));
                const high: u32 = @bitCast(fp8.amaxOfBf16(words[2 * index + 1]));
                const pair_amax = (low >> 16) | (high & 0xffff_0000);
                amax.* = fp8.maxMagnitudesOverLanes(pair_amax, chunks_per_scale_group);
            }
            const scale_block =
                if (chunks_per_scale_group % slots == 0) slots else chunks_per_scale_group;
            const own_slot = position.lane % scale_block % slots;
            var own_amax: u32 = 0;
            inline for (0..slots) |slot| {
                const word = amaxes[slot / 2];
                if (own_slot == slot) {
                    own_amax = if (slot % 2 == 0) word << 16 else word & 0xffff_0000;
                }
            }
            const own_scale = if (scale_format == .e8m0)
                fp8.e8m0OfBf16Amax(@bitCast(own_amax & 0x7fff_ffff))
            else
                fp8.scaleOf(.f32, @bitCast(own_amax & 0x7fff_ffff));
            const own_shared = Quantizer.sharedPart(own_scale);
            const warp_lane = (@workItemId(1) * @workGroupSize(0) + @workItemId(0)) % 32;
            const group_first = warp_lane - warp_lane % scale_block;
            var quantizers: [slots]Quantizer = undefined;
            inline for (&quantizers, 0..) |*quantizer, slot| {
                const source = group_first + @as(u32, slot);
                const shared = shuffleIndex(u32, own_shared, source);
                const scale: fp8.Scale(scale_format) = if (scale_format == .f32)
                    shuffleIndex(f32, own_scale, source)
                else
                    0;
                quantizer.* = .fromShared(scale, shared);
            }
            storeCodes(Quantizer, position, codes, quantizers, words);
            const group_lane = position.lane % chunks_per_scale_group;
            const first_chunk = position.lane - group_lane + group_lane * position.threads_per_row;
            if (group_lane < slots and first_chunk < position.chunks) {
                const scale_group_index: u64 = first_chunk / chunks_per_scale_group;
                y_scales[scale_group_index * y_scale_stride + position.row] = own_scale;
            }
        }

        /// Every work item of a row needs the scale of the whole row. Each work item therefore
        /// keeps its results in registers until `overRow` returns the row maximum.
        inline fn quantizeRow(
            position: anytype,
            h: Row(options.x),
            r: f32,
            factors: Factors,
            codes: [*]addrspace(.global) fp8.Codes,
            row_scale: ScaleOutput,
            in_rows: bool,
        ) void {
            var results: [slots]Packed(.bf16) = undefined;
            var amax: f32 = 0;
            inline for (&results, h, 0..) |*result, h_bits, slot| {
                result.* = @splat(0);
                if (position.holds(slot)) result.* = outputBf16(h_bits, r, factors, position, slot);
                amax = fp8.maxNan(amax, fp8.amaxOfBf16(result.*));
            }
            const scale = fp8.scaleOf(.f32, overRow(fp8.maxNan, position, amax, 0));
            if (in_rows and position.lane == 0) row_scale[0] = scale;
            const Quantizer = fp8.Quantizer(.f32);
            storeCodes(Quantizer, position, codes, @splat(.init(scale)), results);
        }

        /// Quantizes the slots of a work item with the tensorwise `scale`, choosing the division
        /// method once from the largest magnitude of the BF16 results of the work item.
        inline fn quantizeTensorwise(
            position: anytype,
            h: Row(options.x),
            r: f32,
            factors: Factors,
            codes: [*]addrspace(.global) fp8.Codes,
            scale: f32,
        ) void {
            var words: [slots]Packed(.bf16) = undefined;
            var all_words: @Vector(slots * pairs_per_slot, u32) = @splat(0);
            inline for (&words, h, 0..) |*word, h_bits, slot| {
                word.* = @splat(0);
                if (position.holds(slot)) word.* = outputBf16(h_bits, r, factors, position, slot);
                inline for (0..pairs_per_slot) |index| {
                    all_words[slot * pairs_per_slot + index] = word.*[index];
                }
            }
            const quantizer: fp8.TensorwiseQuantizer = .init(scale);
            if (quantizer.isPlainUpTo(fp8.amaxOfBf16(all_words))) {
                inline for (words, 0..) |word, slot| {
                    if (position.holds(slot)) {
                        codes[position.chunk(slot)] = quantizer.plainCodes(word);
                    }
                }
            } else {
                inline for (words, 0..) |word, slot| {
                    if (position.holds(slot)) codes[position.chunk(slot)] = quantizer.codes(word);
                }
            }
        }

        /// Stores the codes of the slots that the work item holds, choosing the division method
        /// once for all of them.
        inline fn storeCodes(
            comptime Quantizer: type,
            position: anytype,
            codes: [*]addrspace(.global) fp8.Codes,
            quantizers: [slots]Quantizer,
            words: [slots]Packed(.bf16),
        ) void {
            var plain = true;
            inline for (quantizers, 0..) |quantizer, slot| {
                if (position.holds(slot)) plain = plain and quantizer.isPlain();
            }
            if (plain) {
                inline for (quantizers, words, 0..) |quantizer, word, slot| {
                    if (position.holds(slot)) {
                        codes[position.chunk(slot)] = quantizer.plainCodes(word);
                    }
                }
            } else {
                inline for (quantizers, words, 0..) |quantizer, word, slot| {
                    if (position.holds(slot)) codes[position.chunk(slot)] = quantizer.codes(word);
                }
            }
        }

        inline fn loadRow(
            comptime element: Element,
            position: anytype,
            start: anytype,
        ) Row(element) {
            var row: Row(element) = undefined;
            inline for (&row, 0..) |*bits, slot| {
                if (position.holds(slot)) bits.* = load(element, start, position.chunk(slot));
            }
            return row;
        }

        /// Returns `rsqrt(mean(v * v) + epsilon)` over the values `v` of the row. The branch norm
        /// of the dual kernels uses half 0 of `partials`, and the norm of `h` uses half 1.
        inline fn reciprocalRoot(
            comptime element: Element,
            position: anytype,
            row: Row(element),
            comptime half: u1,
            width: u32,
            epsilon: f32,
        ) f32 {
            var squares: Pair = @splat(0);
            inline for (row, 0..) |bits, slot| {
                if (position.holds(slot)) {
                    if (element == .bf16 and bf16.mixed_fma) {
                        inline for (0..pairs_per_slot) |word| {
                            squares[0] = bf16.fmaHalves(0, bits[word], bits[word], squares[0]);
                            squares[1] = bf16.fmaHalves(1, bits[word], bits[word], squares[1]);
                        }
                    } else {
                        for (unpack(element, bits)) |values| squares = pair.fma(values, values, squares);
                    }
                }
            }
            const sum = overRow(add, position, squares[0] + squares[1], half);
            return nvptx.math.rsqrtApproxF32(false, sum / @as(f32, @floatFromInt(width)) + epsilon);
        }

        /// Combines the values of the work items of a row with `combine`. Rows of whole warps
        /// exchange the warp results through `half` of `partials`. A kernel that calls `overRow`
        /// twice with one half calls `overRow` with the other half between the two calls. The
        /// barrier of the middle call keeps every warp from storing into a half while another warp
        /// still reads the earlier results from that half (PTX ISA 9.7.15.1).
        inline fn overRow(
            comptime combine: fn (f32, f32) callconv(.@"inline") f32,
            position: anytype,
            value: f32,
            comptime half: u1,
        ) f32 {
            var result = value;
            if (position.threads_per_row <= 32) {
                var mask = position.threads_per_row / 2;
                while (mask > 0) : (mask /= 2) result = combine(result, shuffleXor(result, mask));
                return result;
            }
            const warps_per_row = position.threads_per_row / 32;
            const first_warp = @workItemId(1) * warps_per_row;
            result = overWarp(combine, result);
            if (position.lane % 32 == 0) partials[half][first_warp + position.lane / 32] = result;
            nvptx.barrierSyncAligned(.work_group, null);
            const lane = position.lane % 32;
            return overWarp(combine, if (lane < warps_per_row) partials[half][first_warp + lane] else 0);
        }

        inline fn normalizeBranch(
            position: anytype,
            branch: *Row(.bf16),
            r: f32,
            factors: Factors,
        ) void {
            inline for (branch, 0..) |*bits, slot| {
                if (position.holds(slot)) {
                    const gains = factors.branchGainBits(position, slot);
                    bits.* = normalizedBf16(.bf16, unpack(.bf16, bits.*), r, gains);
                }
            }
        }

        inline fn multiplyGate(
            position: anytype,
            addend: *Row(.bf16),
            gate_row: OptionalInput,
        ) void {
            inline for (addend, 0..) |*bits, slot| {
                if (position.holds(slot)) {
                    const gates = load(.bf16, gate_row, position.chunk(slot));
                    inline for (0..pairs_per_slot) |word| {
                        bits[word] = @backingInt(Bf16x2.mul(@fromBackingInt(gates[word]), @fromBackingInt(bits[word])));
                    }
                }
            }
        }

        /// A BF16 sum of BF16 values rounds once, as the BF16 rounding of their exact FP32 sum
        /// does.
        inline fn addToResidualStream(
            position: anytype,
            h: *Row(options.x),
            addend: Row(addend_element),
            residual_out_row: OptionalOutput,
        ) void {
            inline for (h, addend, 0..) |*h_bits, addend_bits, slot| {
                if (position.holds(slot)) {
                    if (options.x == .bf16 and addend_element == .bf16) {
                        inline for (0..pairs_per_slot) |word| {
                            const sum = Bf16x2.add(@fromBackingInt(h_bits[word]), @fromBackingInt(addend_bits[word]));
                            h_bits[word] = @backingInt(sum);
                        }
                    } else {
                        var sums = unpack(options.x, h_bits.*);
                        for (&sums, unpack(addend_element, addend_bits)) |*sum, value| sum.* = pair.add(sum.*, value);
                        h_bits.* = pack(options.x, sums);
                    }
                    store(options.x, residual_out_row, position.chunk(slot), h_bits.*);
                }
            }
        }

        /// Returns the BF16 output of a slot of `h`. The modulation computes in BF16, where each
        /// operation on BF16 operands rounds once, as the FP32 operation and the BF16 rounding of
        /// its result do.
        inline fn outputBf16(
            h_bits: Packed(options.x),
            r: f32,
            factors: Factors,
            position: anytype,
            comptime slot: u32,
        ) Packed(.bf16) {
            const gains = factors.gainBits(position, slot);
            var bits = normalizedBf16(options.x, unpack(options.x, h_bits), r, gains);
            if (options.modulation != .none) {
                const scales = factors.scaleBits(position, slot);
                inline for (0..pairs_per_slot) |word| {
                    const factor = Bf16x2.add(@fromBackingInt(scales[word]), bf16_ones);
                    bits[word] = @backingInt(Bf16x2.mul(@fromBackingInt(bits[word]), factor));
                }
            }
            if (options.modulation == .scale_shift) {
                const shifts = factors.shiftBits(position, slot);
                inline for (0..pairs_per_slot) |word| {
                    const shifted = Bf16x2.add(@fromBackingInt(bits[word]), @fromBackingInt(shifts[word]));
                    bits[word] = @backingInt(shifted);
                }
            }
            return bits;
        }

        inline fn outputF32(
            h_bits: Packed(options.x),
            r: f32,
            factors: Factors,
            position: anytype,
            comptime slot: u32,
        ) SlotPairs {
            return normalizedPairs(options.x, unpack(options.x, h_bits), r, factors.gainBits(position, slot));
        }

        /// Returns the BF16 bits of `normalizedPairs`. With BF16 weights and BF16 input, the
        /// product of the weight and the rounded `h * r` is exact in FP32, so one BF16 multiply
        /// gives its BF16 rounding.
        inline fn normalizedBf16(
            comptime input: Element,
            values: SlotPairs,
            r: f32,
            gains: Packed(weight_element),
        ) Packed(.bf16) {
            var bits: Packed(.bf16) = undefined;
            if (options.weight == .bf16 and input == .bf16) {
                const factor = pair.splat(r);
                inline for (values, 0..) |value, word| {
                    const scaled = pair.mul(value, factor);
                    const rounded: Bf16x2 = @fromBackingInt(bf16.pack(scaled[0], scaled[1]));
                    bits[word] = @backingInt(Bf16x2.mul(@fromBackingInt(gains[word]), rounded));
                }
                return bits;
            }
            const results = normalizedPairs(input, values, r, gains);
            inline for (results, 0..) |result, word| bits[word] = bf16.pack(result[0], result[1]);
            return bits;
        }

        inline fn normalizedPairs(
            comptime input: Element,
            values: SlotPairs,
            r: f32,
            gains: Packed(weight_element),
        ) SlotPairs {
            const factor = pair.splat(r);
            const weights: SlotPairs = if (options.weight == .none) undefined else unpack(weight_element, gains);
            var results: SlotPairs = undefined;
            inline for (&results, values, 0..) |*result, value, index| {
                const scaled = pair.mul(value, factor);
                result.* = switch (options.weight) {
                    .none => scaled,
                    .bf16 => pair.mul(weights[index], roundTo(input, scaled)),
                    .f32 => pair.mul(weights[index], scaled),
                    .zero_centered_bf16, .zero_centered_f32 => pair.mul(scaled, pair.add(pair.splat(1), weights[index])),
                };
            }
            return results;
        }

        /// Loads the values of chunk `chunk` from a row of `element` values at `start`.
        inline fn load(comptime element: Element, start: anytype, chunk: u32) Packed(element) {
            switch (Packed(element)) {
                HalfChunk => {
                    const halves: [*]allowzero addrspace(.global) const HalfChunk = @ptrCast(start);
                    return halves[chunk];
                },
                Chunk => return start[chunk],
                else => return .{ start[2 * chunk], start[2 * chunk + 1] },
            }
        }

        inline fn store(
            comptime element: Element,
            start: anytype,
            chunk: u32,
            bits: Packed(element),
        ) void {
            switch (Packed(element)) {
                HalfChunk => {
                    const halves: [*]allowzero addrspace(.global) HalfChunk = @ptrCast(start);
                    halves[chunk] = bits;
                },
                Chunk => start[chunk] = bits,
                else => {
                    start[2 * chunk] = bits[0];
                    start[2 * chunk + 1] = bits[1];
                },
            }
        }

        inline fn unpack(comptime element: Element, bits: Packed(element)) SlotPairs {
            var values: SlotPairs = undefined;
            switch (element) {
                .bf16 => inline for (&values, 0..) |*value, word| {
                    value.* = bf16.unpack(bits[word]);
                },
                .f32 => {
                    var floats: [values_per_slot]f32 = undefined;
                    inline for (0..values_per_slot / 4) |part| {
                        const words = if (Packed(element) == Chunk) bits else bits[part];
                        floats[4 * part ..][0..4].* = @bitCast(words);
                    }
                    inline for (&values, 0..) |*value, index| {
                        value.* = .{ floats[2 * index], floats[2 * index + 1] };
                    }
                },
            }
            return values;
        }

        inline fn pack(comptime element: Element, values: SlotPairs) Packed(element) {
            var bits: Packed(element) = undefined;
            switch (element) {
                .bf16 => inline for (values, 0..) |value, word| {
                    bits[word] = bf16.pack(value[0], value[1]);
                },
                .f32 => {
                    var floats: [values_per_slot]f32 = undefined;
                    inline for (values, 0..) |value, index| {
                        floats[2 * index] = value[0];
                        floats[2 * index + 1] = value[1];
                    }
                    inline for (0..values_per_slot / 4) |part| {
                        const words: Chunk = @bitCast(floats[4 * part ..][0..4].*);
                        if (Packed(element) == Chunk) bits = words else bits[part] = words;
                    }
                },
            }
            return bits;
        }
    };
}

inline fn shuffleXor(x: f32, mask: u32) f32 {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return nvptx.shflSync(.bfly, f32, x, mask, clamp, .all);
}

inline fn shuffleIndex(comptime T: type, x: T, lane: u32) T {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return nvptx.shflSync(.idx, T, x, lane, clamp, .all);
}

inline fn overWarp(comptime combine: fn (f32, f32) callconv(.@"inline") f32, x: f32) f32 {
    var result = x;
    inline for (.{ 16, 8, 4, 2, 1 }) |mask| result = combine(result, warp.shuffleXor(result, mask));
    return result;
}

inline fn add(a: f32, b: f32) f32 {
    return a + b;
}

inline fn roundTo(comptime element: Element, x: Pair) Pair {
    return if (element == .bf16) bf16.unpack(bf16.pack(x[0], x[1])) else x;
}

inline fn rowOf(comptime element: Element, base: anytype, row: u32, stride: u32) @TypeOf(base) {
    const elements_per_chunk = @sizeOf(Chunk) / @as(u32, if (element == .bf16) 2 else 4);
    return base + @as(u64, row) * (stride / elements_per_chunk);
}

inline fn fp8Row(base: Output, row: u32, stride: u32) [*]addrspace(.global) fp8.Codes {
    const codes: [*]addrspace(.global) fp8.Codes = @ptrCast(base);
    return codes + @as(u64, row) * (stride / @sizeOf(fp8.Codes));
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        const config = @import("config");
        const options = configurations.instances[config.variant % configurations.instances.len];
        const engine: configurations.Engine = @fromBackingInt(@intCast(config.variant / configurations.instances.len));
        for (configurations.sizesOf(options)) |size| {
            const name = configurations.kernelName(options, size, engine);
            const Kernel = RmsNorm(options, size);
            const entry = switch (engine) {
                .registers => &Kernel.kernel,
                .bulk => &Kernel.bulkKernel,
            };
            @export(entry, .{ .name = name });
        }
    }
}
