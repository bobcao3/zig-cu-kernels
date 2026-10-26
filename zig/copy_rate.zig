//! Copy kernels that measure the copy rate of a device. The register kernels move 16-byte chunks
//! with `ld.global.v4` and `st.global.v4`. The bulk kernel stages tiles in shared memory with bulk
//! copies, one thread of each work group issuing all copies of the work group.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const config = @import("config");
const configurations = @import("copy_rate_configurations.zig");
const bulk = @import("bulk.zig");
const pair = @import("pair.zig");

const Chunk = @Vector(4, u32);

fn Registers(comptime unroll: u32) type {
    return struct {
        /// Each work group copies `unroll` consecutive blocks of `@workGroupSize(0)` chunks per pass.
        fn kernel(
            source: [*]addrspace(.global) const Chunk,
            target: [*]addrspace(.global) Chunk,
            chunks: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = configurations.register_threads },
        } }) void {
            const threads = @workGroupSize(0);
            const step = threads * unroll * nvptx.nctaid(0);
            var first = @workGroupId(0) * threads * unroll + @workItemId(0);
            while (first < chunks) : (first += step) {
                var values: [unroll]Chunk = undefined;
                inline for (&values, 0..) |*value, block| {
                    const index = first + block * threads;
                    if (index < chunks) value.* = source[index];
                }
                inline for (values, 0..) |value, block| {
                    const index = first + block * threads;
                    if (index < chunks) target[index] = value;
                }
            }
        }
    };
}

/// Each chunk of `source` yields 8 bytes of `target`, the 2:1 mix of reads and writes of a norm with
/// an FP8 output.
fn Halve(comptime unroll: u32) type {
    return struct {
        fn kernel(
            source: [*]addrspace(.global) const Chunk,
            target: [*]addrspace(.global) @Vector(2, u32),
            chunks: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = configurations.register_threads },
        } }) void {
            const threads = @workGroupSize(0);
            const step = threads * unroll * nvptx.nctaid(0);
            var first = @workGroupId(0) * threads * unroll + @workItemId(0);
            while (first < chunks) : (first += step) {
                var values: [unroll]Chunk = undefined;
                inline for (&values, 0..) |*value, block| {
                    const index = first + block * threads;
                    if (index < chunks) value.* = source[index];
                }
                inline for (values, 0..) |value, block| {
                    const index = first + block * threads;
                    if (index < chunks) target[index] = .{ value[0] ^ value[1], value[2] ^ value[3] };
                }
            }
        }
    };
}

/// Reads every chunk of `source` and writes `target[0]` only when the XOR of all words of the
/// work item equals `never`, which the caller picks so that the write does not happen.
fn Read(comptime unroll: u32) type {
    return struct {
        fn kernel(
            source: [*]addrspace(.global) const Chunk,
            target: [*]addrspace(.global) u32,
            chunks: u32,
            never: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = configurations.register_threads },
        } }) void {
            const threads = @workGroupSize(0);
            const step = threads * unroll * nvptx.nctaid(0);
            var first = @workGroupId(0) * threads * unroll + @workItemId(0);
            var accumulator: Chunk = @splat(0);
            while (first < chunks) : (first += step) {
                inline for (0..unroll) |block| {
                    const index = first + block * threads;
                    if (index < chunks) accumulator ^= source[index];
                }
            }
            const folded = accumulator[0] ^ accumulator[1] ^ accumulator[2] ^ accumulator[3];
            if (folded == never) target[0] = folded;
        }
    };
}

/// Each chunk of 8 BF16 values of `source` yields 8 E4M3 codes of `target`: the values times
/// `factor`, converted with `cvt.rn.satfinite.e4m3x2.f32`. The kernel measures the rate of the
/// conversion at the 2:1 mix of a norm with an FP8 output.
fn Cast(comptime unroll: u32) type {
    return struct {
        fn kernel(
            source: [*]addrspace(.global) const Chunk,
            target: [*]addrspace(.global) @Vector(2, u32),
            chunks: u32,
            factor: f32,
        ) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = configurations.register_threads },
        } }) void {
            const threads = @workGroupSize(0);
            const step = threads * unroll * nvptx.nctaid(0);
            var first = @workGroupId(0) * threads * unroll + @workItemId(0);
            while (first < chunks) : (first += step) {
                var values: [unroll]Chunk = undefined;
                inline for (&values, 0..) |*value, block| {
                    const index = first + block * threads;
                    if (index < chunks) value.* = source[index];
                }
                inline for (values, 0..) |value, block| {
                    const index = first + block * threads;
                    if (index < chunks) target[index] = castChunk(value, factor);
                }
            }
        }
    };
}

/// The kernel of `Cast` with one more conversion per pair: the products round to BF16 with
/// `cvt.rn.bf16x2.f32` before they convert to E4M3, as the norms with an FP8 output round their
/// BF16 result before they quantize it.
fn CastRounded(comptime unroll: u32) type {
    return struct {
        fn kernel(
            source: [*]addrspace(.global) const Chunk,
            target: [*]addrspace(.global) @Vector(2, u32),
            chunks: u32,
            factor: f32,
        ) callconv(.{ .nvptx_kernel = .{
            .max_work_group_size = .{ .x = configurations.register_threads },
        } }) void {
            const threads = @workGroupSize(0);
            const step = threads * unroll * nvptx.nctaid(0);
            var first = @workGroupId(0) * threads * unroll + @workItemId(0);
            while (first < chunks) : (first += step) {
                var values: [unroll]Chunk = undefined;
                inline for (&values, 0..) |*value, block| {
                    const index = first + block * threads;
                    if (index < chunks) value.* = source[index];
                }
                inline for (values, 0..) |value, block| {
                    const index = first + block * threads;
                    if (index < chunks) {
                        var rounded: Chunk = undefined;
                        inline for (0..4) |word| {
                            const products = pair.mul(nvptx.Bf16x2.toF32(@fromBackingInt(value[word])), pair.splat(factor));
                            rounded[word] = @backingInt(nvptx.Bf16x2.fromF32(products[0], products[1]));
                        }
                        target[index] = castChunk(rounded, 1.0);
                    }
                }
            }
        }
    };
}

inline fn castChunk(words: Chunk, factor: f32) @Vector(2, u32) {
    var quotients: [4]pair.Pair = undefined;
    inline for (&quotients, 0..) |*quotient, word| {
        quotient.* = pair.mul(nvptx.Bf16x2.toF32(@fromBackingInt(words[word])), pair.splat(factor));
    }
    var codes: @Vector(2, u32) = undefined;
    inline for (0..2) |half| {
        const low = quotients[2 * half];
        const high = quotients[2 * half + 1];
        codes[half] = @backingInt(nvptx.E4m3x4.fromF32(low[0], low[1], high[0], high[1]));
    }
    return codes;
}

extern var shared_storage: [0]u8 align(128) addrspace(.shared);
var full: [configurations.max_stages]bulk.Mbarrier addrspace(.shared) = undefined;

/// Work group `g` copies the tiles `g`, `g + nctaid`, and so on, through `stages` buffers of
/// `tile_bytes` bytes. A buffer receives its next tile once the bulk store of its previous tile has
/// read it. Assumes `1 <= stages <= max_stages` and a multiple of 16 for `tile_bytes`.
fn bulkKernel(
    source: [*]addrspace(.global) const u8,
    target: [*]addrspace(.global) u8,
    tiles: u32,
    tile_bytes: u32,
    stages: u32,
) callconv(.{ .nvptx_kernel = .{ .max_work_group_size = .{ .x = 32 } } }) void {
    const first = @workGroupId(0);
    if (@workItemId(0) != 0 or first >= tiles) return;
    const step = nvptx.nctaid(0);
    const count = (tiles - first + step - 1) / step;
    const buffers: [*]addrspace(.shared) u8 = &shared_storage;
    bulk.initBarriers(full[0..stages]);
    const Tiles = struct {
        fn offset(k: u32, origin: u32, stride: u32, bytes: u32) u64 {
            return @as(u64, origin + k * stride) * bytes;
        }
    };
    var k: u32 = 0;
    while (k < @min(stages, count)) : (k += 1) {
        _ = full[k].arriveExpectTx(tile_bytes);
        const offset = Tiles.offset(k, first, step, tile_bytes);
        bulk.load(buffers + k * tile_bytes, source + offset, tile_bytes, &full[k]);
    }
    k = 0;
    while (k < count) : (k += 1) {
        const stage = k % stages;
        bulk.wait(&full[stage], k / stages);
        const offset = Tiles.offset(k, first, step, tile_bytes);
        bulk.store(target + offset, buffers + stage * tile_bytes, tile_bytes);
        bulk.commitGroup();
        if (k >= 1 and k - 1 + stages < count) {
            const next = k - 1 + stages;
            const next_stage = (k - 1) % stages;
            bulk.waitGroup(1, true);
            _ = full[next_stage].arriveExpectTx(tile_bytes);
            const next_offset = Tiles.offset(next, first, step, tile_bytes);
            bulk.load(buffers + next_stage * tile_bytes, source + next_offset, tile_bytes, &full[next_stage]);
        }
    }
    bulk.waitGroup(0, false);
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        switch (@as(configurations.Kind, @fromBackingInt(@intCast(config.variant)))) {
            .registers => for (configurations.unrolls) |unroll| {
                @export(&Registers(unroll).kernel, .{ .name = configurations.registerKernelName(unroll) });
                @export(&Halve(unroll).kernel, .{ .name = configurations.halveKernelName(unroll) });
                @export(&Read(unroll).kernel, .{ .name = configurations.readKernelName(unroll) });
            },
            .bulk => @export(&bulkKernel, .{ .name = configurations.bulk_kernel_name }),
            .cast => for (configurations.unrolls) |unroll| {
                @export(&Cast(unroll).kernel, .{ .name = configurations.castKernelName(unroll) });
                @export(&CastRounded(unroll).kernel, .{ .name = configurations.castRoundedKernelName(unroll) });
            },
        }
    }
}
