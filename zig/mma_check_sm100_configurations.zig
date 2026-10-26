const configurations = @import("configurations.zig");

pub const threads = 128;
pub const k = 128;
pub const n = 256;
pub const tile_alignment = 1024;
pub const scale_atom_bytes = 512;
pub const one_cta_shared_bytes = tile_alignment + (128 + n) * k + 3 * scale_atom_bytes;
pub const pair_shared_bytes = tile_alignment + 2 * 128 * k;

pub const dense_name = "mma_check_sm100_f8f6f4";
pub const scaled_name = "mma_check_sm100_mxf8f6f4";
pub const pair_name = "mma_check_sm100_pair_f8f6f4";

pub const modules: []const configurations.Module = &.{.{
    .name = "mma_check_sm100",
    .source = "mma_check_sm100",
    .arch = 100,
    .targets = &.{"sm_100f"},
    .kernels = &.{
        .{
            .name = dense_name,
            .launch = .{ .threads = threads, .shared_bytes = one_cta_shared_bytes },
        },
        .{
            .name = scaled_name,
            .launch = .{ .threads = threads, .shared_bytes = one_cta_shared_bytes },
        },
        .{
            .name = pair_name,
            .launch = .{ .threads = threads, .shared_bytes = pair_shared_bytes },
        },
    },
}};
