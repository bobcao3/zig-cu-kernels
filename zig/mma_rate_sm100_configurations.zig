const configurations = @import("configurations.zig");

pub const threads = 128;
pub const stages = 4;
pub const k = 128;
pub const steps = k / 32;
pub const stage_bytes = 2 * 128 * k;
pub const scale_stage_bytes = 3 * 512;
pub const shared_bytes = 1024 + stages * (stage_bytes + scale_stage_bytes);
pub const flops_per_batch = stages * steps * 2 * 256 * 256 * 32;

pub const dense_name = "mma_rate_sm100_f8f6f4";
pub const scaled_name = "mma_rate_sm100_mxf8f6f4";

pub const modules: []const configurations.Module = &.{.{
    .name = "mma_rate_sm100",
    .source = "mma_rate_sm100",
    .arch = 100,
    .targets = &.{"sm_100f"},
    .kernels = &.{
        .{ .name = dense_name, .launch = .{ .threads = threads, .shared_bytes = shared_bytes } },
        .{ .name = scaled_name, .launch = .{ .threads = threads, .shared_bytes = shared_bytes } },
    },
}};
