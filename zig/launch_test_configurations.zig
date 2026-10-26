const configurations = @import("configurations.zig");

pub const parameter_copy_kernel_name = "launch_test_copy_parameters";
pub const tensor_map_words = 32;

pub const modules: []const configurations.Module = &.{.{
    .name = "launch_test",
    .source = "launch_test",
    .arch = 89,
    .targets = &.{ "sm_89", "sm_90a", "sm_100f", "sm_120f" },
    .kernels = &.{.{ .name = parameter_copy_kernel_name, .launch = .{ .threads = 32 } }},
}};
