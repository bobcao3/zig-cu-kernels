const configurations = @import("configurations.zig");

pub const Recipe = enum { mxfp8, fp8_b128_e8m0, fp8_b128_fp32, fp8_rowwise };

pub const Epilogue = enum { store, bias, residual };

pub const threads = 128;
pub const tile_columns = 128;
pub const rows_per_work_group = 4;
pub const columns_per_thread = 4;

pub fn kernelName(comptime recipe: Recipe, comptime epilogue: Epilogue) []const u8 {
    return @tagName(recipe) ++ "_decode_reduce_" ++ @tagName(epilogue);
}

pub const modules: []const configurations.Module = &.{.{
    .name = "decode_reduce",
    .source = "decode_reduce",
    .arch = 89,
    .targets = &.{ "sm_89", "sm_90a", "sm_100f", "sm_120f" },
    .kernels = kernels(),
}};

fn kernels() []const configurations.Kernel {
    var list: []const configurations.Kernel = &.{};
    for (@typeInfo(Recipe).@"enum".field_names) |recipe| {
        for (@typeInfo(Epilogue).@"enum".field_names) |epilogue| {
            list = list ++ .{configurations.Kernel{
                .name = recipe ++ "_decode_reduce_" ++ epilogue,
                .launch = .{ .threads = threads, .tile_n = tile_columns, .tile_m = rows_per_work_group },
            }};
        }
    }
    return list;
}
