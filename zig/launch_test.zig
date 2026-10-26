//! A kernel that copies its parameters, among them a 128-byte tensor map, to global memory.

const builtin = @import("builtin");
const configurations = @import("launch_test_configurations.zig");

pub const TensorMap = extern struct {
    bytes: [configurations.tensor_map_words * 4]u8 align(64),
};

fn copyParameters(
    head: u32,
    map: TensorMap,
    target: [*]addrspace(.global) u32,
    tail: u32,
) callconv(.{ .nvptx_kernel = .{ .max_work_group_size = .{ .x = 32 } } }) void {
    if (@workItemId(0) != 0 or @workGroupId(0) != 0) return;
    const words: [configurations.tensor_map_words]u32 = @bitCast(map.bytes);
    target[0] = head;
    inline for (words, 1..) |word, index| target[index] = word;
    target[words.len + 1] = tail;
}

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        @export(&copyParameters, .{ .name = configurations.parameter_copy_kernel_name });
    }
}
