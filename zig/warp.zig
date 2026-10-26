const std = @import("std");
const nvptx = std.nvptx;

pub inline fn shuffleXor(x: f32, comptime mask: u32) f32 {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return nvptx.shflSync(.bfly, f32, x, mask, clamp, .all);
}

pub inline fn sum(x: f32) f32 {
    var total = x;
    inline for (.{ 16, 8, 4, 2, 1 }) |mask| total += shuffleXor(total, mask);
    return total;
}

pub inline fn max(x: f32) f32 {
    var result = x;
    inline for (.{ 16, 8, 4, 2, 1 }) |mask| result = @max(result, shuffleXor(result, mask));
    return result;
}
