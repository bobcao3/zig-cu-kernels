//! Wrappers for the fifth-generation tensor core instructions (`tcgen05`, PTX ISA 9.7.18) that the
//! FP8 GEMMs use. The functions defined here emit inline assembly and leave the target check to
//! ptxas; `fence` and `waitSyncAligned` come from `std.nvptx.tcgen05`.

const std = @import("std");
const nvptx = std.nvptx;

pub const capacity = 512;

pub const CtaGroup = enum(u2) {
    one = 1,
    two = 2,

    fn text(comptime group: CtaGroup) []const u8 {
        return std.fmt.comptimePrint(".cta_group::{d}", .{@backingInt(group)});
    }
};

pub const Swizzle = enum(u3) {
    none = 0,
    bytes_128_atom_32 = 1,
    bytes_128 = 2,
    bytes_64 = 4,
    bytes_32 = 6,

    pub fn fromBytes(comptime bytes: u32) Swizzle {
        return switch (bytes) {
            128 => .bytes_128,
            64 => .bytes_64,
            32 => .bytes_32,
            0 => .none,
            else => @compileError("swizzle widths are 0, 32, 64 or 128 bytes"),
        };
    }
};

pub const Major = enum(u1) { k = 0, mn = 1 };

/// The fields follow Table 49 of PTX ISA 9.7.18.4.1.
const SharedDescriptor = packed struct(u64) {
    start: u15 = 0,
    reserved_15: u1 = 0,
    leading_byte_offset: u15,
    reserved_31: u1 = 0,
    stride_byte_offset: u14,
    fixed: u3 = 0b001,
    base_offset: u3,
    leading_absolute: bool = false,
    lut_b_offset: u1 = 0,
    reserved_54: u7 = 0,
    swizzle: Swizzle,
};

fn encode(bytes: u32) u64 {
    return (bytes & 0x3ffff) >> 4;
}

pub const SharedLayout = struct {
    leading_byte_offset: u32,
    stride_byte_offset: u32,
    swizzle: Swizzle,
    base_offset: u3 = 0,

    fn bits(comptime layout: SharedLayout) u64 {
        if (layout.leading_byte_offset % 16 != 0 or layout.stride_byte_offset % 16 != 0) {
            @compileError("descriptor byte offsets must be multiples of 16");
        }
        const descriptor: SharedDescriptor = .{
            .leading_byte_offset = @intCast(encode(layout.leading_byte_offset)),
            .stride_byte_offset = @intCast(encode(layout.stride_byte_offset)),
            .base_offset = layout.base_offset,
            .swizzle = layout.swizzle,
        };
        return @bitCast(descriptor);
    }

    pub fn kMajor(comptime swizzle_bytes: u32) SharedLayout {
        return .{
            .leading_byte_offset = 16,
            .stride_byte_offset = 8 * swizzle_bytes,
            .swizzle = .fromBytes(swizzle_bytes),
        };
    }

    /// The layout that ThunderKittens gives `tcgen05.cp` for 32 contiguous rows of 16 bytes.
    pub const scale_atom: SharedLayout = .{
        .leading_byte_offset = 128,
        .stride_byte_offset = 128,
        .swizzle = .none,
    };

    /// Zero offsets make a `32x128b` copy repeat the first 8 rows of 16 bytes over all 32 rows.
    pub const repeated_scale_rows: SharedLayout = .{
        .leading_byte_offset = 0,
        .stride_byte_offset = 0,
        .swizzle = .none,
    };
};

pub fn sharedDescriptor(start: u32, comptime layout: SharedLayout) u64 {
    const fixed = comptime layout.bits();
    return fixed | encode(start);
}

pub fn kMajorDescriptor(start: u32, comptime swizzle_bytes: u32) u64 {
    return sharedDescriptor(start, comptime .kMajor(swizzle_bytes));
}

/// The bytes that one MMA step covers along K in a K-major row: 32 one-byte elements for
/// `.kind::f8f6f4` and `.kind::mxf8f6f4`, or 16 two-byte elements for `.kind::f16`.
pub const k_step_bytes = 32;

pub fn advance(descriptor: u64, comptime bytes: u32) u64 {
    return descriptor + (bytes >> 4);
}

pub fn kMajorOffsets(
    comptime steps: u32,
    comptime rows: u32,
    comptime swizzle_bytes: u32,
) [steps]u32 {
    const steps_per_box = swizzle_bytes / k_step_bytes;
    var offsets: [steps]u32 = undefined;
    for (&offsets, 0..) |*offset, step| {
        const box_bytes = step / steps_per_box * rows * swizzle_bytes;
        offset.* = (box_bytes + step % steps_per_box * k_step_bytes) >> 4;
    }
    return offsets;
}

pub const Element = enum(u3) { e4m3 = 0, e5m2 = 1, e2m3 = 3, e3m2 = 4, e2m1 = 5 };

pub const Shape = struct {
    cta_group: CtaGroup,
    m: u32,
    n: u32,
    a: Element = .e4m3,
    b: Element = .e4m3,
    a_major: Major = .k,
    b_major: Major = .k,

    fn check(comptime shape: Shape, comptime block_scaled: bool) void {
        const m_valid = switch (shape.cta_group) {
            .one => shape.m == 128 or (shape.m == 64 and !block_scaled),
            .two => shape.m == 128 or shape.m == 256,
        };
        const narrow = shape.cta_group == .one and (shape.m == 64 or block_scaled);
        const n_step: u32 = if (narrow) 8 else 16;
        if (!m_valid or shape.n < n_step or shape.n > 256 or shape.n % n_step != 0) {
            @compileError(std.fmt.comptimePrint("no tcgen05.mma shape M={d} N={d}", .{
                shape.m,
                shape.n,
            }));
        }
    }
};

/// The fields follow Table 51 of PTX ISA 9.7.18.4.2.
const DenseDescriptor = packed struct(u32) {
    sparsity_selector: u2 = 0,
    sparse: bool = false,
    saturate: bool = false,
    d_type: enum(u2) { f16 = 0, f32 = 1 } = .f32,
    reserved_6: u1 = 0,
    a_type: Element,
    b_type: Element,
    negate_a: bool = false,
    negate_b: bool = false,
    a_major: Major,
    b_major: Major,
    n: u6,
    reserved_23: u1 = 0,
    m: u5,
    k_64: bool = false,
    maximum_shift: u2 = 0,
};

/// The fields follow Table 52 of PTX ISA 9.7.18.4.2.
const BlockScaledDescriptor = packed struct(u32) {
    reserved_0: u2 = 0,
    sparse: bool = false,
    reserved_3: u1 = 0,
    b_scale_id: u2,
    reserved_6: u1 = 0,
    a_type: Element,
    b_type: Element,
    negate_a: bool = false,
    negate_b: bool = false,
    a_major: Major,
    b_major: Major,
    n: u6,
    ue8m0_scales: bool = true,
    reserved_24: u2 = 0,
    a_scale_128_lanes: bool = false,
    m: u2,
    a_scale_id: u2,
    k_64: bool = false,
};

pub fn f8f6f4Descriptor(comptime shape: Shape) u32 {
    comptime shape.check(false);
    const descriptor: DenseDescriptor = .{
        .a_type = shape.a,
        .b_type = shape.b,
        .a_major = shape.a_major,
        .b_major = shape.b_major,
        .n = shape.n >> 3,
        .m = shape.m >> 4,
    };
    return @bitCast(descriptor);
}

/// The scale ID selects the byte of each 32-bit tensor memory word that holds the scale factors
/// of the step (PTX ISA 9.7.18.10.7.2).
pub fn mxf8f6f4Descriptor(
    comptime shape: Shape,
    comptime a_scale_id: u2,
    comptime b_scale_id: u2,
) u32 {
    comptime shape.check(true);
    const descriptor: BlockScaledDescriptor = .{
        .b_scale_id = b_scale_id,
        .a_type = shape.a,
        .b_type = shape.b,
        .a_major = shape.a_major,
        .b_major = shape.b_major,
        .n = shape.n >> 3,
        .m = shape.m >> 7,
        .a_scale_id = a_scale_id,
    };
    return @bitCast(descriptor);
}

pub fn mxf8f6f4StepDescriptors(comptime shape: Shape, comptime steps: u32) [steps]u32 {
    var descriptors: [steps]u32 = undefined;
    for (&descriptors, 0..) |*descriptor, step| {
        descriptor.* = mxf8f6f4Descriptor(shape, step, step);
    }
    return descriptors;
}

pub fn address(lane: u32, column: u32) u32 {
    return lane << 16 | column;
}

fn checkColumns(comptime columns: u32) void {
    if (columns < 32 or columns > capacity or !std.math.isPowerOfTwo(columns)) {
        @compileError("tcgen05.alloc takes a power of two from 32 to 512 columns");
    }
}

pub fn allocate(
    comptime cta_group: CtaGroup,
    slot: *addrspace(.shared) u32,
    comptime columns: u32,
) void {
    comptime checkColumns(columns);
    asm volatile (std.fmt.comptimePrint(
            "tcgen05.alloc{s}.sync.aligned.shared::cta.b32 [%[slot]], {d};",
            .{ cta_group.text(), columns },
        )
        :
        : [slot] "r" (slot),
        : .{ .memory = true });
}

pub fn relinquishAllocPermit(comptime cta_group: CtaGroup) void {
    const text = comptime "tcgen05.relinquish_alloc_permit" ++ cta_group.text() ++ ".sync.aligned;";
    asm volatile (text ::: .{ .memory = true });
}

pub fn deallocate(comptime cta_group: CtaGroup, base: u32, comptime columns: u32) void {
    comptime checkColumns(columns);
    asm volatile (std.fmt.comptimePrint(
            "tcgen05.dealloc{s}.sync.aligned.b32 %[base], {d};",
            .{ cta_group.text(), columns },
        )
        :
        : [base] "r" (base),
        : .{ .memory = true });
}

const enable_input =
    "{\n.reg .pred accumulate;\nsetp.ne.b32 accumulate, %[accumulate], 0;\n";

pub fn mma(
    comptime cta_group: CtaGroup,
    d: u32,
    a: u64,
    b: u64,
    instruction: u32,
    accumulate: bool,
) void {
    asm volatile (enable_input ++ "tcgen05.mma" ++ cta_group.text() ++
            ".kind::f8f6f4 [%[d]], %[a], %[b], %[instruction], accumulate;\n}"
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

/// The `.block32` qualifier works on `sm_100f`, while its alias `.scale_vec::1X` needs `sm_100a`
/// (PTX ISA 9.7.18.10.10, target notes).
pub fn mmaBlockScaled(
    comptime cta_group: CtaGroup,
    d: u32,
    a: u64,
    b: u64,
    instruction: u32,
    scale_a: u32,
    scale_b: u32,
    accumulate: bool,
) void {
    asm volatile (enable_input ++ "tcgen05.mma" ++ cta_group.text() ++
            ".kind::mxf8f6f4.block_scale.block32 [%[d]], %[a], %[b], %[instruction], " ++
            "[%[scale_a]], [%[scale_b]], accumulate;\n}"
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
          [instruction] "r" (instruction),
          [scale_a] "r" (scale_a),
          [scale_b] "r" (scale_b),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

fn stepsText(
    comptime cta_group: CtaGroup,
    comptime block_scaled: bool,
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
    comptime instructions: []const u32,
) []const u8 {
    if (a_offsets.len != b_offsets.len or a_offsets.len != instructions.len) {
        @compileError("each MMA step needs an A offset, a B offset and a descriptor");
    }
    const kind = if (block_scaled) ".kind::mxf8f6f4.block_scale.block32" else ".kind::f8f6f4";
    const scales = if (block_scaled) "[%[scale_a]], [%[scale_b]], " else "";
    var text: []const u8 = enable_input ++
        ".reg .b32 a_low, a_high, b_low, b_high, a_step, b_step, instruction;\n" ++
        ".reg .b64 a, b;\nmov.b64 {a_low, a_high}, %[a];\nmov.b64 {b_low, b_high}, %[b];\n";
    for (a_offsets, b_offsets, instructions, 0..) |a_offset, b_offset, instruction, step| {
        text = text ++ std.fmt.comptimePrint(
            "add.u32 a_step, a_low, {d};\nmov.b64 a, {{a_step, a_high}};\n" ++
                "add.u32 b_step, b_low, {d};\nmov.b64 b, {{b_step, b_high}};\n" ++
                "mov.b32 instruction, {d};\n" ++
                "tcgen05.mma{s}{s} [%[d]], a, b, instruction, {s}{s};\n",
            .{
                a_offset,
                b_offset,
                instruction,
                cta_group.text(),
                kind,
                scales,
                if (step == 0) "accumulate" else "1",
            },
        );
    }
    return text ++ "}";
}

/// Issues one MMA per step from the calling thread. Step `i` adds `a_offsets[i]` and
/// `b_offsets[i]` (in 16-byte units) to the start addresses of the descriptors and uses
/// `instructions[i]`. Only the first step reads `accumulate`; the later steps accumulate.
pub fn mmaSteps(
    comptime cta_group: CtaGroup,
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
    comptime instructions: []const u32,
    d: u32,
    a: u64,
    b: u64,
    accumulate: bool,
) void {
    asm volatile (stepsText(cta_group, false, a_offsets, b_offsets, instructions)
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

pub fn mmaBlockScaledSteps(
    comptime cta_group: CtaGroup,
    comptime a_offsets: []const u32,
    comptime b_offsets: []const u32,
    comptime instructions: []const u32,
    d: u32,
    a: u64,
    b: u64,
    scale_a: u32,
    scale_b: u32,
    accumulate: bool,
) void {
    asm volatile (stepsText(cta_group, true, a_offsets, b_offsets, instructions)
        :
        : [d] "r" (d),
          [a] "l" (a),
          [b] "l" (b),
          [scale_a] "r" (scale_a),
          [scale_b] "r" (scale_b),
          [accumulate] "r" (@as(u32, @intFromBool(accumulate))),
        : .{ .memory = true });
}

/// Copies 32 rows of 128 bits from shared memory to each of the four 32-lane quarters of tensor
/// memory (PTX ISA 9.7.18.9.2).
pub fn copyWarpx4(comptime cta_group: CtaGroup, tensor_address: u32, source: u64) void {
    asm volatile ("tcgen05.cp" ++ cta_group.text() ++ ".32x128b.warpx4 [%[t]], %[s];"
        :
        : [t] "r" (tensor_address),
          [s] "l" (source),
        : .{ .memory = true });
}

pub fn commit(comptime cta_group: CtaGroup, barrier: *addrspace(.shared) nvptx.Mbarrier) void {
    asm volatile ("tcgen05.commit" ++ cta_group.text() ++
            ".mbarrier::arrive::one.shared::cluster.b64 [%[barrier]];"
        :
        : [barrier] "r" (barrier),
        : .{ .memory = true });
}

/// Bit `i` of `cta_mask` selects the barrier at the same shared memory offset in the CTA whose
/// `%cluster_ctarank` is `i` (PTX ISA 9.7.18.12.1).
pub fn commitMulticast(
    comptime cta_group: CtaGroup,
    barrier: *addrspace(.shared) nvptx.Mbarrier,
    cta_mask: u16,
) void {
    asm volatile ("{\n.reg .b16 mask;\ncvt.u16.u32 mask, %[mask];\ntcgen05.commit" ++
            cta_group.text() ++ ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 " ++
            "[%[barrier]], mask;\n}"
        :
        : [barrier] "r" (barrier),
          [mask] "r" (@as(u32, cta_mask)),
        : .{ .memory = true });
}

pub const LoadShape = enum {
    @"32x32b",
    @"16x256b",

    fn registers(comptime shape: LoadShape, comptime repeat: u32) u32 {
        const per_repeat: u32 = switch (shape) {
            .@"32x32b" => 1,
            .@"16x256b" => 4,
        };
        if (!std.math.isPowerOfTwo(repeat) or repeat * per_repeat > 128) {
            @compileError("tcgen05.ld takes .x1 to .x128 and at most 128 registers");
        }
        return repeat * per_repeat;
    }
};

pub const Pack = enum { none, pack_16b };

/// The backend prints a vector of one element as a bare register, and ptxas expects braces.
fn registerList(comptime operand: []const u8, comptime count: u32) []const u8 {
    return if (count == 1) "{%[" ++ operand ++ "]}" else "%[" ++ operand ++ "]";
}

/// Loads `.shape.x{repeat}` into registers of type `T` (`f32` or `u32`). With `.pack_16b`, each
/// register holds the 16-bit elements of two adjacent columns (PTX ISA 9.7.18.8.2).
pub fn load(
    comptime T: type,
    comptime shape: LoadShape,
    comptime repeat: u32,
    comptime pack: Pack,
    tensor_address: u32,
) @Vector(shape.registers(repeat), T) {
    const text = std.fmt.comptimePrint("tcgen05.ld.sync.aligned.{t}.x{d}{s}.b32 {s}, [%[a]];", .{
        shape,
        repeat,
        if (pack == .pack_16b) ".pack::16b" else "",
        comptime registerList("d", shape.registers(repeat)),
    });
    const Values = @Vector(shape.registers(repeat), T);
    return switch (T) {
        f32 => asm volatile (text
            : [d] "=f" (-> Values),
            : [a] "r" (tensor_address),
            : .{ .memory = true }),
        u32 => asm volatile (text
            : [d] "=r" (-> Values),
            : [a] "r" (tensor_address),
            : .{ .memory = true }),
        else => @compileError("tcgen05.ld fills f32 or u32 registers"),
    };
}

pub fn store(tensor_address: u32, values: anytype) void {
    const count = @typeInfo(@TypeOf(values)).vector.len;
    const text = std.fmt.comptimePrint(
        "tcgen05.st.sync.aligned.32x32b.x{d}.b32 [%[a]], {s};",
        .{ count, comptime registerList("v", count) },
    );
    switch (@typeInfo(@TypeOf(values)).vector.child) {
        f32 => asm volatile (text
            :
            : [a] "r" (tensor_address),
              [v] "f" (values),
            : .{ .memory = true }),
        u32 => asm volatile (text
            :
            : [a] "r" (tensor_address),
              [v] "r" (values),
            : .{ .memory = true }),
        else => @compileError("tcgen05.st takes f32 or u32 vectors"),
    }
}

pub const fence = nvptx.tcgen05.fence;
pub const waitSyncAligned = nvptx.tcgen05.waitSyncAligned;

test "instruction descriptors match the PTX ISA tables" {
    const expect = std.testing.expectEqual;
    const pair: Shape = .{ .cta_group = .two, .m = 256, .n = 256 };
    try expect(0x1040_0010, comptime f8f6f4Descriptor(pair));
    try expect(0x0820_0010, comptime f8f6f4Descriptor(.{ .cta_group = .one, .m = 128, .n = 128 }));
    try expect(0x0811_0490, comptime f8f6f4Descriptor(.{
        .cta_group = .one,
        .m = 128,
        .n = 64,
        .a = .e5m2,
        .b = .e5m2,
        .b_major = .mn,
    }));
    try expect(0x10c0_0000, comptime mxf8f6f4Descriptor(pair, 0, 0));
    try expect(0x70c0_0030, comptime mxf8f6f4Descriptor(pair, 3, 3));
    const narrow: Shape = .{ .cta_group = .one, .m = 128, .n = 8 };
    try expect(0x0882_0000, comptime mxf8f6f4Descriptor(narrow, 0, 0));
    const steps = comptime mxf8f6f4StepDescriptors(pair, 4);
    try expect(0x30c0_0010, steps[1]);
}

test "shared memory descriptors match the PTX ISA table" {
    const expect = std.testing.expectEqual;
    try expect(0x4000_4040_0001_0040, kMajorDescriptor(0x400, 128));
    try expect(0x4000_4040_0001_0042, advance(kMajorDescriptor(0x400, 128), k_step_bytes));
    try expect([4]u32{ 0, 2, 4, 6 }, comptime kMajorOffsets(4, 128, 128));
    try expect(0x0000_4008_0008_0040, sharedDescriptor(0x400, .scale_atom));
    try expect(0x0000_4000_0000_0040, sharedDescriptor(0x400, .repeated_scale_rows));
    try expect([2]u32{ 0, 256 }, comptime kMajorOffsets(2, 128, 32));
}
