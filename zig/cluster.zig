//! Cluster special registers, shared memory addresses, barriers and launch control (PTX ISA 2.2.2).

const std = @import("std");
const nvptx = std.nvptx;

pub const rank = nvptx.clusterCtarank;
pub const count = nvptx.clusterNctarank;
pub const electOne = nvptx.electSync;
pub const griddepcontrol = nvptx.griddepcontrol;

const cluster_registers: nvptx.Requirement = .{
    .ptx = 78,
    .sm = .{ .base = 90 },
    .threads = .thread,
};

/// Reads a component of a special register of PTX ISA 10.12 to 10.15.
fn readRegister(comptime name: []const u8, comptime dimension: u2) u32 {
    nvptx.require(name, cluster_registers);
    if (dimension > 2) @compileError(name ++ " needs a dimension from 0 to 2");
    return asm ("mov.u32 %[d], %" ++ name ++ "." ++ "xyz"[dimension..][0..1] ++ ";"
        : [d] "=r" (-> u32),
    );
}

pub fn ctaid(comptime dimension: u2) u32 {
    return readRegister("cluster_ctaid", dimension);
}

pub fn nctaid(comptime dimension: u2) u32 {
    return readRegister("cluster_nctaid", dimension);
}

pub fn clusterid(comptime dimension: u2) u32 {
    return readRegister("clusterid", dimension);
}

pub fn nclusterid(comptime dimension: u2) u32 {
    return readRegister("nclusterid", dimension);
}

pub fn map(pointer: anytype, target: u32) u32 {
    const address: u32 = if (@TypeOf(pointer) == u32) pointer else @intCast(@intFromPtr(pointer));
    return nvptx.mapaSharedCluster(address, target);
}

/// Returns the `.shared::cluster` address of `pointer` in the even CTA of the CTA pair.
pub fn leaderAddress(pointer: anytype) u32 {
    return map(pointer, rank() & ~@as(u32, 1));
}

pub fn peerAddress(pointer: anytype) u32 {
    return map(pointer, rank() ^ 1);
}

pub const ArriveSemantics = nvptx.ClusterArriveSemantics;

pub fn arrive(comptime semantics: ?ArriveSemantics, comptime aligned: bool) void {
    nvptx.barrierClusterArrive(semantics, aligned);
}

pub fn wait(comptime aligned: bool) void {
    nvptx.barrierClusterWait(true, aligned);
}

pub fn sync() void {
    arrive(.release, true);
    wait(true);
}

/// Emits `setmaxnreg` (PTX ISA 9.7.21.5), which every warp of the warpgroup must execute.
pub fn setMaxRegisters(comptime direction: enum { inc, dec }, comptime registers: u32) void {
    nvptx.require("setmaxnreg", .{
        .ptx = 80,
        .sm = .{ .family = &.{ 90, 100, 103, 110, 120, 121 } },
        .threads = .warpgroup,
    });
    if (registers < 24 or registers > 256 or registers % 8 != 0) {
        @compileError(std.fmt.comptimePrint("setmaxnreg cannot set {d} registers", .{registers}));
    }
    asm volatile (std.fmt.comptimePrint(
            "setmaxnreg.{t}.sync.aligned.u32 {d};",
            .{ direction, registers },
        ) ::: .{ .memory = true });
}

pub const CancelResponse = @Vector(4, u32);

/// With `multicast`, every CTA of the cluster receives the response (PTX ISA 9.7.15.18).
pub fn tryCancel(
    response: *addrspace(.shared) CancelResponse,
    barrier: *addrspace(.shared) nvptx.Mbarrier,
    comptime multicast: bool,
) void {
    nvptx.require("clusterlaunchcontrol.try_cancel", .{
        .ptx = 86,
        .sm = .{ .base = 100 },
        .threads = .thread,
    });
    if (multicast) nvptx.require("clusterlaunchcontrol.try_cancel.multicast::cluster::all", .{
        .ptx = 86,
        .sm = .{ .family = &.{ 100, 103, 110, 120, 121 } },
        .threads = .thread,
    });
    asm volatile ("clusterlaunchcontrol.try_cancel.async.shared::cta" ++
            ".mbarrier::complete_tx::bytes" ++
            (if (multicast) ".multicast::cluster::all" else "") ++
            ".b128 [%[response]], [%[barrier]];"
        :
        : [response] "r" (response),
          [barrier] "r" (barrier),
        : .{ .memory = true });
}

/// Returns `maxInt(u32)` when the request canceled no cluster (PTX ISA 9.7.15.19).
pub fn canceledCtaid(
    response: *addrspace(.shared) const CancelResponse,
    comptime dimension: u2,
) u32 {
    if (dimension > 2) @compileError("a CTA id has the dimensions 0 to 2");
    return asm volatile ("{\n.reg .b128 response;\n.reg .pred canceled;\n.reg .b32 ctaid;\n" ++
            "ld.shared.b128 response, [%[address]];\n" ++
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;\n" ++
            "mov.b32 ctaid, -1;\n" ++
            "@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::" ++
            "xyz"[dimension..][0..1] ++ ".b32.b128 ctaid, response;\n" ++
            "mov.b32 %[ctaid], ctaid;\n}"
        : [ctaid] "=r" (-> u32),
        : [address] "r" (response),
        : .{ .memory = true });
}
