//! Bulk copies between global and shared memory (PTX ISA 9.7.9.25.4.1). Each copy moves a multiple
//! of 16 bytes between addresses that are multiples of 16.

const std = @import("std");
const nvptx = std.nvptx;

pub const Mbarrier = nvptx.Mbarrier;
pub const commitGroup = nvptx.cp_async_bulk.commitGroup;
pub const waitGroup = nvptx.cp_async_bulk.waitGroup;

/// Copies `bytes` bytes from `source` to `destination`. The copy counts its bytes on `barrier`.
pub inline fn load(
    destination: [*]addrspace(.shared) u8,
    source: [*]addrspace(.global) const u8,
    bytes: u32,
    barrier: *addrspace(.shared) Mbarrier,
) void {
    asm volatile ("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes " ++
            "[%[destination]], [%[source]], %[bytes], [%[barrier]];"
        :
        : [destination] "r" (destination),
          [source] "l" (source),
          [bytes] "r" (bytes),
          [barrier] "r" (barrier),
        : .{ .memory = true });
}

/// Copies `bytes` bytes from `source` to `destination` in the current bulk async-group.
pub inline fn store(
    destination: [*]addrspace(.global) u8,
    source: [*]addrspace(.shared) const u8,
    bytes: u32,
) void {
    asm volatile ("cp.async.bulk.global.shared::cta.bulk_group [%[destination]], [%[source]], %[bytes];"
        :
        : [destination] "l" (destination),
          [source] "r" (source),
          [bytes] "r" (bytes),
        : .{ .memory = true });
}

/// Initializes each barrier of `barriers` for one arrival and makes the initialization visible to
/// the bulk copies (PTX ISA 9.7.13.15.11).
pub inline fn initBarriers(barriers: []addrspace(.shared) Mbarrier) void {
    for (barriers) |*barrier| barrier.init(1);
    nvptx.fenceMbarrierInit();
    nvptx.fenceProxyAsync(.shared);
}

pub inline fn wait(barrier: *addrspace(.shared) Mbarrier, phase: u32) void {
    while (!barrier.tryWaitParity(@truncate(phase))) {}
}
