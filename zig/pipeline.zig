//! A ring of shared memory buffers with a full and an empty barrier for each buffer.

const std = @import("std");
const nvptx = std.nvptx;
const cluster = @import("cluster.zig");

const Mbarrier = nvptx.Mbarrier;

pub const Scope = nvptx.mbarrier.BarrierScope;

pub fn Position(comptime stages: u32) type {
    comptime std.debug.assert(stages >= 1);
    return struct {
        index: u32 = 0,
        phase: u1 = 0,

        pub fn next(position: @This()) @This() {
            if (position.index + 1 == stages) return .{ .index = 0, .phase = ~position.phase };
            return .{ .index = position.index + 1, .phase = position.phase };
        }

        pub fn advance(position: @This(), steps: u32) @This() {
            const total = position.index + steps;
            const turns: u1 = @truncate(total / stages);
            return .{ .index = total % stages, .phase = position.phase ^ turns };
        }
    };
}

pub fn Pipeline(comptime stages: u32, comptime scope: Scope) type {
    return extern struct {
        full: [stages]Mbarrier,
        empty: [stages]Mbarrier,

        const Ring = @This();
        pub const Slot = Position(stages);

        /// One thread initializes the ring. Then the kernel synchronizes the CTA or the cluster.
        pub fn init(ring: *addrspace(.shared) Ring, producers: u32, consumers: u32) void {
            for (&ring.full) |*barrier| barrier.init(producers);
            for (&ring.empty) |*barrier| barrier.init(consumers);
            nvptx.fenceMbarrierInit();
            nvptx.fenceProxyAsync(.shared);
        }

        pub fn fullBarrier(
            ring: *addrspace(.shared) Ring,
            slot: Slot,
        ) *addrspace(.shared) Mbarrier {
            return &ring.full[slot.index];
        }

        pub fn emptyBarrier(
            ring: *addrspace(.shared) Ring,
            slot: Slot,
        ) *addrspace(.shared) Mbarrier {
            return &ring.empty[slot.index];
        }

        /// In the first pass, the wait names the phase before the current phase, and a wait for
        /// that phase returns true (PTX ISA 9.7.15.16.19).
        pub fn producerAcquire(ring: *addrspace(.shared) Ring, slot: Slot) void {
            waitPhase(ring.emptyBarrier(slot), ~slot.phase);
        }

        pub fn producerExpect(ring: *addrspace(.shared) Ring, slot: Slot, bytes: u32) void {
            _ = ring.fullBarrier(slot).arriveExpectTx(bytes);
        }

        pub fn producerExpectRemote(
            ring: *addrspace(.shared) Ring,
            slot: Slot,
            bytes: u32,
            target: u32,
        ) void {
            const address = cluster.map(ring.fullBarrier(slot), target);
            nvptx.mbarrier.arriveExpectTxCluster(address, bytes, release);
        }

        pub fn consumerWait(ring: *addrspace(.shared) Ring, slot: Slot) void {
            waitPhase(ring.fullBarrier(slot), slot.phase);
        }

        pub fn consumerRelease(ring: *addrspace(.shared) Ring, slot: Slot) void {
            _ = ring.emptyBarrier(slot).arrive();
        }

        /// Consumers release a buffer to every CTA whose copies write into the buffer.
        pub fn consumerReleaseRemote(ring: *addrspace(.shared) Ring, slot: Slot, target: u32) void {
            const address = cluster.map(ring.emptyBarrier(slot), target);
            nvptx.mbarrier.arriveCluster(address, release);
        }

        const release: nvptx.mbarrier.ArriveOrder = .{ .semantics = .release, .scope = scope };

        fn waitPhase(barrier: *addrspace(.shared) Mbarrier, phase: u1) void {
            const order: ?nvptx.mbarrier.WaitOrder = switch (scope) {
                .cta => null,
                .cluster => .{ .semantics = .acquire, .scope = .cluster },
            };
            while (!barrier.tryWaitPhase(phase, order, null)) {}
        }
    };
}

test "a position walks the ring and flips the phase at each turn" {
    inline for (1..5) |stages| {
        var position: Position(stages) = .{};
        for (0..64) |count| {
            try std.testing.expectEqual(count % stages, position.index);
            try std.testing.expectEqual(count / stages & 1, position.phase);
            const jumped = Position(stages).advance(.{}, @intCast(count));
            try std.testing.expectEqual(position, jumped);
            position = position.next();
        }
    }
}
