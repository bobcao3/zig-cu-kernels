//! Tensor copies between global memory and shared memory (PTX ISA 9.7.10.28.5.3). The coordinates
//! of a copy start with the innermost dimension.

const std = @import("std");
const nvptx = std.nvptx;

const Mbarrier = nvptx.Mbarrier;

pub const commitGroup = nvptx.cp_async_bulk.commitGroup;
pub const waitGroup = nvptx.cp_async_bulk.waitGroup;

/// A kernel receives a tensor map by value.
pub const TensorMap = extern struct {
    bytes: [128]u8 align(64),
};

pub fn prefetch(map: *const TensorMap) void {
    nvptx.cache.prefetchTensormap(map);
}

/// An opaque L2 eviction policy from `createpolicy` (PTX ISA 9.7.10.21).
pub const CachePolicy = enum(u64) { _ };

pub const Eviction = enum { evict_first, evict_last, evict_normal, evict_unchanged };

pub inline fn createPolicy(comptime eviction: Eviction) CachePolicy {
    nvptx.require("createpolicy", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    const policy = asm ("createpolicy.fractional.L2::" ++ @tagName(eviction) ++ ".b64 %[policy];"
        : [policy] "=l" (-> u64),
    );
    return @fromBackingInt(@intCast(policy));
}

pub inline fn sharedAddress(address: anytype) u32 {
    if (@TypeOf(address) == u32) return address;
    return @intCast(@intFromPtr(address));
}

fn copyText(
    comptime dimensions: usize,
    comptime destination: []const u8,
    comptime qualifiers: []const u8,
    comptime tail: []const u8,
) []const u8 {
    return std.fmt.comptimePrint(
        "cp.async.bulk.tensor.{d}d.{s}.global.tile.mbarrier::complete_tx::bytes{s} " ++
            "[%[destination]], [%[map], {s}], [%[barrier]]{s};",
        .{ dimensions, destination, qualifiers, coordinateList(dimensions), tail },
    );
}

/// The backend writes a vector of one element without the braces of a vector operand.
fn coordinateList(comptime dimensions: usize) []const u8 {
    if (dimensions < 1 or dimensions > 5) @compileError("a tensor copy has 1 to 5 dimensions");
    return if (dimensions == 1) "{%[coordinates]}" else "%[coordinates]";
}

/// Every copy takes as `policy` either `null` or a `CachePolicy`, which adds `.L2::cache_hint`.
fn hinted(comptime Policy: type) bool {
    if (Policy == @TypeOf(null)) return false;
    if (Policy != CachePolicy) {
        @compileError("expected null or a CachePolicy, found " ++ @typeName(Policy));
    }
    return true;
}

const cta_pair = ".cta_group::2";
const hint = ".L2::cache_hint";

fn Coordinates(comptime dimensions: usize) type {
    return @Vector(dimensions, i32);
}

pub inline fn load(
    comptime dimensions: usize,
    destination: anytype,
    map: *const TensorMap,
    coordinates: [dimensions]i32,
    barrier: *addrspace(.shared) Mbarrier,
    policy: anytype,
) void {
    nvptx.require("cp.async.bulk.tensor.shared::cta", .{
        .ptx = 86,
        .sm = .{ .base = 90 },
        .threads = .thread,
    });
    const vector: Coordinates(dimensions) = coordinates;
    if (comptime hinted(@TypeOf(policy))) {
        asm volatile (copyText(dimensions, "shared::cta", hint, ", %[policy]")
            :
            : [destination] "r" (sharedAddress(destination)),
              [map] "l" (map),
              [coordinates] "r" (vector),
              [barrier] "r" (barrier),
              [policy] "l" (@backingInt(@as(CachePolicy, policy))),
            : .{ .memory = true });
    } else {
        asm volatile (copyText(dimensions, "shared::cta", "", "")
            :
            : [destination] "r" (sharedAddress(destination)),
              [map] "l" (map),
              [coordinates] "r" (vector),
              [barrier] "r" (barrier),
            : .{ .memory = true });
    }
}

/// Copies to the same offset in each CTA whose rank has its bit set in `cta_mask`. Each of these
/// CTAs counts the bytes on its own barrier at the offset of `barrier`.
pub inline fn loadMulticast(
    comptime dimensions: usize,
    destination: anytype,
    map: *const TensorMap,
    coordinates: [dimensions]i32,
    barrier: *addrspace(.shared) Mbarrier,
    cta_mask: u16,
    policy: anytype,
) void {
    const address = sharedAddress(barrier);
    loadCluster(dimensions, "", true, destination, map, coordinates, address, cta_mask, policy);
}

/// With `.cta_group::2`, `barrier` is a `.shared::cluster` address in the CTA of `destination` or
/// in the other CTA of its pair.
pub inline fn loadPair(
    comptime dimensions: usize,
    destination: anytype,
    map: *const TensorMap,
    coordinates: [dimensions]i32,
    barrier: u32,
    policy: anytype,
) void {
    loadCluster(dimensions, cta_pair, false, destination, map, coordinates, barrier, 0, policy);
}

/// For each destination CTA, the copy signals the barrier at the offset of `barrier` in the CTA of
/// the pair whose rank has the parity of the rank of the CTA that holds `barrier`.
pub inline fn loadPairMulticast(
    comptime dimensions: usize,
    destination: anytype,
    map: *const TensorMap,
    coordinates: [dimensions]i32,
    barrier: u32,
    cta_mask: u16,
    policy: anytype,
) void {
    loadCluster(
        dimensions,
        cta_pair,
        true,
        destination,
        map,
        coordinates,
        barrier,
        cta_mask,
        policy,
    );
}

inline fn loadCluster(
    comptime dimensions: usize,
    comptime cta_group: []const u8,
    comptime multicast: bool,
    destination: anytype,
    map: *const TensorMap,
    coordinates: [dimensions]i32,
    barrier: u32,
    cta_mask: u16,
    policy: anytype,
) void {
    const pair = cta_group.len != 0;
    if (pair) {
        nvptx.require("cp.async.bulk.tensor.cta_group::2", .{
            .ptx = 88,
            .sm = .{ .family = &.{ 100, 103, 110 } },
            .threads = .thread,
        });
    } else {
        nvptx.require("cp.async.bulk.tensor.multicast", .{
            .ptx = 80,
            .sm = .{ .base = 90 },
            .threads = .thread,
        });
    }
    const vector: Coordinates(dimensions) = coordinates;
    const space = "shared::cluster";
    const multicast_qualifiers = ".multicast::cluster" ++ cta_group;
    if (multicast) {
        if (comptime hinted(@TypeOf(policy))) {
            const text = comptime copyText(
                dimensions,
                space,
                multicast_qualifiers ++ hint,
                ", %[cta_mask], %[policy]",
            );
            asm volatile (text
                :
                : [destination] "r" (sharedAddress(destination)),
                  [map] "l" (map),
                  [coordinates] "r" (vector),
                  [barrier] "r" (barrier),
                  [cta_mask] "h" (cta_mask),
                  [policy] "l" (@backingInt(@as(CachePolicy, policy))),
                : .{ .memory = true });
        } else {
            const tail = ", %[cta_mask]";
            const text = comptime copyText(dimensions, space, multicast_qualifiers, tail);
            asm volatile (text
                :
                : [destination] "r" (sharedAddress(destination)),
                  [map] "l" (map),
                  [coordinates] "r" (vector),
                  [barrier] "r" (barrier),
                  [cta_mask] "h" (cta_mask),
                : .{ .memory = true });
        }
    } else if (comptime hinted(@TypeOf(policy))) {
        const text = comptime copyText(dimensions, space, cta_group ++ hint, ", %[policy]");
        asm volatile (text
            :
            : [destination] "r" (sharedAddress(destination)),
              [map] "l" (map),
              [coordinates] "r" (vector),
              [barrier] "r" (barrier),
              [policy] "l" (@backingInt(@as(CachePolicy, policy))),
            : .{ .memory = true });
    } else {
        asm volatile (copyText(dimensions, space, cta_group, "")
            :
            : [destination] "r" (sharedAddress(destination)),
              [map] "l" (map),
              [coordinates] "r" (vector),
              [barrier] "r" (barrier),
            : .{ .memory = true });
    }
}

pub inline fn store(
    comptime dimensions: usize,
    map: *const TensorMap,
    coordinates: [dimensions]i32,
    source: anytype,
    policy: anytype,
) void {
    nvptx.require("cp.async.bulk.tensor.global", .{
        .ptx = 80,
        .sm = .{ .base = 90 },
        .threads = .thread,
    });
    const vector: Coordinates(dimensions) = coordinates;
    const text = std.fmt.comptimePrint(
        "cp.async.bulk.tensor.{d}d.global.shared::cta.tile.bulk_group",
        .{dimensions},
    );
    if (comptime hinted(@TypeOf(policy))) {
        asm volatile (text ++ hint ++ " [%[map], " ++ coordinateList(dimensions) ++
                "], [%[source]], %[policy];"
            :
            : [map] "l" (map),
              [coordinates] "r" (vector),
              [source] "r" (sharedAddress(source)),
              [policy] "l" (@backingInt(@as(CachePolicy, policy))),
            : .{ .memory = true });
    } else {
        asm volatile (text ++ " [%[map], " ++ coordinateList(dimensions) ++ "], [%[source]];"
            :
            : [map] "l" (map),
              [coordinates] "r" (vector),
              [source] "r" (sharedAddress(source)),
            : .{ .memory = true });
    }
}
