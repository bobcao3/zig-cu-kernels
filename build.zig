const std = @import("std");
const configurations = @import("zig/configurations.zig");

const Module = configurations.Module;
const Launch = configurations.Launch;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub fn build(b: *std.Build) void {
    @setEvalBranchQuota(400_000);
    const optimize_option =
        b.option(std.lang.Optimize, "kernel-optimize", "Kernel optimization mode");
    const optimize = optimize_option orelse .fast;
    const ptxas_option = b.option([]const u8, "ptxas", "Path to ptxas");
    const sources = b.option([]const []const u8, "source", "Only build the modules of these sources");
    const module_names = b.option([]const []const u8, "module", "Only build these modules");
    const targets = b.option([]const []const u8, "sm", "Only assemble cubins for these targets");
    const ptxas = ptxas_option orelse b.findProgram(.{ .names = &.{"ptxas"} }) orelse
        b.findProgram(.{ .names = &.{"/usr/local/cuda/bin/ptxas"} }) orelse "ptxas";
    const fatbinary = b.pathJoin(&.{ std.fs.path.dirname(ptxas) orelse "", "fatbinary" });
    for (targets orelse &.{}) |target| {
        if (isTarget(target)) continue;
        std.process.fatal("-Dsm={s} names no target of any module", .{target});
    }

    var manifest: std.Io.Writer.Allocating = .init(b.allocator);
    var declared: std.Io.Writer.Allocating = .init(b.allocator);
    if (sources != null or module_names != null or targets != null) {
        declared.writer.writeAll("partial\n") catch @panic("OOM");
    }
    manifest.writer.writeAll("# kernel module arch") catch @panic("OOM");
    inline for (@typeInfo(Launch).@"struct".field_names) |field_name| {
        manifest.writer.print(" {s}", .{field_name}) catch @panic("OOM");
    }
    manifest.writer.print("\n# sources sha256 {s}\n# options", .{&sourceDigest(b)}) catch
        @panic("OOM");
    if (optimize_option) |mode| {
        manifest.writer.print(" -Dkernel-optimize={t}", .{mode}) catch @panic("OOM");
    }
    const filters = .{ .{ "source", sources }, .{ "module", module_names }, .{ "sm", targets } };
    inline for (filters) |filter| {
        for (filter[1] orelse &.{}) |value| {
            manifest.writer.print(" -D{s}={s}", .{ filter[0], value }) catch @panic("OOM");
        }
    }
    manifest.writer.writeByte('\n') catch @panic("OOM");

    const remove_manifest = b.addSystemCommand(&.{ "rm", "-f" });
    remove_manifest.setName("remove kernels.manifest");
    remove_manifest.addDirectoryArg2(
        .{ .relative = .{ .base = .install_prefix } },
        .{ .suffix = "/kernels.manifest" },
    );
    remove_manifest.has_side_effects = true;
    var module_installs: std.ArrayList(*std.Build.Step) = .empty;
    inline for (configurations.modules) |module| {
        inline for (module.kernels) |kernel| {
            declared.writer.print("{s}\n", .{kernel.name}) catch @panic("OOM");
        }
        if (selected(sources, module.source) and selected(module_names, module.name)) {
            const module_targets = selectedTargets(b, module, targets);
            manifest.writer.print("# targets {s}", .{module.name}) catch @panic("OOM");
            for (module_targets) |target| {
                manifest.writer.print(" {s}", .{target}) catch @panic("OOM");
            }
            manifest.writer.writeByte('\n') catch @panic("OOM");
            inline for (module.kernels) |kernel| {
                manifest.writer.print("{s} {s} {d}", .{ kernel.name, module.name, module.arch }) catch
                    @panic("OOM");
                inline for (@typeInfo(Launch).@"struct".field_names) |field_name| {
                    manifest.writer.print(" {d}", .{@field(kernel.launch, field_name)}) catch
                        @panic("OOM");
                }
                manifest.writer.writeByte('\n') catch @panic("OOM");
            }
            addModule(b, module, optimize, ptxas, fatbinary, module_targets, &module_installs);
        }
    }
    for (module_installs.items) |module_install| module_install.dependOn(&remove_manifest.step);
    const files = b.addWriteFiles();
    inline for (.{
        .{ "kernels.manifest", &manifest },
        .{ "kernels.declared", &declared },
    }) |output| {
        const install = b.addInstallFile(files.add(output[0], output[1].written()), output[0]);
        install.step.dependOn(&remove_manifest.step);
        for (module_installs.items) |module_install| install.step.dependOn(module_install);
        b.getInstallStep().dependOn(&install.step);
    }

    const tests = b.step("test", "Run the host unit tests");
    const roots = .{ "zig/configurations.zig", "zig/tcgen05.zig", "zig/wgmma.zig", "zig/pipeline.zig" };
    inline for (roots) |root| {
        const unit_tests = b.addTest(.{ .root_module = b.createModule(.{
            .root_source_file = b.path(root),
            .target = b.graph.host,
        }) });
        tests.dependOn(&b.addRunArtifact(unit_tests).step);
    }
}

fn selected(filter: ?[]const []const u8, name: []const u8) bool {
    const names = filter orelse return true;
    for (names) |candidate| if (std.mem.eql(u8, candidate, name)) return true;
    return false;
}

fn isTarget(name: []const u8) bool {
    inline for (configurations.modules) |module| {
        for (comptime configurations.cubinTargets(module)) |target| {
            if (std.mem.eql(u8, target, name)) return true;
        }
    }
    return false;
}

fn selectedTargets(
    b: *std.Build,
    comptime module: Module,
    filter: ?[]const []const u8,
) []const []const u8 {
    if (filter == null) return comptime configurations.defaultTargets(module);
    var list: std.ArrayList([]const u8) = .empty;
    for (comptime configurations.cubinTargets(module)) |target| {
        if (selected(filter, target)) list.append(b.allocator, target) catch @panic("OOM");
    }
    return list.items;
}

fn modulePtx(
    b: *std.Build,
    comptime module: Module,
    optimize: std.lang.Optimize,
    cpu: []const u8,
) std.Build.LazyPath {
    const config = b.addOptions();
    config.addOption(u32, "variant", module.variant);
    const own = std.mem.eql(u8, cpu, configurations.ownCpu(module));
    const kernel = b.addExecutable(.{
        .name = if (own) module.name else b.fmt("{s}_sm{s}", .{ module.name, cpu["sm_".len..] }),
        .use_llvm = false,
        .root_module = b.createModule(.{
            .root_source_file = b.path(b.fmt("zig/{s}.zig", .{module.source})),
            .target = b.resolveTargetQuery(std.Target.Query.parse(.{
                .arch_os_abi = "nvptx64-cuda",
                .cpu_features = b.fmt("{s}+ptx90", .{cpu}),
            }) catch @panic("invalid target")),
            .optimize = optimize,
            .imports = &.{.{ .name = "config", .module = config.createModule() }},
        }),
    });
    return kernel.getEmittedBin();
}

/// Returns the SHA-256 digest of `build.zig` and the regular files `zig/*.zig` in hexadecimal.
/// The Python function `zig_cu_kernels.runtime.source_digest` computes the same digest.
fn sourceDigest(b: *std.Build) [Sha256.digest_length * 2]u8 {
    const io = b.graph.io;
    var root = b.root.openDir(io, ".", .{}) catch |err|
        std.process.fatal("unable to open the build root: {t}", .{err});
    defer root.close(io);
    var folder = root.openDir(io, "zig", .{ .iterate = true }) catch |err|
        std.process.fatal("unable to open zig/: {t}", .{err});
    defer folder.close(io);
    b.dependOnDirectoryContents(b.path("zig"));

    var paths: std.ArrayList([]const u8) = .empty;
    paths.append(b.allocator, "build.zig") catch @panic("OOM");
    var entries = folder.iterate();
    const listing_error = "unable to list zig/: {t}";
    while (entries.next(io) catch |err| std.process.fatal(listing_error, .{err})) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".zig") and isRegularFile(io, folder, entry)) {
            paths.append(b.allocator, b.fmt("zig/{s}", .{entry.name})) catch @panic("OOM");
        }
    }
    std.mem.sortUnstable([]const u8, paths.items, {}, lessThan);

    var digest: Sha256 = .init(.{});
    for (paths.items) |path| {
        b.dependOnFileContents(b.path(path));
        const contents = root.readFileAlloc(io, path, b.allocator, .unlimited) catch |err|
            std.process.fatal("unable to read {s}: {t}", .{ path, err });
        var size: [8]u8 = undefined;
        std.mem.writeInt(u64, &size, contents.len, .little);
        digest.update(path);
        digest.update(&.{0});
        digest.update(&size);
        digest.update(contents);
    }
    return std.fmt.bytesToHex(digest.finalResult(), .lower);
}

/// Some file systems list every entry with the type `DT_UNKNOWN`. For such an entry, the function
/// reads the type with `lstat`, as Python's `os.DirEntry.is_file(follow_symlinks=False)` does.
fn isRegularFile(io: std.Io, folder: std.Io.Dir, entry: std.Io.Dir.Entry) bool {
    if (entry.kind != .unknown) return entry.kind == .file;
    const stat = folder.statFile(io, entry.name, .{ .follow_symlinks = false }) catch |err|
        std.process.fatal("unable to read the type of zig/{s}: {t}", .{ entry.name, err });
    return stat.kind == .file;
}

fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

fn addModule(
    b: *std.Build,
    comptime module: Module,
    optimize: std.lang.Optimize,
    ptxas: []const u8,
    fatbinary: []const u8,
    targets: []const []const u8,
    installs: *std.ArrayList(*std.Build.Step),
) void {
    if (targets.len == 0) {
        const ptx = modulePtx(b, module, optimize, configurations.ownCpu(module));
        const ptx_install = b.addInstallFile(ptx, b.fmt("ptx/{s}.ptx", .{module.name}));
        installs.append(b.allocator, &ptx_install.step) catch @panic("OOM");
        return;
    }
    const pack = b.addSystemCommand(&.{ fatbinary, "-64" });
    const cubin = pack.addPrefixedOutputFileArg("--create=", b.fmt("{s}.cubin", .{module.name}));
    var cpus: [8][]const u8 = undefined;
    var ptxs: [8]std.Build.LazyPath = undefined;
    var compiled: usize = 0;
    for (targets) |target| {
        const cpu = configurations.ptxCpu(module, target);
        const ptx = for (cpus[0..compiled], ptxs[0..compiled]) |known, known_ptx| {
            if (std.mem.eql(u8, known, cpu)) break known_ptx;
        } else ptx: {
            const ptx = modulePtx(b, module, optimize, cpu);
            const name = if (std.mem.eql(u8, cpu, configurations.ownCpu(module)))
                b.fmt("ptx/{s}.ptx", .{module.name})
            else
                b.fmt("ptx/{s}.sm{s}.ptx", .{ module.name, cpu["sm_".len..] });
            const ptx_install = b.addInstallFile(ptx, name);
            installs.append(b.allocator, &ptx_install.step) catch @panic("OOM");
            cpus[compiled] = cpu;
            ptxs[compiled] = ptx;
            compiled += 1;
            break :ptx ptx;
        };
        const assemble = b.addSystemCommand(&.{ ptxas, b.fmt("-arch={s}", .{target}) });
        assemble.addFileArg(ptx);
        assemble.addArg("-o");
        const image = assemble.addOutputFileArg(b.fmt("{s}.{s}.cubin", .{ module.name, target }));
        pack.addPrefixedFileArg(b.fmt("--image3=kind=elf,sm={s},file=", .{
            target["sm_".len..],
        }), image);
    }
    const cubin_install = b.addInstallFile(cubin, b.fmt("cubin/{s}.cubin", .{module.name}));
    installs.append(b.allocator, &cubin_install.step) catch @panic("OOM");
}
