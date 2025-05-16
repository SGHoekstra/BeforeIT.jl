# scripts/create_complete_bundle.jl
using Pkg
using UUIDs
using TOML

# Activate your project
Pkg.activate(".")

# 1. Read Manifest.toml to get ALL dependencies
println("Step 1: Identifying all dependencies from Manifest.toml...")

project_root = abspath(joinpath(@__DIR__, ".."))
manifest_path = joinpath(project_root, "Manifest.toml")

if !isfile(manifest_path)
    error("Manifest.toml not found. Run Pkg.instantiate() first.")
end

# Parse the manifest to get all packages
manifest_data = TOML.parsefile(manifest_path)

# Handling different Manifest formats (Julia 1.6+ vs older)
all_pkgs = []
if haskey(manifest_data, "deps")  # Older format
    for (pkg_name, pkg_info) in manifest_data["deps"]
        push!(all_pkgs, pkg_name)
    end
else  # Newer format (Julia 1.6+)
    for (pkg_name, pkg_versions) in manifest_data
        if pkg_name != "julia_version"
            push!(all_pkgs, pkg_name)
        end
    end
end

println("Found $(length(all_pkgs)) total dependencies in Manifest.toml")

# 2. Develop ALL packages from Manifest
println("\nStep 2: Developing all packages to get source code...")
developed_pkgs = 0
for pkg_name in all_pkgs
    print("  Developing $pkg_name... ")
    try
        Pkg.develop(pkg_name)
        developed_pkgs += 1
        println("✓")
    catch e
        println("⚠️  Error: $e")
    end
end
println("Successfully developed $developed_pkgs out of $(length(all_pkgs)) packages")

# 3. Create bundle directory structure
println("\nStep 3: Creating bundle directory...")
bundle_dir = abspath(joinpath(@__DIR__, "..", "..", "beforeit_bundle"))
println("Creating bundle at: $bundle_dir")
mkpath(bundle_dir)

# 4. Copy all project files
println("\nStep 4: Copying BeforeIT.jl repository...")
project_files = readdir(project_root)

# Directories to exclude from copying
exclude_dirs = ["logs", ".git", ".github", ".vscode"]
exclude_files = []

# Copy all files from the project root
for item in project_files
    src_path = joinpath(project_root, item)
    dst_path = joinpath(bundle_dir, item)
    
    # Skip excluded directories and files
    if (isdir(src_path) && item in exclude_dirs) || 
       (isfile(src_path) && item in exclude_files)
        println("  Skipping $item...")
        continue
    end
    
    # Copy the item
    println("  Copying $item...")
    try
        if isdir(src_path)
            cp(src_path, dst_path, force=true)
        else
            cp(src_path, dst_path, force=true)
        end
    catch e
        println("    Warning: Error copying $item: $e")
    end
end

# 5. Copy ALL developed packages
println("\nStep 5: Copying package sources...")
dev_path = joinpath(DEPOT_PATH[1], "dev")
dev_bundle_path = joinpath(bundle_dir, "pkg_sources")
mkpath(dev_bundle_path)

copied_pkgs = 0
if isdir(dev_path)
    for pkg_dir in readdir(dev_path)
        src = joinpath(dev_path, pkg_dir)
        if isdir(src)
            print("  Copying sources for $pkg_dir... ")
            dst = joinpath(dev_bundle_path, pkg_dir)
            try
                cp(src, dst, force=true)
                copied_pkgs += 1
                println("✓")
            catch e
                println("⚠️  Error: $e")
            end
        end
    end
else
    println("  No dev packages found at $dev_path")
end
println("Copied $copied_pkgs package source directories")

# 6. Create registry backup (optional but helpful)
println("\nStep 6: Backing up registry metadata...")
registry_path = joinpath(DEPOT_PATH[1], "registries")
registry_bundle_path = joinpath(bundle_dir, "registry_backup")

if isdir(registry_path)
    println("  Copying registry data...")
    try
        cp(registry_path, registry_bundle_path, force=true)
        println("  Registry backup successful")
    catch e
        println("  Warning: Could not backup registry: $e")
    end
else
    println("  No registry found at $registry_path")
end

# 7. Create setup script (cross-platform)
println("\nStep 7: Creating setup scripts...")

setup_script = """
using Pkg

# Set up local depot path for packages
depot_path = joinpath(@__DIR__, "package_depot")
ENV["JULIA_DEPOT_PATH"] = depot_path
mkpath(depot_path)

# Set up registries directory (if backup exists)
registry_backup = joinpath(@__DIR__, "registry_backup")
if isdir(registry_backup)
    registry_path = joinpath(depot_path, "registries")
    mkpath(registry_path)
    println("Restoring registry metadata...")
    for reg_dir in readdir(registry_backup)
        src = joinpath(registry_backup, reg_dir)
        dst = joinpath(registry_path, reg_dir)
        if !isdir(dst) && isdir(src)
            cp(src, dst, force=true)
        end
    end
end

# Set up dev path pointing to our bundled source code
dev_path = joinpath(depot_path, "dev")
mkpath(dev_path)

# Link to our bundled dev packages (using symlinks on Unix, junctions on Windows)
pkgs_path = joinpath(@__DIR__, "pkg_sources")
if isdir(pkgs_path)
    println("Setting up package sources...")
    for pkg_dir in readdir(pkgs_path)
        src = joinpath(pkgs_path, pkg_dir)
        if isdir(src)
            dst = joinpath(dev_path, pkg_dir)
            if !isdir(dst)
                println("  Setting up: \$pkg_dir")
                try
                    symlink(src, dst)
                catch e
                    println("  Warning: Could not symlink \$pkg_dir: \$e")
                    println("  Attempting to copy instead...")
                    cp(src, dst, force=true)
                end
            end
        end
    end
end

# Activate the project
Pkg.activate(@__DIR__)

# Build all packages (this may take some time)
println("\\nBuilding all packages from source... (this may take a while)")
Pkg.build()

println("\\nSetup complete! You can now use BeforeIT.jl.")
"""

write(joinpath(bundle_dir, "setup.jl"), setup_script)

# 8. Create cross-platform startup script
startup_script = """
#!/usr/bin/env julia
# This script starts Julia with the correct environment configuration

# Get the directory where this script is located
script_dir = dirname(abspath(PROGRAM_FILE))

# Set the depot path to the local package directory
ENV["JULIA_DEPOT_PATH"] = joinpath(script_dir, "package_depot")

# Start Julia with the project activated
using Pkg
Pkg.activate(script_dir)

# Optional: print a welcome message
println(\"\"\"
BeforeIT.jl Environment
-----------------------
Package depot: \$(ENV["JULIA_DEPOT_PATH"])
Project: \$(Base.active_project())
\"\"\")

# Now let the user interact with the REPL
using REPL
REPL.run_repl()
"""

write(joinpath(bundle_dir, "start_julia.jl"), startup_script)

# 9. Create a README
readme = """
# BeforeIT.jl Cross-Platform Bundle

This bundle contains:
1. The complete BeforeIT.jl project
2. Source code for all dependencies from Manifest.toml
3. Registry backup (for package metadata)
4. Cross-platform setup scripts

## First-time setup (run once)

1. Make sure Julia is installed on your system
2. Run Julia and execute:
include("setup.jl")
This will build all packages from source - it may take some time.

## Regular usage

For regular usage, simply run the startup script:
julia start_julia.jl

Or set the environment variables manually:

On Windows
set JULIA_DEPOT_PATH=%CD%\package_depot
julia --project=.

On Unix (macOS/Linux)
export JULIA_DEPOT_PATH=$(pwd)/package_depot
julia --project=.

## Troubleshooting

If you encounter symlink issues on Windows (common in some secure environments):
1. Run Julia as administrator for the setup step, or
2. Edit setup.jl and replace the symlink command with a cp (copy) command
"""

write(joinpath(bundle_dir, "README.md"), readme)

println("\nBundle created successfully at: $bundle_dir")
println("Now zip this directory with:")
println("  cd $(dirname(bundle_dir))")
println("  zip -r beforeit_bundle.zip beforeit_bundle")
println("\nTransfer the zip file to your secure environment and follow the instructions in README.md.")