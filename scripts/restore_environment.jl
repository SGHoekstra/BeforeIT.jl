# restore_environment.jl
using Pkg

println("Restoring your original Julia environment...")

# 1. Get a list of all developed packages
dev_path = joinpath(DEPOT_PATH[1], "dev")
developed_pkgs = []

if isdir(dev_path)
    for pkg_dir in readdir(dev_path)
        if isdir(joinpath(dev_path, pkg_dir))
            push!(developed_pkgs, pkg_dir)
        end
    end
end

println("Found $(length(developed_pkgs)) developed packages to restore:")
for pkg in developed_pkgs
    println("  - $pkg")
end

# 2. Activate your project
println("\nActivating your project...")
Pkg.activate(".")

# 3. Free all developed packages, returning them to normal registry versions
println("\nReverting packages to registry versions...")
for pkg_name in developed_pkgs
    println("  Freeing $pkg_name...")
    try
        Pkg.free(pkg_name)
    catch e
        println("    Warning: Could not free $pkg_name: $e")
    end
end

# 4. Remove any remaining bad packages and reinstall from scratch
println("\nReinstalling packages from registry...")
Pkg.resolve()
Pkg.instantiate(verbose=true)

# 5. Remove the development directory completely to ensure no lingering problems
if isdir(dev_path) && !isempty(readdir(dev_path))
    println("\nRemove the development directory? This will delete ALL your developed packages.")
    println("Development directory: $dev_path")
    print("Type 'yes' to confirm or anything else to skip: ")
    response = readline()
    
    if lowercase(response) == "yes"
        println("Removing development directory...")
        for item in readdir(dev_path)
            item_path = joinpath(dev_path, item)
            if isdir(item_path)
                try
                    rm(item_path, recursive=true, force=true)
                    println("  Removed $item")
                catch e
                    println("  Could not remove $item: $e")
                end
            end
        end
    else
        println("Skipping removal of development directory.")
    end
end

println("\nEnvironment restoration complete!")
println("Your packages should now be back to normal registry versions.")