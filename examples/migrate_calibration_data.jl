using JLD2, FileIO

"""
Automatic Data Migration Script: CalibrateBeforeIT.jl → BeforeIT.jl

This script automatically discovers available countries and time periods from
CalibrateBeforeIT.jl output and migrates them to the folder structure that
BeforeIT.jl's save_all_simulations function expects.

Note that this script assumes that the CalibrateBeforeIT.jl output is located
in a folder named `../CalibrateBeforeIT.jl/data/020_calibration_output` relative
to the current working directory. Adjust the `CBIT_PATH` constant below if needed.
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to CalibrateBeforeIT.jl output (assumes same parent directory)
const CBIT_PATH = "../CalibrateBeforeIT.jl/data/020_calibration_output"

# BeforeIT.jl data directory
const BEFOREIT_DATA_PATH = "data"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    discover_available_countries()

Automatically discover which countries are available in CalibrateBeforeIT.jl output.

# Returns
- Vector of country codes (e.g., ["AT", "BE", "NL", "IT", ...])
"""
function discover_available_countries()
    if !isdir(CBIT_PATH)
        error("CalibrateBeforeIT.jl output directory not found: $CBIT_PATH")
    end

    # Get all subdirectories (country codes)
    all_entries = readdir(CBIT_PATH)
    countries = filter(entry -> isdir(joinpath(CBIT_PATH, entry)), all_entries)

    @info "Discovered $(length(countries)) countries in CalibrateBeforeIT.jl output"
    @info "Countries: $(join(sort(countries), ", "))"

    return sort(countries)
end

"""
    discover_available_periods(country_code)

Automatically discover which time periods are available for a specific country.

# Arguments
- `country_code`: Two-letter country code (e.g., "NL", "IT")

# Returns
- Vector of period strings (e.g., ["2020Q1", "2020Q2", ...])
"""
function discover_available_periods(country_code)
    country_dir = joinpath(CBIT_PATH, country_code)

    if !isdir(country_dir)
        @warn "Country directory not found: $country_dir"
        return String[]
    end

    # Find all parameter files
    all_files = readdir(country_dir)
    param_files = filter(f -> endswith(f, "_parameters_initial_conditions.jld2"), all_files)

    # Extract periods from filenames
    periods = map(f -> replace(f, "_parameters_initial_conditions.jld2" => ""), param_files)

    @info "  Found $(length(periods)) periods for $country_code: $(join(sort(periods), ", "))"

    return sort(periods)
end

"""
    migrate_calibration_data(source_dir, target_dir, country_code)

Migrate calibration data (CalibrationData object) to BeforeIT.jl format.

# Arguments
- `source_dir`: Source directory (CalibrateBeforeIT.jl country folder)
- `target_dir`: Target directory (BeforeIT.jl country folder)
- `country_code`: Country code for logging
"""
function migrate_calibration_data(source_dir, target_dir, country_code)
    # Priority order: calibration_object.jld2 > calibration_data.jld2
    calibration_object_file = joinpath(source_dir, "calibration_object.jld2")
    calibration_data_file = joinpath(source_dir, "calibration_data.jld2")

    if isfile(calibration_object_file)
        target_file = joinpath(target_dir, "calibration_object.jld2")
        cp(calibration_object_file, target_file, force=true)
        @info "  ✓ Copied calibration_object.jld2 (complete CalibrationData)"

    elseif isfile(calibration_data_file)
        target_file = joinpath(target_dir, "calibration_data.jld2")
        cp(calibration_data_file, target_file, force=true)
        @info "  ✓ Copied calibration_data.jld2 (legacy format)"

    else
        @warn "  ⚠ No calibration data found for $country_code"
    end
end

"""
    migrate_period_data(source_dir, target_dir, period)

Migrate parameters and initial conditions for a specific time period.

# Arguments
- `source_dir`: Source directory (CalibrateBeforeIT.jl country folder)
- `target_dir`: Target directory (BeforeIT.jl country folder)
- `period`: Time period string (e.g., "2020Q1")
"""
function migrate_period_data(source_dir, target_dir, period)
    source_file = joinpath(source_dir, "$(period)_parameters_initial_conditions.jld2")

    if !isfile(source_file)
        @warn "  ⚠ Missing parameter file for period $period"
        return false
    end

    try
        # Load combined parameter file
        data = load(source_file)

        # Check that required keys exist
        if !haskey(data, "parameters") || !haskey(data, "initial_conditions")
            @error "  ✗ Invalid file structure for $period - missing required keys"
            return false
        end

        # Save to separate BeforeIT.jl files
        param_file = joinpath(target_dir, "parameters", "$(period).jld2")
        init_file = joinpath(target_dir, "initial_conditions", "$(period).jld2")

        save(param_file, data["parameters"])
        save(init_file, data["initial_conditions"])

        return true

    catch e
        @error "  ✗ Failed to migrate $period: $e"
        return false
    end
end

"""
    migrate_country_data(country_code, available_periods)

Migrate all data for a specific country to BeforeIT.jl format.

# Arguments
- `country_code`: Two-letter country code
- `available_periods`: Vector of available time periods

# Returns
- Number of successfully migrated periods
"""
function migrate_country_data(country_code, available_periods)
    if isempty(available_periods)
        @warn "No periods found for $country_code, skipping"
        return 0
    end

    @info "Migrating $country_code: $(length(available_periods)) periods"

    # Set up directories
    source_dir = joinpath(CBIT_PATH, country_code)
    target_dir = joinpath(BEFOREIT_DATA_PATH, lowercase(country_code))

    # Create BeforeIT.jl directory structure
    mkpath(target_dir)
    mkpath(joinpath(target_dir, "parameters"))
    mkpath(joinpath(target_dir, "initial_conditions"))

    # Migrate calibration data
    migrate_calibration_data(source_dir, target_dir, country_code)

    # Migrate all periods
    successful_migrations = 0
    for period in available_periods
        if migrate_period_data(source_dir, target_dir, period)
            successful_migrations += 1
        end
    end

    @info "  Summary: $successful_migrations/$(length(available_periods)) periods migrated successfully"

    return successful_migrations
end

# =============================================================================
# MAIN MIGRATION SCRIPT
# =============================================================================


@info "=== CalibrateBeforeIT.jl → BeforeIT.jl Data Migration ==="
@info "Source: $CBIT_PATH"
@info "Target: $BEFOREIT_DATA_PATH"
@info ""

# Discover available countries
try
    available_countries = discover_available_countries()

    if isempty(available_countries)
        @warn "No countries found in CalibrateBeforeIT.jl output"
        return
    end

    # Migration summary
    total_countries = 0
    total_periods = 0
    successful_countries = 0

    # Process each country
    for country_code in available_countries
        @info "Processing $country_code..."

        # Discover available periods for this country
        available_periods = discover_available_periods(country_code)

        if !isempty(available_periods)
            # Migrate country data
            migrated_periods = migrate_country_data(country_code, available_periods)

            total_countries += 1
            total_periods += length(available_periods)

            if migrated_periods > 0
                successful_countries += 1
            end
        end

        @info ""  # Add blank line between countries
    end

    # Final summary
    @info "=== Migration Complete ==="
    @info "Processed: $total_countries countries"
    @info "Successful: $successful_countries countries"
    @info "Total periods: $total_periods"
    @info ""
    @info "BeforeIT.jl data is now ready for save_all_simulations()"

catch e
    @error "Migration failed: $e"
    rethrow(e)
end
