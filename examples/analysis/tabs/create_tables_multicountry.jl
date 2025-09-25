# Multi-Country Table Creation Script
# This script creates forecast comparison tables for all countries with calibration data
# Extends the functionality of create_tables.jl to work across multiple countries

import BeforeIT as Bit
using Dates, DelimitedFiles, Statistics, Printf, LaTeXStrings, CSV, DataFrames, FileIO, MAT, JLD2

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Control flags - set these to control which parts of the script run
RUN_SIMULATION = false   # Set to true to run simulations
RUN_ANALYSIS = true    # Set to true to run table generation

# Simulation parameters
T = 12
N_SIMS = 100
QUARTERS = DateTime(2010, 03, 31):Dates.Month(3):DateTime(2019, 12, 31)
HORIZONS = [1, 2, 4, 8, 12]

# =============================================================================
# SIMULATION SETUP SCRIPT
# =============================================================================

if RUN_SIMULATION
    @info "Starting simulation setup for all countries with calibration data"

    # Discover available countries
    countries = Bit.discover_countries_with_calibration()

    for country in countries
        @info "Setting up simulations for $country"

        try
            # Load calibration data from migrated structure
            calibration = Bit.load_calibration_data(country)

            default_folder = "data/$(country)"

            # Check if folder exists (should already exist from migration)
            if !isdir(default_folder)
                error("Data folder not found for $country at $default_folder.")
            end

            # Check if parameters and initial conditions already exist
            param_dir = joinpath(default_folder, "parameters")
            init_dir = joinpath(default_folder, "initial_conditions")

            param_exists = isdir(param_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(param_dir)))
            init_exists = isdir(init_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(init_dir)))

            if !param_exists || !init_exists
                @warn "Parameters or initial conditions missing for $country. Skipping."
                continue
            end

            # Run simulations using the migrated data
            Bit.save_all_simulations(default_folder; T = T, n_sims = N_SIMS)

            # Generate predictions from simulations
            # Use calibration.data as real_data for predictions
            Bit.save_all_predictions_from_sims(default_folder, calibration.data)

            @info "Simulation data setup completed successfully for $country"

        catch e
            @error "Failed to setup simulation data for $country: $e"
        end
    end

    @info "Simulation setup completed for all countries"
end

# =============================================================================
# ANALYSIS SCRIPT
# =============================================================================

if RUN_ANALYSIS
    @info "\n=== ANALYSIS PHASE ==="
    @info "Starting table generation for all countries with prediction data..."

    # Discover countries with prediction data
    countries_with_predictions = Bit.discover_countries_with_predictions()

    if isempty(countries_with_predictions)
        @error "No countries with prediction data found!"
        exit(1)
    end

    # Load utility functions for creating error tables
    dir = @__DIR__
    include(joinpath(dir, "analysis_utils.jl"))
    include(joinpath(dir, "error_table_ar.jl"))
    include(joinpath(dir, "error_table_abm.jl"))
    include(joinpath(dir, "error_table_validation_var.jl"))
    include(joinpath(dir, "error_table_validation_abm.jl"))

    # Track analysis progress
    analysis_successful = 0
    analysis_failed = String[]

    # Process each country for table generation
    for country in countries_with_predictions
        @info "Generating tables for $country"

        try
            # Load calibration data
            calibration = Bit.load_calibration_data(country)
            data = calibration.data
            ea = calibration.ea

            # Generate all tables
            error_table_ar(country, ea, data, QUARTERS, HORIZONS)
            error_table_validation_var(country, ea, data, QUARTERS, HORIZONS)
            error_table_abm(country, ea, data, QUARTERS, HORIZONS)
            error_table_validation_abm(country, ea, data, QUARTERS, HORIZONS)

            analysis_successful += 1
            @info "✓ Tables generated successfully for $country"

        catch e
            @error "✗ Failed to generate tables for $country: $e"
            push!(analysis_failed, country)
        end
    end

    @info "\n=== ANALYSIS SUMMARY ==="
    @info "Total countries: $(length(countries_with_predictions))"
    @info "Analysis successful: $analysis_successful"
    @info "Analysis failed: $(length(analysis_failed))"

    if !isempty(analysis_failed)
        @info "Failed countries: $(join(analysis_failed, ", "))"
    end
end

# =============================================================================
# FINAL SUMMARY
# =============================================================================

@info "\n" * "="^60
@info "=== MULTI-COUNTRY TABLE CREATION COMPLETED ==="
@info "="^60