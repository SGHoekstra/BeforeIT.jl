# Multi-Country Simulation and Table Creation
# Runs the prediction pipeline for all countries with calibration data

import BeforeIT as Bit
using Dates

# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_SIMULATION = true   # Run simulations for all countries
RUN_ANALYSIS = true     # Generate error tables

T = 12          # Forecast horizon (quarters)
N_SIMS = 100     # Number of simulations per quarter

QUARTERS = DateTime(2010, 03, 31):Dates.Month(3):DateTime(2019, 12, 31)
HORIZONS = [1, 2, 4, 8, 12]

# =============================================================================
# SIMULATION PHASE
# =============================================================================

if RUN_SIMULATION
    @info "Starting simulations for all countries..."

    for country in Bit.discover_countries_with_calibration()
        @info "Processing $country"

        try
            calibration = Bit.load_calibration_data(country)
            folder = "data/$(country)"

            # Run simulations and generate predictions (same as prediction_pipeline_multiple.jl)
            Bit.save_all_simulations(folder; T = T, n_sims = N_SIMS)
            Bit.save_all_predictions_from_sims(folder, calibration.data)

            @info "✓ $country completed"
        catch e
            @error "✗ $country failed: $e"
        end
    end
end

# =============================================================================
# ANALYSIS PHASE
# =============================================================================

if RUN_ANALYSIS
    @info "Generating error tables..."

    # Load analysis functions
    include(joinpath(@__DIR__, "analysis_utils.jl"))
    include(joinpath(@__DIR__, "error_table_ar.jl"))
    include(joinpath(@__DIR__, "error_table_abm.jl"))
    include(joinpath(@__DIR__, "error_table_validation_var.jl"))
    include(joinpath(@__DIR__, "error_table_validation_abm.jl"))

    for country in Bit.discover_countries_with_predictions()
        @info "Generating tables for $country"

        try
            calibration = Bit.load_calibration_data(country)
            data = calibration.data
            ea = calibration.ea

            error_table_ar(country, ea, data, QUARTERS, HORIZONS)
            error_table_abm(country, ea, data, QUARTERS, HORIZONS)
            error_table_validation_var(country, ea, data, QUARTERS, HORIZONS)
            error_table_validation_abm(country, ea, data, QUARTERS, HORIZONS)

            @info "✓ $country completed"
        catch e
            @error "✗ $country failed: $e"
        end
    end
end

@info "Done."
