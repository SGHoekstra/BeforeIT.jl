# Multi-Country Simulation and Table Creation
# Runs the prediction pipeline for all countries with calibration data
#
# Supports base model and extension variants (e.g., CANVAS, GrowthRateAR1).
# Set MODEL_VARIANT, PREDICTION_FOLDER, and EXTENSION_FILE below.

import BeforeIT as Bit
using Dates, JLD2, CSV, DataFrames, Statistics

# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_SIMULATION = false  # Run simulations for all countries
RUN_ANALYSIS = true     # Generate error tables

T = 12          # Forecast horizon (quarters)
N_SIMS = 100    # Number of simulations per quarter

QUARTERS = DateTime(2010, 03, 31):Dates.Month(3):DateTime(2019, 12, 31)
HORIZONS = [1, 2, 4, 8, 12]

# =============================================================================
# MODEL VARIANT CONFIGURATION
# =============================================================================
# Change these to run analysis for different model variants.
# Results will be saved to: data/{country}/analysis/{MODEL_VARIANT}/
#
# Options:
#   MODEL_VARIANT = "base"           PREDICTION_FOLDER = "abm_predictions"               EXTENSION_FILE = nothing
#   MODEL_VARIANT = "growth_rate"    PREDICTION_FOLDER = "abm_predictions_growth_rate"   EXTENSION_FILE = "../../GrowthRateAR1_extension.jl"
#   MODEL_VARIANT = "canvas"         PREDICTION_FOLDER = "abm_predictions_canvas"        EXTENSION_FILE = "../../CANVAS_extension.jl"

MODEL_VARIANT = "growth_rate"
PREDICTION_FOLDER = "abm_predictions_growth_rate"
EXTENSION_FILE = "../../GrowthRateAR1_extension.jl"

# =============================================================================
# EXTENSION INCLUDE (must be at top level for method dispatch)
# =============================================================================

if EXTENSION_FILE !== nothing
    include(joinpath(@__DIR__, EXTENSION_FILE))
    @info "Loaded extension: $EXTENSION_FILE (variant: $MODEL_VARIANT)"
end

# =============================================================================
# HELPER: Discover countries with a given prediction folder
# =============================================================================

"""
Discover countries that have the configured prediction folder with .jld2 files.
"""
function discover_countries_with_variant_predictions(prediction_folder::String)
    data_dir = "data"
    countries = String[]

    for entry in readdir(data_dir)
        country_dir = joinpath(data_dir, entry)
        if isdir(country_dir)
            preds_dir = joinpath(country_dir, prediction_folder)
            if isdir(preds_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(preds_dir)))
                push!(countries, entry)
            end
        end
    end

    return sort(countries)
end

# =============================================================================
# SIMULATION PHASE
# =============================================================================

if RUN_SIMULATION
    if MODEL_VARIANT != "base" && !@isdefined(create_model)
        error("Extension variant '$MODEL_VARIANT' requires create_model() factory. " *
              "Check that EXTENSION_FILE '$EXTENSION_FILE' defines it.")
    end

    @info "Starting simulations for all countries (variant: $MODEL_VARIANT)..."

    for country in Bit.discover_countries_with_calibration()
        folder = "data/$(country)"

        # For non-base variants, check if simulations already exist
        if MODEL_VARIANT != "base"
            sim_folder = joinpath(folder, "simulations_$(MODEL_VARIANT)")
            if isdir(sim_folder)
                existing_files = filter(f -> endswith(f, ".jld2"), readdir(sim_folder))
                if length(existing_files) >= 40
                    @info "Skipping $country: simulations already exist ($(length(existing_files)) files)"
                    continue
                end
            end
        end

        @info "Processing $country"

        try
            if MODEL_VARIANT == "base"
                # Base model: standard pipeline
                calibration = Bit.load_calibration_data(country)
                Bit.save_all_simulations(folder; T = T, n_sims = N_SIMS)
                Bit.save_all_predictions_from_sims(folder, calibration.data)
            else
                # Extension variant: use model_factory and suffixed folders
                Bit.save_all_simulations(folder;
                    T = T,
                    n_sims = N_SIMS,
                    model_factory = create_model,
                    output_suffix = MODEL_VARIANT
                )

                # Convert simulations to predictions
                calibration = Bit.load_calibration_data(country)
                mkpath(joinpath(folder, PREDICTION_FOLDER))
                Bit.save_all_predictions_from_sims(folder, calibration.data;
                    simulation_suffix = "simulations_$(MODEL_VARIANT)",
                    prediction_suffix = PREDICTION_FOLDER
                )
            end

            @info "Completed $country"
        catch e
            @error "Failed $country: $e"
            for (exc, bt) in Base.catch_stack()
                showerror(stderr, exc, bt)
                println(stderr)
            end
        end
    end
end

# =============================================================================
# ANALYSIS PHASE
# =============================================================================

if RUN_ANALYSIS
    @info "Generating error tables (variant: $MODEL_VARIANT)..."

    # Load analysis functions
    include(joinpath(@__DIR__, "analysis_utils.jl"))
    include(joinpath(@__DIR__, "error_table_ar.jl"))
    include(joinpath(@__DIR__, "error_table_abm.jl"))
    include(joinpath(@__DIR__, "error_table_validation_var.jl"))
    include(joinpath(@__DIR__, "error_table_validation_abm.jl"))

    # Discover countries based on variant
    analysis_countries = if MODEL_VARIANT == "base"
        Bit.discover_countries_with_predictions()
    else
        discover_countries_with_variant_predictions(PREDICTION_FOLDER)
    end

    for country in analysis_countries
        @info "Generating tables for $country (variant: $(MODEL_VARIANT))"

        try
            calibration = Bit.load_calibration_data(country)
            data = calibration.data
            ea = calibration.ea

            # Ensure analysis folder exists
            mkpath(joinpath("data", country, "analysis", MODEL_VARIANT))

            # AR/VAR benchmarks (use same variant folder for consistency)
            error_table_ar(country, ea, data, QUARTERS, HORIZONS;
                          model_variant=MODEL_VARIANT)
            error_table_validation_var(country, ea, data, QUARTERS, HORIZONS; model_variant=MODEL_VARIANT)

            # ABM predictions (load from prediction_folder, save to variant folder)
            error_table_abm(country, ea, data, QUARTERS, HORIZONS;
                           model_variant=MODEL_VARIANT, prediction_folder=PREDICTION_FOLDER)
            error_table_validation_abm(country, ea, data, QUARTERS, HORIZONS;
                                       model_variant=MODEL_VARIANT, prediction_folder=PREDICTION_FOLDER)

            @info "Completed $country"
        catch e
            @error "Failed $country: $e"
        end
    end
end

@info "Done."
