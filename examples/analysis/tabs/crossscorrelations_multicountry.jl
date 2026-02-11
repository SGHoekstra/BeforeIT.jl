# Multi-Country Cross-Correlation Analysis
# Compares cross-correlation and autocorrelation patterns across countries
# Builds on: examples/analysis/figs/crosscorrelations.jl (single-country version)
#
# Supports base model and extension variants (e.g., CANVAS, GrowthRateAR1).
# Set MODEL_VARIANT, PREDICTION_FOLDER, and EXTENSION_FILE below.

using StatsBase, LinearAlgebra, Statistics, Dates
using JLD2, FileIO, CSV, DataFrames, Plots
import BeforeIT as Bit

include("../../../src/utils/correlation_utils.jl")

# =============================================================================
# CONFIGURATION
# =============================================================================

CORRELATION_LAGS = 15
AUTOCORR_LAGS = 20

# Cross-correlation simulations need longer series for proper HP filtering
RUN_SIMULATION = true   # Set to false after first run
T_CROSSCORR = 40        # 10 years — long enough for HP filter + ±15 lag cross-correlations
N_SIMS_CROSSCORR = 100

VARIABLES = [
    "real_gdp_quarterly",
    "real_household_consumption_quarterly",
    "real_capitalformation_quarterly",
    "wages_quarterly",
    "gdp_deflator_quarterly"
]

# =============================================================================
# MODEL VARIANT CONFIGURATION
# =============================================================================
# Change these to run analysis for different model variants.
# Results will be saved to:
#   analysis/tabs/cross_country_results/{MODEL_VARIANT}/
#   analysis/figs/multicountry/crosscorrelations/{MODEL_VARIANT}/
#
# Options:
#   MODEL_VARIANT = "base"           PREDICTION_FOLDER = "abm_predictions_crosscorr"               EXTENSION_FILE = nothing
#   MODEL_VARIANT = "growth_rate"    PREDICTION_FOLDER = "abm_predictions_growth_rate_crosscorr"   EXTENSION_FILE = "../../GrowthRateAR1_extension.jl"
#   MODEL_VARIANT = "canvas"         PREDICTION_FOLDER = "abm_predictions_canvas_crosscorr"        EXTENSION_FILE = "../../CANVAS_extension.jl"

MODEL_VARIANT = "canvas"
PREDICTION_FOLDER = "abm_predictions_canvas_crosscorr"
EXTENSION_FILE = "../../CANVAS_extension.jl"

# Derive simulation suffix from variant
SIMULATION_SUFFIX = MODEL_VARIANT == "base" ? "crosscorr" : "$(MODEL_VARIANT)_crosscorr"

# =============================================================================
# EXTENSION INCLUDE (must be at top level for method dispatch)
# =============================================================================

if EXTENSION_FILE !== nothing
    include(joinpath(@__DIR__, EXTENSION_FILE))
    @info "Loaded extension: $EXTENSION_FILE (variant: $MODEL_VARIANT)"
end

# =============================================================================
# SIMULATION PHASE (run once, then set RUN_SIMULATION = false)
# =============================================================================

if RUN_SIMULATION
    if MODEL_VARIANT != "base" && !@isdefined(create_model)
        error("Extension variant '$MODEL_VARIANT' requires create_model() factory. " *
              "Check that EXTENSION_FILE '$EXTENSION_FILE' defines it.")
    end

    @info "Running T=$T_CROSSCORR simulations for cross-correlation analysis (variant: $MODEL_VARIANT)..."

    for country in Bit.discover_countries_with_calibration()
        # Skip if simulations already exist
        sim_folder = joinpath("data", country, "simulations_$(SIMULATION_SUFFIX)")
        if isdir(sim_folder)
            existing_files = filter(f -> endswith(f, ".jld2"), readdir(sim_folder))
            if length(existing_files) >= 40
                @info "Skipping $country: simulations already exist ($(length(existing_files)) files)"
                continue
            end
        end

        @info "Simulating $country"
        try
            calibration = Bit.load_calibration_data(country)
            folder = "data/$(country)"

            if MODEL_VARIANT == "base"
                Bit.save_all_simulations(folder; T=T_CROSSCORR, n_sims=N_SIMS_CROSSCORR,
                                         output_suffix=SIMULATION_SUFFIX)
            else
                Bit.save_all_simulations(folder; T=T_CROSSCORR, n_sims=N_SIMS_CROSSCORR,
                                         model_factory=create_model,
                                         output_suffix=SIMULATION_SUFFIX)
            end

            Bit.save_all_predictions_from_sims(folder, calibration.data;
                                                simulation_suffix="simulations_$(SIMULATION_SUFFIX)",
                                                prediction_suffix=PREDICTION_FOLDER)
            @info "✓ $country"
        catch e
            @error "✗ $country: $e"
        end
    end
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function discover_countries_with_crosscorr_predictions()
    countries = String[]
    for entry in readdir("data")
        pred_dir = joinpath("data", entry, PREDICTION_FOLDER)
        if isdir(pred_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(pred_dir)))
            push!(countries, entry)
        end
    end
    return sort(countries)
end

function compute_country_correlations(country, variables, corr_lags, auto_lags;
                                       prediction_folder=PREDICTION_FOLDER)
    calibration = Bit.load_calibration_data(country)
    real_data = calibration.data
    folder = "data/$(country)/$(prediction_folder)"

    if !isdir(folder)
        return nothing
    end

    files = filter(f -> endswith(f, ".jld2"), readdir(folder))
    if isempty(files)
        return nothing
    end

    # Get structure from first file
    first_pred = load(joinpath(folder, first(files)))["predictions_dict"]
    n_quarters = length(files)

    # Create HP filter cache for real data
    hp_cache = create_hp_filter_cache(real_data, variables, 10)

    # Process simulation data
    # n_seeds must match gdp_size[2] so indexing is correct
    gdp_size = size(first_pred["real_gdp_quarterly"])
    _, crosscorr, autocorr, _ = process_all_simulation_data(
        folder, variables, gdp_size, gdp_size[2], n_quarters,
        corr_lags, 20, auto_lags, variables[1:min(3, length(variables))], 10
    )

    # Process real data correlations
    real_crosscorr, real_autocorr, _ = process_real_data_correlations(
        real_data, hp_cache, variables, corr_lags, auto_lags
    )

    # Compute RMSE and per-lag absolute error for each variable
    results = Dict{String, Dict{String, Any}}()
    for var in variables
        if haskey(crosscorr, var) && haskey(real_crosscorr, var)
            sim_cross = crosscorr[var]
            real_cross = vec(real_crosscorr[var])
            sim_auto = autocorr[var]
            real_auto = vec(real_autocorr[var])

            # RMSE across lags and simulations
            cross_rmse = sqrt(mean((mean(sim_cross, dims=2) .- real_cross).^2))
            auto_rmse = sqrt(mean((mean(sim_auto, dims=2) .- real_auto).^2))

            # Per-lag absolute error: |ABM_mean_at_lag - Real_at_lag|
            cross_lag_error = abs.(vec(mean(sim_cross, dims=2)) .- real_cross)
            auto_lag_error = abs.(vec(mean(sim_auto, dims=2)) .- real_auto)

            results[var] = Dict(
                "cross_rmse" => cross_rmse,
                "auto_rmse" => auto_rmse,
                "cross_lag_error" => cross_lag_error,
                "auto_lag_error" => auto_lag_error
            )
        end
    end

    return results
end

# =============================================================================
# MAIN SCRIPT
# =============================================================================

@info "Multi-Country Cross-Correlation Analysis (variant: $MODEL_VARIANT)"

countries = discover_countries_with_crosscorr_predictions()
@info "Found $(length(countries)) countries with cross-correlation predictions"

# Collect results
all_results = Dict{String, Dict{String, Dict{String, Any}}}()

for country in countries
    @info "Processing $country"
    try
        results = compute_country_correlations(country, VARIABLES, CORRELATION_LAGS, AUTOCORR_LAGS)
        if results !== nothing && !isempty(results)
            all_results[country] = results
            @info "✓ $country"
        else
            @warn "✗ $country: no data"
        end
    catch e
        @error "✗ $country: $e"
    end
end

if isempty(all_results)
    @error "No countries processed successfully"
    exit(1)
end

valid_countries = collect(keys(all_results))
@info "Successfully processed $(length(valid_countries)) countries"

# Create output directories
mkpath("analysis/tabs/cross_country_results/$(MODEL_VARIANT)")
mkpath("analysis/figs/multicountry/crosscorrelations/$(MODEL_VARIANT)/heatmaps")

# Create RMSE matrices
n_countries = length(valid_countries)
n_variables = length(VARIABLES)

cross_matrix = fill(NaN, n_countries, n_variables)
auto_matrix = fill(NaN, n_countries, n_variables)

for (i, country) in enumerate(valid_countries)
    for (j, var) in enumerate(VARIABLES)
        if haskey(all_results[country], var)
            cross_matrix[i, j] = all_results[country][var]["cross_rmse"]
            auto_matrix[i, j] = all_results[country][var]["auto_rmse"]
        end
    end
end

# Export CSVs
clean_vars = [replace(replace(v, "_quarterly" => ""), "_" => " ") for v in VARIABLES]

cross_df = DataFrame(cross_matrix, clean_vars)
cross_df.country = valid_countries
select!(cross_df, :country, Not(:country))
CSV.write("analysis/tabs/cross_country_results/$(MODEL_VARIANT)/cross_correlations_overall_rmse.csv", cross_df)

auto_df = DataFrame(auto_matrix, clean_vars)
auto_df.country = valid_countries
select!(auto_df, :country, Not(:country))
CSV.write("analysis/tabs/cross_country_results/$(MODEL_VARIANT)/auto_correlations_overall_rmse.csv", auto_df)

# =============================================================================
# PER-VARIABLE LAG HEATMAPS
# =============================================================================

n_cross_lags = 2 * CORRELATION_LAGS + 1  # 31
n_auto_lags = AUTOCORR_LAGS + 1          # 21

cross_lag_matrices = Dict{String, Matrix{Float64}}()
auto_lag_matrices = Dict{String, Matrix{Float64}}()

for var in VARIABLES
    cm = fill(NaN, n_countries, n_cross_lags)
    am = fill(NaN, n_countries, n_auto_lags)
    for (i, country) in enumerate(valid_countries)
        if haskey(all_results[country], var)
            cm[i, :] = all_results[country][var]["cross_lag_error"]
            am[i, :] = all_results[country][var]["auto_lag_error"]
        end
    end
    cross_lag_matrices[var] = cm
    auto_lag_matrices[var] = am
end

mkpath("analysis/figs/multicountry/crosscorrelations/$(MODEL_VARIANT)/heatmaps/by_variable")

cross_lag_labels = collect(-CORRELATION_LAGS:CORRELATION_LAGS)
auto_lag_labels = collect(0:AUTOCORR_LAGS)

for var in VARIABLES
    clean_var = replace(var, "_quarterly" => "")

    # Cross-correlation lag heatmap
    cm = cross_lag_matrices[var]
    p = heatmap(cm,
        color=:Blues, colorbar=false,
        size=(max(800, n_cross_lags * 25), max(400, n_countries * 40)),
        margin=15Plots.mm,
        xticks=(1:5:n_cross_lags, cross_lag_labels[1:5:end]),
        yticks=(1:n_countries, valid_countries))
    savefig(p, "analysis/figs/multicountry/crosscorrelations/$(MODEL_VARIANT)/heatmaps/by_variable/$(clean_var)_cross_correlations_lag_rmse_heatmap.png")

    # Auto-correlation lag heatmap
    am = auto_lag_matrices[var]
    p = heatmap(am,
        color=:Blues, colorbar=false,
        size=(max(600, n_auto_lags * 30), max(400, n_countries * 40)),
        margin=15Plots.mm,
        xticks=(1:5:n_auto_lags, auto_lag_labels[1:5:end]),
        yticks=(1:n_countries, valid_countries))
    savefig(p, "analysis/figs/multicountry/crosscorrelations/$(MODEL_VARIANT)/heatmaps/by_variable/$(clean_var)_auto_correlations_lag_rmse_heatmap.png")

    @info "✓ Saved $clean_var lag heatmaps"
end

# =============================================================================
# OVERALL RMSE HEATMAPS
# =============================================================================

# Create heatmaps
for (matrix, name, _) in [
    (cross_matrix, "cross_correlations", "Cross-Correlations RMSE"),
    (auto_matrix, "auto_correlations", "Auto-Correlations RMSE")
]
    var_labels = [replace(v, "_quarterly" => "") for v in VARIABLES]
    valid_vals = matrix[.!isnan.(matrix)]
    min_val, max_val = extrema(valid_vals)
    val_range = max_val - min_val

    p = heatmap(matrix,
        color=:Blues, colorbar=false,
        size=(max(600, n_variables * 120), max(400, n_countries * 40)),
        margin=15Plots.mm,
        xticks=(1:n_variables, var_labels),
        yticks=(1:n_countries, valid_countries),
        xrotation=45)

    for i in 1:n_countries, j in 1:n_variables
        val = matrix[i, j]
        if !isnan(val)
            normalized = val_range > 0 ? (val - min_val) / val_range : 0.0
            txt_color = normalized > 0.7 ? :white : :black
            annotate!(j, i, text(string(round(val, digits=3)), 6, txt_color, :center))
        end
    end

    savefig(p, "analysis/figs/multicountry/crosscorrelations/$(MODEL_VARIANT)/heatmaps/$(name)_overall_rmse_heatmap.png")
    @info "✓ Saved $name heatmap"
end

@info "Done."
