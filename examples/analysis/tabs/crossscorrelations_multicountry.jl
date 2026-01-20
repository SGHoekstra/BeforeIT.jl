# Multi-Country Cross-Correlation Analysis
# Compares cross-correlation and autocorrelation patterns across countries
# Builds on: examples/analysis/figs/crosscorrelations.jl (single-country version)

using StatsBase, LinearAlgebra, Statistics, Dates
using JLD2, FileIO, CSV, DataFrames, Plots
import BeforeIT as Bit

include("../../../src/utils/correlation_utils.jl")

# =============================================================================
# CONFIGURATION
# =============================================================================

T = 16
N_SIMS = 20
CORRELATION_LAGS = 15
AUTOCORR_LAGS = 20

VARIABLES = [
    "real_gdp_quarterly",
    "real_household_consumption_quarterly",
    "real_capitalformation_quarterly",
    "wages_quarterly",
    "gdp_deflator_quarterly"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function compute_country_correlations(country, variables, n_sims, corr_lags, auto_lags)
    calibration = Bit.load_calibration_data(country)
    real_data = calibration.data
    folder = "data/$(country)/abm_predictions"

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
    gdp_size = size(first_pred["real_gdp_quarterly"])
    _, crosscorr, autocorr, _ = process_all_simulation_data(
        folder, variables, gdp_size, n_sims, n_quarters,
        corr_lags, 20, auto_lags, variables[1:min(3, length(variables))], 10
    )

    # Process real data correlations
    real_crosscorr, real_autocorr, _ = process_real_data_correlations(
        real_data, hp_cache, variables, corr_lags, auto_lags
    )

    # Compute RMSE for each variable
    results = Dict{String, Dict{String, Float64}}()
    for var in variables
        if haskey(crosscorr, var) && haskey(real_crosscorr, var)
            sim_cross = crosscorr[var]
            real_cross = vec(real_crosscorr[var])
            sim_auto = autocorr[var]
            real_auto = vec(real_autocorr[var])

            # RMSE across lags and simulations
            cross_rmse = sqrt(mean((mean(sim_cross, dims=2) .- real_cross).^2))
            auto_rmse = sqrt(mean((mean(sim_auto, dims=2) .- real_auto).^2))

            results[var] = Dict("cross_rmse" => cross_rmse, "auto_rmse" => auto_rmse)
        end
    end

    return results
end

# =============================================================================
# MAIN SCRIPT
# =============================================================================

@info "Multi-Country Cross-Correlation Analysis"

countries = Bit.discover_countries_with_predictions()
@info "Found $(length(countries)) countries"

# Collect results
all_results = Dict{String, Dict{String, Dict{String, Float64}}}()

for country in countries
    @info "Processing $country"
    try
        results = compute_country_correlations(country, VARIABLES, N_SIMS, CORRELATION_LAGS, AUTOCORR_LAGS)
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
mkpath("analysis/tabs/cross_country_results")
mkpath("analysis/figs/multicountry/crosscorrelations/heatmaps")

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
CSV.write("analysis/tabs/cross_country_results/cross_correlations_overall_rmse.csv", cross_df)

auto_df = DataFrame(auto_matrix, clean_vars)
auto_df.country = valid_countries
select!(auto_df, :country, Not(:country))
CSV.write("analysis/tabs/cross_country_results/auto_correlations_overall_rmse.csv", auto_df)

# Create heatmaps
for (matrix, name, title) in [
    (cross_matrix, "cross_correlations", "Cross-Correlations RMSE"),
    (auto_matrix, "auto_correlations", "Auto-Correlations RMSE")
]
    var_labels = [replace(v, "_quarterly" => "") for v in VARIABLES]
    p = heatmap(matrix,
        title=title,
        xlabel="Variables", ylabel="Countries",
        color=:viridis,
        size=(max(600, n_variables * 120), max(400, n_countries * 40)),
        margin=15Plots.mm,
        xticks=(1:n_variables, var_labels),
        yticks=(1:n_countries, valid_countries),
        xrotation=45)

    for i in 1:n_countries, j in 1:n_variables
        val = matrix[i, j]
        if !isnan(val)
            annotate!(j, i, text(string(round(val, digits=3)), 6, :white, :center))
        end
    end

    savefig(p, "analysis/figs/multicountry/crosscorrelations/heatmaps/$(name)_overall_rmse_heatmap.png")
    @info "✓ Saved $name heatmap"
end

@info "Done."
