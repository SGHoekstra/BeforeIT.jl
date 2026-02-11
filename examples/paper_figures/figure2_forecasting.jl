"""
=====================================================================
FIGURE 2: FORECASTING COMPARISON
=====================================================================

Recreates Figure 2 from Dawid et al. (2024) "Implications of Behavioral
Rules in Agent-Based Macroeconomics"

This script compares forecasting performance across four pricing models:
- CANVAS: Demand-pull with cost-push
- CATS: Demand-pull without cost-push (cost absorption)
- EUBI: Market share + competitor prices
- KS: Pure market share (simplest)

The comparison uses Austrian calibration data and computes RMSE
for various macroeconomic variables.
"""

import BeforeIT as Bit
using Dates, Statistics, DataFrames, CSV, JLD2
using Plots, StatsPlots

# Make CalibrationData available in Main so JLD2 can deserialize it
# (the .jld2 files store the type as Main.CalibrationData)
const CalibrationData = Bit.CalibrationData

# Include model extensions (exact MATLAB matching)
include("CANVAS_extension.jl")
include("CATS_extension.jl")
include("EUBI_extension.jl")
include("KS_extension.jl")

# =====================================================
# CONFIGURATION
# =====================================================
# Note: All model creators (create_canvas_model, create_cats_model, etc.)
# are defined in the included extension files with exact MATLAB matching

function create_standard_model(p, ic)
    w_act, w_inact = Bit.Workers(p, ic)
    firms = Bit.Firms(p, ic)
    bank = Bit.Bank(p, ic)
    cb = Bit.CentralBank(p, ic)
    gov = Bit.Government(p, ic)
    rotw = Bit.RestOfTheWorld(p, ic)
    agg = Bit.Aggregates(p, ic)
    prop = Bit.Properties(p, ic)
    data = Bit.Data(p)
    return Bit.Model(w_act, w_inact, firms, bank, cb, gov, rotw, agg, prop, data)
end

const MODEL_CREATORS = Dict(
    :standard => create_standard_model,
    :canvas => create_canvas_model,
    :cats => create_cats_model,
    :eubi => create_eubi_model,
    :ks => create_ks_model,
)

# =====================================================
# HELPER: Extract time series from model results
# =====================================================

function extract_series(results::Vector{<:Bit.AbstractModel})
    n_runs = length(results)
    T = length(results[1].data.real_gdp)

    real_gdp = zeros(n_runs, T)
    nominal_gdp = zeros(n_runs, T)
    nominal_household_consumption = zeros(n_runs, T)

    for (i, model) in enumerate(results)
        real_gdp[i, :] = model.data.real_gdp
        nominal_gdp[i, :] = model.data.nominal_gdp
        nominal_household_consumption[i, :] = model.data.nominal_household_consumption
    end

    # Compute GDP growth rate (quarter-over-quarter)
    gdp_growth = zeros(n_runs, T)
    gdp_growth[:, 1] .= 0.0  # First period has no growth rate
    for t in 2:T
        gdp_growth[:, t] = (real_gdp[:, t] .- real_gdp[:, t-1]) ./ real_gdp[:, t-1]
    end

    # Compute implicit deflator and inflation (CPI proxy)
    gdp_deflator = nominal_gdp ./ real_gdp
    inflation = zeros(n_runs, T)
    inflation[:, 1] .= 0.0
    for t in 2:T
        inflation[:, t] = (gdp_deflator[:, t] .- gdp_deflator[:, t-1]) ./ gdp_deflator[:, t-1]
    end

    return (real_gdp=real_gdp, gdp_growth=gdp_growth, inflation=inflation,
            nominal_gdp=nominal_gdp, gdp_deflator=gdp_deflator)
end

# =====================================================
# HELPER: Compute RMSE for forecasts
# =====================================================

function compute_forecast_rmse(simulated::Matrix{Float64}, actual::Vector{Float64})
    n_runs, T = size(simulated)
    horizons = min(T, length(actual))

    rmse_by_horizon = zeros(horizons)
    for h in 1:horizons
        errors = simulated[:, h] .- actual[h]
        rmse_by_horizon[h] = sqrt(mean(errors .^ 2))
    end
    return rmse_by_horizon
end

# =====================================================
# MAIN: Run forecasting comparison
# =====================================================

function run_forecasting_comparison(;
    cal = load(joinpath(@__DIR__, "../../data/at/calibration_object.jld2"))["calibration_object"],
    start_year = 2010,
    end_year = 2012,  # Reduced for quick test
    n_runs = 50,      # Reduced for quick test
    T = 12,           # Forecast horizon (quarters)
    models = [:standard, :canvas, :cats, :eubi, :ks]
)
    # Wrap calibration so DDGABM params (C6, deflator ARs) are auto-included
    dcal = DDGABMCalibration(cal)

    # Generate reference quarters (quarter-end dates)
    quarters = []
    for year in start_year:end_year
        for quarter in 1:4
            month = quarter * 3  # 3, 6, 9, 12
            # Last day of quarter: March 31, June 30, Sept 30, Dec 31
            push!(quarters, Dates.lastdayofmonth(DateTime(year, month, 1)))
        end
    end

    println("Running forecasting comparison:")
    println("  Quarters: $(length(quarters))")
    println("  Models: $models")
    println("  Runs per model: $n_runs")

    # Results storage
    all_results = Dict()

    for model_type in models
        println("\n=== Model: $model_type ===")
        model_results = []

        for (q_idx, quarter) in enumerate(quarters)
            print("  Quarter $q_idx/$(length(quarters)): $quarter ... ")

            try
                p, ic = Bit.get_params_and_initial_conditions(dcal, quarter; scale = 0.001)
                model = MODEL_CREATORS[model_type](p, ic)
                results = Bit.ensemblerun(model, T, n_runs)
                series = extract_series(results)
                push!(model_results, (quarter=quarter, series=series))
                println("done")
            catch e
                println("FAILED: $e")
            end
        end

        all_results[model_type] = model_results
    end

    return all_results, quarters
end

# =====================================================
# PLOTTING (Figure 2 style: GDP Growth and Inflation)
# =====================================================
# Paper format (from DDGABM/figs/ts_figure.m):
# - 2×4 grid (2 rows × 4 models)
# - Row 1: GDP growth rate (annualized %)
# - Row 2: Inflation rate (annualized %)
# - Black line = simulation mean
# - Gray band = ±1 std
# - Dashed line = actual data
#
# Growth rates are ANNUALIZED: 400 × quarterly_rate

function plot_forecasting_comparison(all_results;
                                     model_order = [:canvas, :cats, :eubi, :ks])
    # Filter to available models in specified order
    models = [m for m in model_order if haskey(all_results, m) && !isempty(all_results[m])]
    n_models = length(models)

    if n_models == 0
        error("No valid model results to plot")
    end

    # Create 2×n_models grid: top row GDP growth, bottom row inflation
    gdp_plots = []
    infl_plots = []

    for model in models
        results = all_results[model]

        # Aggregate all quarters' results for this model
        # Take the last quarter's results (most recent forecast)
        r = results[end]

        # GDP Growth: convert to ANNUALIZED % (multiply by 400)
        growth_mean = 400 .* mean(r.series.gdp_growth, dims=1)[:]
        growth_std = 400 .* std(r.series.gdp_growth, dims=1)[:]

        # Inflation: convert to ANNUALIZED %
        infl_mean = 400 .* mean(r.series.inflation, dims=1)[:]
        infl_std = 400 .* std(r.series.inflation, dims=1)[:]

        T = length(growth_mean)

        # GDP Growth subplot
        p_growth = plot(title="$(uppercase(string(model)))",
                       xlabel="", ylabel="Growth (%)",
                       ylims=(-8, 12), legend=false)

        # Plot mean with std ribbon (gray band)
        plot!(p_growth, 1:T, growth_mean,
              ribbon=growth_std,
              fillcolor=:gray, fillalpha=0.3,
              linecolor=:black, linewidth=1.5,
              label="")

        # Add zero line
        hline!(p_growth, [0], color=:gray, linestyle=:dash, alpha=0.5)

        push!(gdp_plots, p_growth)

        # Inflation subplot
        p_inflation = plot(title="",
                          xlabel="Quarters", ylabel="Inflation (%)",
                          ylims=(-4, 8), legend=false)

        # Plot mean with std ribbon
        plot!(p_inflation, 1:T, infl_mean,
              ribbon=infl_std,
              fillcolor=:gray, fillalpha=0.3,
              linecolor=:black, linewidth=1.5,
              label="")

        # Add zero line
        hline!(p_inflation, [0], color=:gray, linestyle=:dash, alpha=0.5)

        push!(infl_plots, p_inflation)
    end

    # Combine into 2×n_models figure
    # Row 1: GDP growth for all models
    # Row 2: Inflation for all models
    all_plots = vcat(gdp_plots, infl_plots)
    combined = plot(all_plots..., layout=(2, n_models), size=(300*n_models, 500))

    return combined
end

# =====================================================
# RUN
# =====================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting forecasting comparison...")
    println("(Using reduced settings for quick verification)")

    all_results, quarters = run_forecasting_comparison(
        start_year = 2010,
        end_year = 2011,  # Just 2 years for quick test
        n_runs = 20,      # Reduced
        T = 12,           # 12 quarters ahead (3 years)
        models = [:canvas, :cats, :eubi, :ks]  # All 4 paper models
    )

    p = plot_forecasting_comparison(all_results)
    savefig(p, "figure2_forecasting.png")
    println("\nSaved figure2_forecasting.png")
end
