"""
=====================================================================
FIGURE 3: SHOCK IMPULSE RESPONSE FUNCTIONS
=====================================================================

Recreates Figure 3 from Dawid et al. (2024) "Implications of Behavioral
Rules in Agent-Based Macroeconomics"

This script computes IRFs for three shocks across four pricing models:
1. Productivity shock: +5% permanent change in α_bar_i
2. Government spending shock: +10% temporary (4 quarters)
3. Import price shock: +10% temporary (4 quarters)

Models compared:
- CANVAS: Demand-pull with cost-push
- CATS: Demand-pull without cost-push
- EUBI: Market share + competitor prices
- KS: Pure market share
"""

import BeforeIT as Bit
using Dates, Statistics, JLD2
using Plots, StatsPlots

# Include the FULL model extensions (all use markup-based pricing P=(1+μ)×AC_e)
include("CANVAS_full_extension.jl")  # Full CANVAS: δ_rec=1, δ_AC=0 (cost pass-through)
include("CATS_full_extension.jl")    # Full CATS: δ_rec=1, δ_AC=1 (cost absorption)
include("EUBI_full_extension.jl")    # Full EUBI: δ_rec=0, δ_MS=1, δ_P=1, δ_AC=1
include("KS_full_extension.jl")      # Full KS: δ_rec=0, δ_MS=0.04, δ_AC=0

# =====================================================
# CORRECTED SHOCK IMPLEMENTATIONS
# =====================================================
# The built-in shocks have bugs:
# 1. ProductivityShock applies every step (compounds!)
# 2. ImportPriceShock is overwritten by rotw_import_export
# 3. GovernmentSpendingShock works through AR process
#
# These corrected versions apply shocks properly using model.agg.t checks
# (not mutable state, which doesn't work with ensemblerun's shared shock)

"""
Corrected ProductivityShock - applies ONCE at t=1 only
"""
struct ProductivityShockOnce <: Bit.AbstractShock
    multiplier::Float64
end

function (s::ProductivityShockOnce)(model::Bit.Model)
    if model.agg.t == 1
        model.firms.alpha_bar_i .= model.firms.alpha_bar_i .* s.multiplier
    end
end

"""
Corrected GovernmentSpendingShock - directly multiplies C_G at t=1
MATLAB (simulate_abm.m line 307-309): C_G(T_prime+1:end) = 1.1 .* C_G(T_prime+1:end)
We multiply C_G once at t=1, which then persists through the AR(1) dynamics.
"""
struct GovSpendingShockPermanent <: Bit.AbstractShock
    multiplier::Float64
end

function (s::GovSpendingShockPermanent)(model::Bit.Model)
    if model.agg.t == 1
        # EXACT MATLAB: directly multiply C_G by the multiplier
        # The AR process will propagate this level change forward
        model.gov.C_G = model.gov.C_G * s.multiplier
    end
end

# NOTE: Import price shock now uses built-in Bit.ImportPriceShock
# which properly shocks P_I (exogenous import price index) matching DDGABM line 311:
#   P_I(T_prime+1:end) = 1.1 .* P_I(T_prime+1:end)
# P_I then propagates to P_m = P_I each period, affecting import prices without feedback

# =====================================================
# CANVAS MODEL: Using Full CANVAS from CANVAS_full_extension.jl
# =====================================================
# Full CANVAS uses markup-based pricing: P = (1+μ) × AC_e
# This correctly transmits productivity shocks:
#   α↑ → AC_e↓ → P↓ → demand↑ → GDP↑
#
# Note: create_full_canvas_model is defined in CANVAS_full_extension.jl

# =====================================================
# MODEL CREATORS
# =====================================================

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
    :canvas => create_full_canvas_model,  # Full CANVAS with P=(1+μ)×AC_e
    :cats => create_full_cats_model,      # Full CATS with P=(1+μ)×AC_e
    :eubi => create_full_eubi_model,      # Full EUBI with P=(1+μ)×AC_e
    :ks => create_full_ks_model,          # Full KS with P=(1+μ)×AC_e
)

# =====================================================
# SHOCK DEFINITIONS (using new shocks from src/shocks/shocks.jl)
# =====================================================

# Paper shocks are PERMANENT for entire simulation horizon (from simulate_abm.m):
# - Scenario 2: C_G(T_prime+1:end)=1.1.*C_G(T_prime+1:end)  -> permanent +10%
# - Scenario 3: P_I(T_prime+1:end)=1.1.*P_I(T_prime+1:end)  -> permanent +10%
# - Scenario 4: alpha_bar_i=1.1*alpha_bar_i                  -> permanent +10%
#
# Shock implementations:
# - ProductivityShockOnce: applies alpha_bar_i multiplier once at t=1
# - GovSpendingShockPermanent: shifts AR intercept for permanent level change
# - Bit.ImportPriceShock: applies P_I multiplier once at t=1 (now matches DDGABM exactly)

# Shock instances (immutable structs, safe to reuse)
const SHOCKS = Dict(
    :productivity => ProductivityShockOnce(1.10),       # +10% alpha_bar_i at t=1
    :government   => GovSpendingShockPermanent(1.10),   # +10% via AR intercept shift
    :import_price => Bit.ImportPriceShock(1.10, 100),   # +10% P_I at t=1 (DDGABM line 311)
)

# =====================================================
# HELPER: Extract GDP series from model results
# =====================================================

function extract_series(results::Vector{<:Bit.AbstractModel})
    n_runs = length(results)
    T = length(results[1].data.real_gdp)

    real_gdp = zeros(n_runs, T)
    nominal_hh_cons = zeros(n_runs, T)
    real_hh_cons = zeros(n_runs, T)

    for (i, model) in enumerate(results)
        real_gdp[i, :] = model.data.real_gdp
        nominal_hh_cons[i, :] = model.data.nominal_household_consumption
        real_hh_cons[i, :] = model.data.real_household_consumption
    end

    # CPI = household consumption deflator (paper uses this, not GDP deflator)
    cpi = nominal_hh_cons ./ real_hh_cons

    return real_gdp, cpi
end

# =====================================================
# COMPUTE IRF: Ratio format (shocked / baseline)
# =====================================================
# Paper format (from DDGABM/figs/shock_figure.m):
# - IRF shown as ratio: mean(shocked) / mean(baseline)
# - Ratio = 1 means no effect, >1 means positive response

function compute_irf(baseline_gdp::Matrix{Float64}, shocked_gdp::Matrix{Float64})
    # Mean across runs
    baseline_mean = mean(baseline_gdp, dims=1)[:]
    shocked_mean = mean(shocked_gdp, dims=1)[:]

    # Standard error (for confidence bands)
    n_runs = size(baseline_gdp, 1)
    baseline_se = std(baseline_gdp, dims=1)[:] ./ sqrt(n_runs)
    shocked_se = std(shocked_gdp, dims=1)[:] ./ sqrt(n_runs)

    # IRF as RATIO (paper format): shocked / baseline
    irf = shocked_mean ./ baseline_mean

    # Propagate uncertainty for ratio (delta method approximation)
    # se(a/b) ≈ (a/b) * sqrt((se_a/a)^2 + (se_b/b)^2)
    irf_se = irf .* sqrt.((shocked_se ./ shocked_mean).^2 .+ (baseline_se ./ baseline_mean).^2)

    return irf, irf_se
end

# =====================================================
# MAIN: Compute IRFs for all models and shocks
# =====================================================

function compute_all_irfs(;
    cal = load(joinpath(@__DIR__, "../../data/at/calibration_object.jld2"))["calibration_object"],
    calibration_date = DateTime(2010, 03, 31),
    T = 20,  # 20 quarters (5 years)
    n_runs = 50,
    models = [:canvas, :cats, :eubi, :ks],
    shocks = [:productivity, :government, :import_price]
)
    println("Computing IRFs:")
    println("  Models: $models")
    println("  Shocks: $shocks")
    println("  Runs: $n_runs")
    println("  Horizon: $T quarters")

    p, ic = Bit.get_params_and_initial_conditions(cal, calibration_date; scale = 0.001)

    all_irfs = Dict()

    for model_type in models
        println("\n=== Model: $model_type ===")
        model_irfs = Dict()

        # Run baseline (no shock)
        print("  Baseline ... ")
        model = MODEL_CREATORS[model_type](p, ic)

        # Use custom ensemblerun for EUBI (with MATLAB-style gamma AR)
        if model_type == :eubi
            baseline_results = ensemblerun_eubi(model, T, n_runs)
        else
            baseline_results = Bit.ensemblerun(model, T, n_runs)
        end
        baseline_gdp, baseline_cpi = extract_series(baseline_results)
        println("done")

        for shock_type in shocks
            print("  Shock: $shock_type ... ")

            # Create fresh model for shocked run
            model_shocked = MODEL_CREATORS[model_type](p, ic)
            shock = SHOCKS[shock_type]

            # Use custom ensemblerun for EUBI (with MATLAB-style gamma AR)
            if model_type == :eubi
                shocked_results = ensemblerun_eubi(model_shocked, T, n_runs; shock=shock)
            else
                shocked_results = Bit.ensemblerun(model_shocked, T, n_runs; shock=shock)
            end
            shocked_gdp, shocked_cpi = extract_series(shocked_results)

            # Compute IRFs (no SE needed - paper doesn't show error bands)
            irf_gdp, _ = compute_irf(baseline_gdp, shocked_gdp)
            irf_cpi, _ = compute_irf(baseline_cpi, shocked_cpi)

            model_irfs[shock_type] = (
                gdp = irf_gdp,
                cpi = irf_cpi
            )
            println("done")
        end

        all_irfs[model_type] = model_irfs
    end

    return all_irfs
end

# =====================================================
# PLOTTING (Figure 3 format: 2×3 grid)
# =====================================================
# Paper format (from DDGABM/figs/shock_figure.m):
# - 2×3 subplot grid
# - Top row: GDP IRFs for 3 shocks
# - Bottom row: CPI IRFs for 3 shocks
# - 4 lines per plot (CANVAS, CATS, EUBI, KS)
# - IRF as ratio (horizontal line at 1 = no effect)

function plot_irfs(all_irfs; variable=:gdp, title_prefix="GDP",
                   shock_order = [:government, :import_price, :productivity],  # Paper order
                   model_order = [:canvas, :cats, :eubi, :ks])
    # Filter to available shocks and models
    shocks = [s for s in shock_order if any(haskey(all_irfs[m], s) for m in keys(all_irfs) if haskey(all_irfs, m))]
    models = [m for m in model_order if haskey(all_irfs, m)]

    plots = []

    # Line styles to match paper exactly (all black, different line styles)
    styles = Dict(:canvas => :solid, :cats => :dash, :eubi => :dashdot, :ks => :dot)
    widths = Dict(:canvas => 1.0, :cats => 1.0, :eubi => 1.0, :ks => 1.5)

    # Paper shock names (from shock_figure.m)
    shock_names = Dict(
        :government => "Demand Shock",      # scenario 2
        :import_price => "Cost-Push Shock", # scenario 3
        :productivity => "Technology Shock" # scenario 4
    )

    for (s_idx, shock) in enumerate(shocks)
        shock_label = get(shock_names, shock, string(shock))

        # Only show legend in first subplot (like paper)
        show_legend = (s_idx == 1)

        p = plot(title="$shock_label\n$title_prefix",
                 xlabel="Quarters", ylabel="",
                 legend = show_legend ? :topleft : false,
                 grid=true,
                 xlims=(1, 12))

        for model in models
            if haskey(all_irfs[model], shock)
                data = all_irfs[model][shock]
                irf = getfield(data, variable)  # Direct access (no longer nested)

                style = get(styles, model, :solid)
                width = get(widths, model, 1.0)

                # NO ribbon/error bands - paper shows clean lines only
                plot!(p, 1:length(irf), irf,
                      label=uppercase(string(model)),
                      color=:black, linestyle=style, linewidth=width)
            end
        end

        push!(plots, p)
    end

    return plots
end

# =====================================================
# RUN
# =====================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Computing shock IRFs...")
    println("(Figure 3 from Dawid et al. 2024)")

    all_irfs = compute_all_irfs(
        T = 12,            # 3 years (12 quarters)
        n_runs = 100,      # 100 runs for smooth results
        models = [:canvas, :cats, :eubi, :ks],  # All 4 paper models
        shocks = [:productivity, :government, :import_price]  # All 3 shocks
    )

    # Plot GDP IRFs (top row)
    gdp_plots = plot_irfs(all_irfs, variable=:gdp, title_prefix="GDP")

    # Plot CPI IRFs (bottom row) - paper uses household consumption deflator
    cpi_plots = plot_irfs(all_irfs, variable=:cpi, title_prefix="CPI")

    # Combine into 2×3 figure (paper format)
    # Row 1: GDP IRFs for 3 shocks
    # Row 2: CPI IRFs for 3 shocks
    all_plots = vcat(gdp_plots, cpi_plots)
    n_shocks = length(gdp_plots)
    combined = plot(all_plots..., layout=(2, n_shocks), size=(350*n_shocks, 500))
    savefig(combined, "figure3_shocks.png")
    println("\nSaved figure3_shocks.png")
end
