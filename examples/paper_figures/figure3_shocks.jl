"""
=====================================================================
FIGURE 3: SHOCK IMPULSE RESPONSE FUNCTIONS
=====================================================================

Recreates Figure 3 from Dawid et al. (2024) "Implications of Behavioral
Rules in Agent-Based Macroeconomics"

This script computes IRFs for three shocks across four pricing models:
1. Productivity shock: +10% permanent change in α_bar_i
2. Government spending shock: +10% permanent level shift in C_G
3. Import price shock: +10% permanent increase in P_m

Models compared:
- CANVAS: Demand-pull with cost-push
- CATS: Demand-pull without cost-push
- EUBI: Market share + competitor prices
- KS: Pure market share
"""

import BeforeIT as Bit
using Dates, Statistics, JLD2
using Plots, StatsPlots

# Make CalibrationData available in Main so JLD2 can deserialize it
# (the .jld2 files store the type as Main.CalibrationData)
const CalibrationData = Bit.CalibrationData

# Include model extensions (all use markup-based pricing P=(1+μ)×AC_e)
include("CANVAS_extension.jl")  # CANVAS: δ_rec=1, δ_AC=0 (cost pass-through)
include("CATS_extension.jl")    # CATS: δ_rec=1, δ_AC=1 (cost absorption)
include("EUBI_extension.jl")    # EUBI: δ_rec=0, δ_MS=1, δ_P=1, δ_AC=1
include("KS_extension.jl")      # KS: δ_rec=0, δ_MS=0.04, δ_AC=0

# =====================================================
# SHOCK IMPLEMENTATIONS (matching DDGABM/simulate_abm.m)
# =====================================================
# MATLAB pre-computes exogenous trajectories via AR, then applies shocks
# as permanent level shifts to the ENTIRE future trajectory:
#   Scenario 2: C_G(T_prime+1:end) = 1.1 .* C_G(T_prime+1:end)
#   Scenario 3: P_I(T_prime+1:end) = 1.1 .* P_I(T_prime+1:end)
#   Scenario 4: alpha_bar_i = 1.1 * alpha_bar_i
#
# BeforeIT.jl computes AR processes inline (not pre-computed), so we match
# the MATLAB effect through equivalent mechanisms.

"""
ProductivityShock - applies ONCE at t=1 (permanent change to α_bar_i).
MATLAB: alpha_bar_i = 1.1 * alpha_bar_i (one-time, permanent)
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
GovSpendingShockPermanent - permanent level shift in government spending.

For base Government (log-level AR): shifts intercept β_G += (1-α_G)*log(m)
For GovernmentDDGABM (growth-rate AR): multiplies C_G directly at t=1.
  MATLAB: C_G(T_prime+1:end) = 1.1 * C_G(T_prime+1:end)
  With inline AR on growth rates, multiplying the initial level is sufficient
  since the growth-rate AR naturally propagates the new level.
"""
struct GovSpendingShockPermanent <: Bit.AbstractShock
    multiplier::Float64
end

function (s::GovSpendingShockPermanent)(model::Bit.Model)
    if model.agg.t == 1
        if model.gov isa AbstractGovernmentDDGABM
            # DDGABM: multiply C_G directly (growth-rate AR preserves the level)
            model.gov.C_G = model.gov.C_G * s.multiplier
        else
            # Base: shift the log-level AR intercept for permanent level change
            model.gov.beta_G += (1 - model.gov.alpha_G) * log(s.multiplier)
            model.gov.C_G = model.gov.C_G * s.multiplier
        end
    end
end

# =====================================================
# IMPORT PRICE SHOCK
# =====================================================
# For DDGABM RotW: multiply P_I directly at t=1 (growth-rate AR preserves level)
# For base RotW: use _import_price_multiplier Ref + rotw_import_export override

"""
ImportPriceShockPermanent - permanent level shift in import prices.

For RestOfTheWorldDDGABM: multiplies P_I at t=1. The deflator AR on pi_I
naturally propagates the new price level.
MATLAB: P_I(T_prime+1:end) = 1.1 * P_I(T_prime+1:end)

For base RestOfTheWorld: uses _import_price_multiplier Ref (set externally).
"""
struct ImportPriceShockPermanent <: Bit.AbstractShock
    multiplier::Float64
end

function (s::ImportPriceShockPermanent)(model::Bit.Model)
    if model.agg.t == 1
        if model.rotw isa AbstractRestOfTheWorldDDGABM
            # DDGABM: multiply P_I directly (deflator AR preserves the level)
            model.rotw.P_I = model.rotw.P_I * s.multiplier
        end
        # For base RotW, _import_price_multiplier is set externally before ensemblerun
    end
end

const _import_price_multiplier = Ref(1.0)

const ExtensionFirms = Union{
    AbstractFirmsCANVAS, AbstractFirmsCATS,
    AbstractFirmsEUBI, AbstractFirmsKS
}

# Override for base RotW with extension firms (non-DDGABM path)
# DDGABM RotW has its own override in DDGABM_exogenous_extension.jl
function Bit.rotw_import_export(
    rotw::Bit.RestOfTheWorld,
    model::Bit.Model{<:Bit.AbstractWorkers, <:Bit.AbstractWorkers, <:ExtensionFirms,
                      <:Bit.AbstractBank, <:Bit.AbstractCentralBank, <:Bit.AbstractGovernment,
                      <:Bit.AbstractRestOfTheWorld, <:Bit.AbstractAggregates}
)
    c_E_g = model.prop.c_E_g
    c_I_g = model.prop.c_I_g
    P_bar_g = model.agg.P_bar_g
    pi_e = model.agg.pi_e
    epsilon_E, epsilon_I = model.agg.epsilon_E, model.agg.epsilon_I

    L = size(rotw.C_d_l, 1)
    C_E = exp.(rotw.alpha_E * log(rotw.C_E) + rotw.beta_E + epsilon_E)
    C_d_l = C_E ./ L .* ones(L) .* sum(c_E_g .* P_bar_g) .* (1 + pi_e)

    Y_I = exp(rotw.alpha_I * log(rotw.Y_I) + rotw.beta_I .+ epsilon_I)
    Y_m = c_I_g * Y_I
    P_m = P_bar_g * (1 + pi_e) .* _import_price_multiplier[]

    return C_E, Y_I, C_d_l, Y_m, P_m
end

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
    :canvas => create_canvas_model,
    :cats => create_cats_model,
    :eubi => create_eubi_model,
    :ks => create_ks_model,
)

# =====================================================
# SHOCK DEFINITIONS
# =====================================================

# Shock instances (immutable structs, safe to reuse)
const SHOCKS = Dict(
    :productivity => ProductivityShockOnce(1.10),       # +10% alpha_bar_i at t=1
    :government   => GovSpendingShockPermanent(1.10),   # +10% C_G (handles both base & DDGABM)
    :import_price => ImportPriceShockPermanent(1.10),   # +10% P_I (DDGABM) + _import_price_multiplier (base)
)

const IMPORT_PRICE_SHOCK_MULTIPLIER = 1.10

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

    # CPI = household consumption deflator (MATLAB shock_figure.m line 69)
    # This is P_bar_HH = sum(b_HH_g .* P_bar_g), NOT the GDP deflator
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

    # Wrap calibration in DDGABMCalibration so p/ic automatically include
    # 6x6 covariance, deflator AR params, and growth-rate AR params
    dcal = DDGABMCalibration(cal)
    p, ic = Bit.get_params_and_initial_conditions(dcal, calibration_date; scale = 0.001)
    println("  DDGABM params: C6 $(size(p["C6"])), deflator ARs added")

    all_irfs = Dict()

    for model_type in models
        println("\n=== Model: $model_type ===")
        model_irfs = Dict()

        # Run baseline (no shock, no import price multiplier)
        print("  Baseline ... ")
        _import_price_multiplier[] = 1.0
        model = MODEL_CREATORS[model_type](p, ic)

        baseline_results = Bit.ensemblerun(model, T, n_runs)
        baseline_gdp, baseline_cpi = extract_series(baseline_results)
        println("done")

        for shock_type in shocks
            print("  Shock: $shock_type ... ")

            # Set import price multiplier (only active for :import_price shock)
            _import_price_multiplier[] = (shock_type == :import_price) ? IMPORT_PRICE_SHOCK_MULTIPLIER : 1.0

            # Create fresh model for shocked run
            model_shocked = MODEL_CREATORS[model_type](p, ic)
            shock = SHOCKS[shock_type]

            shocked_results = Bit.ensemblerun(model_shocked, T, n_runs; shock=shock)
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
