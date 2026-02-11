"""
=====================================================================
GROWTH-RATE AR(1) EXTENSION
=====================================================================

Implementation of AR(1) on growth rates (instead of log-levels) via method overloading.
This follows the CANVAS_extension.jl pattern for type-based dispatch.

## Standard AR(1) on log-levels (default BeforeIT):
    log(Y_t) = alpha * log(Y_{t-1}) + beta + epsilon
    => Y_t = exp(alpha * log(Y_{t-1}) + beta + epsilon)

## Growth-rate AR(1) (this extension):
    g_t = alpha * g_{t-1} + beta + epsilon
    where g_t = (Y_t - Y_{t-1}) / Y_{t-1}
    => Y_t = Y_{t-1} * (1 + g_t)

The key insight: extended structs store only the lagged growth rate (not full series).
Levels always come from actual model values, ensuring correct scale.

## Usage:
    include("examples/GrowthRateAR1_extension.jl")
    model_gr = create_model(p, ic)
    Bit.run!(model_gr, 12)
"""

import BeforeIT as Bit
using Statistics, LinearAlgebra

# =====================================================
# ABSTRACT TYPES FOR DISPATCH
# =====================================================

abstract type AbstractRestOfTheWorldGR <: Bit.AbstractRestOfTheWorld end
abstract type AbstractGovernmentGR <: Bit.AbstractGovernment end

# =====================================================
# EXTENDED STRUCTS WITH LAGGED GROWTH RATES
# =====================================================

"""
Extended RestOfTheWorld with lagged growth rates for AR(1) dynamics.
Levels come from actual model values (rotw.Y_EA, etc.), ensuring correct scale.
"""
Bit.@object mutable struct RestOfTheWorldGR(Bit.RestOfTheWorld) <: AbstractRestOfTheWorldGR
    # Growth-rate AR(1) parameters (estimated on growth rates, not log-levels)
    alpha_Y_EA_gr::Bit.typeFloat
    beta_Y_EA_gr::Bit.typeFloat
    sigma_Y_EA_gr::Bit.typeFloat
    alpha_E_gr::Bit.typeFloat
    beta_E_gr::Bit.typeFloat
    sigma_E_gr::Bit.typeFloat
    alpha_I_gr::Bit.typeFloat
    beta_I_gr::Bit.typeFloat
    sigma_I_gr::Bit.typeFloat
    # Lagged growth rates for AR(1) dynamics (levels come from actual model values)
    g_prev_Y_EA::Bit.typeFloat
    g_prev_C_E::Bit.typeFloat
    g_prev_Y_I::Bit.typeFloat
end

"""
Extended Government with lagged growth rate for AR(1) dynamics.
"""
Bit.@object mutable struct GovernmentGR(Bit.Government) <: AbstractGovernmentGR
    # Growth-rate AR(1) parameters
    alpha_G_gr::Bit.typeFloat
    beta_G_gr::Bit.typeFloat
    sigma_G_gr::Bit.typeFloat
    # Lagged growth rate (level comes from actual model value)
    g_prev_C_G::Bit.typeFloat
end

# =====================================================
# HELPER: ESTIMATE AR(1) ON GROWTH RATES
# =====================================================

"""
    estimate_gr_ar1(series)

Estimate AR(1) parameters on growth rates of the series.

Returns (alpha, beta, sigma) where:
    g_t = alpha * g_{t-1} + beta + epsilon
    epsilon ~ N(0, sigma^2)
"""
function estimate_gr_ar1(series::Vector)
    if length(series) < 3
        return 0.0, 0.0, 0.0
    end

    # Compute growth rates
    growth_rates = diff(series) ./ series[1:end-1]

    if length(growth_rates) < 2
        return 0.0, mean(growth_rates), std(growth_rates)
    end

    # Estimate AR(1) on growth rates
    alpha, beta, sigma, _ = Bit.estimate_for_calibration_script(growth_rates)

    return alpha, beta, sigma
end

"""
    last_growth_rate(series)

Compute the last growth rate from a historical series.
"""
function last_growth_rate(series::Vector)
    if length(series) < 2
        return 0.0
    end
    return (series[end] - series[end-1]) / series[end-1]
end

"""
    gr_next_value(current_value, g_prev, alpha, beta, sigma)

Compute next growth rate using AR(1) and return (new_value, new_growth_rate).

Uses the actual model value for level (not historical series), ensuring correct scale.
"""
function gr_next_value(current_value::Real, g_prev::Real, alpha::Real, beta::Real, sigma::Real)
    # AR(1) on growth rate
    epsilon = randn() * sigma
    g_new = alpha * g_prev + beta + epsilon

    # Clamp growth rate to reasonable bounds
    g_new = clamp(g_new, -0.3, 0.3)

    # Compute new level from ACTUAL model value (not historical series!)
    Y_new = current_value * (1 + g_new)

    return Y_new, g_new
end

# =====================================================
# OVERRIDE: growth_inflation_EA (Y_EA, gamma_EA, pi_EA)
# =====================================================

function Bit.growth_inflation_EA(rotw::AbstractRestOfTheWorldGR, _model)
    # Compute Y_EA using growth-rate AR(1) with ACTUAL model value
    Y_EA_new, g_new = gr_next_value(
        rotw.Y_EA,           # Use actual model value (correct scale!)
        rotw.g_prev_Y_EA,    # Lagged growth rate
        rotw.alpha_Y_EA_gr,
        rotw.beta_Y_EA_gr,
        rotw.sigma_Y_EA_gr
    )

    # Update lagged growth rate for next iteration
    rotw.g_prev_Y_EA = g_new
    gamma_EA = g_new

    # Inflation pi_EA uses standard log-level AR(1) (unchanged from base)
    epsilon_pi_EA = randn() * rotw.sigma_pi_EA
    pi_EA = exp(rotw.alpha_pi_EA * log(1 + rotw.pi_EA) + rotw.beta_pi_EA + epsilon_pi_EA) - 1

    return Y_EA_new, gamma_EA, pi_EA
end

# =====================================================
# OVERRIDE: gov_expenditure (C_G, C_d_j)
# =====================================================

function Bit.gov_expenditure(gov::AbstractGovernmentGR, model)
    # Unpack non-government arguments
    c_G_g = model.prop.c_G_g
    P_bar_g = model.agg.P_bar_g
    pi_e = model.agg.pi_e

    # Compute C_G using growth-rate AR(1) with ACTUAL model value
    C_G_new, g_new = gr_next_value(
        gov.C_G,             # Use actual model value (correct scale!)
        gov.g_prev_C_G,      # Lagged growth rate
        gov.alpha_G_gr,
        gov.beta_G_gr,
        gov.sigma_G_gr
    )

    # Update lagged growth rate for next iteration
    gov.g_prev_C_G = g_new

    # Compute local government consumptions (same as base)
    J = size(gov.C_d_j, 1)
    C_d_j = C_G_new ./ J .* ones(J) .* sum(c_G_g .* P_bar_g) .* (1 + pi_e)

    return C_G_new, C_d_j
end

# =====================================================
# OVERRIDE: rotw_import_export (C_E, Y_I, ...)
# =====================================================

function Bit.rotw_import_export(rotw::AbstractRestOfTheWorldGR, model)
    # Unpack model arguments
    c_E_g = model.prop.c_E_g
    c_I_g = model.prop.c_I_g
    P_bar_g = model.agg.P_bar_g
    pi_e = model.agg.pi_e

    L = size(rotw.C_d_l, 1)

    # Compute C_E using growth-rate AR(1) with ACTUAL model value
    C_E_new, g_new_E = gr_next_value(
        rotw.C_E,            # Use actual model value (correct scale!)
        rotw.g_prev_C_E,     # Lagged growth rate
        rotw.alpha_E_gr,
        rotw.beta_E_gr,
        rotw.sigma_E_gr
    )

    # Update lagged growth rate for next iteration
    rotw.g_prev_C_E = g_new_E

    # Compute demand for export
    C_d_l = C_E_new ./ L .* ones(L) .* sum(c_E_g .* P_bar_g) .* (1 + pi_e)

    # Compute Y_I using growth-rate AR(1) with ACTUAL model value
    Y_I_new, g_new_I = gr_next_value(
        rotw.Y_I,            # Use actual model value (correct scale!)
        rotw.g_prev_Y_I,     # Lagged growth rate
        rotw.alpha_I_gr,
        rotw.beta_I_gr,
        rotw.sigma_I_gr
    )

    # Update lagged growth rate for next iteration
    rotw.g_prev_Y_I = g_new_I

    # Compute supply of imports (same as base)
    Y_m = c_I_g * Y_I_new
    P_m = P_bar_g * (1 + pi_e)

    return C_E_new, Y_I_new, C_d_l, Y_m, P_m
end

# =====================================================
# FACTORY: Create Growth-Rate AR(1) Model
# =====================================================

"""
    create_model(p, ic)

Create a model using growth-rate AR(1) for exogenous variables.

The standard BeforeIT model uses log-level AR(1):
    log(Y_t) = alpha * log(Y_{t-1}) + beta + epsilon

This model uses growth-rate AR(1):
    g_t = alpha * g_{t-1} + beta + epsilon
    Y_t = Y_{t-1} * (1 + g_t)

AR(1) parameters are re-estimated on growth rates ONCE at model creation.

This function is the standard factory interface for extensions - when `save_all_simulations`
is called with an extension file, it `include`s that file and calls `create_model(p, ic)`.
"""
function create_model(p, ic)
    T_prime = Int(p["T_prime"])

    # =====================================================
    # 1. Initialize standard agents FIRST (to get correct scales)
    # =====================================================

    w_act, w_inact = Bit.Workers(p, ic)
    firms = Bit.Firms(p, ic)
    bank = Bit.Bank(p, ic)
    cb = Bit.CentralBank(p, ic)
    agg = Bit.Aggregates(p, ic)
    prop = Bit.Properties(p, ic)
    data = Bit.Data(p)

    # Initialize standard RestOfTheWorld and Government to get actual model values
    rotw_std = Bit.RestOfTheWorld(p, ic)
    gov_std = Bit.Government(p, ic)

    # =====================================================
    # 2. Load historical series and estimate AR(1) on growth rates
    # =====================================================

    # Y_EA series - use Y_EA_series from IC if available, otherwise construct from Y
    Y_EA_series = if haskey(ic, "Y_EA_series")
        Vector{Bit.typeFloat}(vec(ic["Y_EA_series"]))
    else
        # Fallback: use domestic Y as proxy (less accurate)
        Vector{Bit.typeFloat}(vec(ic["Y"]))
    end

    # C_G, C_E, Y_I series
    C_G_series = Vector{Bit.typeFloat}(vec(ic["C_G"]))[1:T_prime]
    C_E_series = Vector{Bit.typeFloat}(vec(ic["C_E"]))[1:T_prime]
    Y_I_series = Vector{Bit.typeFloat}(vec(ic["Y_I"]))[1:T_prime]

    # Estimate AR(1) parameters on growth rates (scale-invariant)
    alpha_Y_EA_gr, beta_Y_EA_gr, sigma_Y_EA_gr = estimate_gr_ar1(Y_EA_series)
    alpha_G_gr, beta_G_gr, sigma_G_gr = estimate_gr_ar1(C_G_series)
    alpha_E_gr, beta_E_gr, sigma_E_gr = estimate_gr_ar1(C_E_series)
    alpha_I_gr, beta_I_gr, sigma_I_gr = estimate_gr_ar1(Y_I_series)

    # =====================================================
    # 3. RESCALE series to match actual model values
    # =====================================================
    # The historical series (e.g., Y_EA_series from calibration) may have a different
    # scale than the actual model values (e.g., rotw_std.Y_EA). This can happen because
    # Y_EA_series is normalized to domestic output scale during calibration, while
    # Y_EA is the raw Euro Area GDP.
    #
    # Rescaling ensures series[end] matches the actual model starting value.
    # AR(1) parameters are unaffected (computed on growth rates, which are scale-invariant).

    Y_EA_scale = rotw_std.Y_EA / Y_EA_series[end]
    Y_EA_series = Y_EA_series .* Y_EA_scale

    C_G_scale = gov_std.C_G / C_G_series[end]
    C_G_series = C_G_series .* C_G_scale

    C_E_scale = rotw_std.C_E / C_E_series[end]
    C_E_series = C_E_series .* C_E_scale

    Y_I_scale = rotw_std.Y_I / Y_I_series[end]
    Y_I_series = Y_I_series .* Y_I_scale

    # Compute lagged growth rates from rescaled series
    g_prev_Y_EA = last_growth_rate(Y_EA_series)
    g_prev_C_G = last_growth_rate(C_G_series)
    g_prev_C_E = last_growth_rate(C_E_series)
    g_prev_Y_I = last_growth_rate(Y_I_series)

    # =====================================================
    # 4. Initialize extended agents with growth-rate params
    # =====================================================

    # Create extended RestOfTheWorld with growth-rate AR(1)
    rotw = RestOfTheWorldGR(
        # Base RestOfTheWorld fields
        rotw_std.alpha_E,
        rotw_std.beta_E,
        rotw_std.sigma_E,
        rotw_std.alpha_I,
        rotw_std.beta_I,
        rotw_std.sigma_I,
        rotw_std.Y_EA,
        rotw_std.gamma_EA,
        rotw_std.pi_EA,
        rotw_std.alpha_pi_EA,
        rotw_std.beta_pi_EA,
        rotw_std.sigma_pi_EA,
        rotw_std.alpha_Y_EA,
        rotw_std.beta_Y_EA,
        rotw_std.sigma_Y_EA,
        rotw_std.D_RoW,
        rotw_std.Y_I,
        rotw_std.C_E,
        copy(rotw_std.C_d_l),
        rotw_std.C_l,
        copy(rotw_std.Y_m),
        copy(rotw_std.Q_m),
        copy(rotw_std.Q_d_m),
        copy(rotw_std.P_m),
        rotw_std.P_l,
        # Growth-rate AR(1) parameters
        alpha_Y_EA_gr,
        beta_Y_EA_gr,
        sigma_Y_EA_gr,
        alpha_E_gr,
        beta_E_gr,
        sigma_E_gr,
        alpha_I_gr,
        beta_I_gr,
        sigma_I_gr,
        # Lagged growth rates (levels come from actual model values)
        g_prev_Y_EA,
        g_prev_C_E,
        g_prev_Y_I
    )

    # Create extended Government with growth-rate AR(1)
    gov = GovernmentGR(
        # Base Government fields
        gov_std.alpha_G,
        gov_std.beta_G,
        gov_std.sigma_G,
        gov_std.Y_G,
        gov_std.C_G,
        gov_std.L_G,
        gov_std.sb_inact,
        gov_std.sb_other,
        copy(gov_std.C_d_j),
        gov_std.C_j,
        gov_std.P_j,
        # Growth-rate AR(1) parameters
        alpha_G_gr,
        beta_G_gr,
        sigma_G_gr,
        # Lagged growth rate (level comes from actual model value)
        g_prev_C_G
    )

    return Bit.Model(w_act, w_inact, firms, bank, cb, gov, rotw, agg, prop, data)
end

# =====================================================
# DIAGNOSTIC FUNCTION
# =====================================================

"""
    compare_gr_params(p, ic)

Compare log-level AR(1) parameters with growth-rate AR(1) parameters.
Also shows scale factors between historical series and actual model values.
Useful for diagnostics and understanding the difference between methods.
"""
function compare_gr_params(p, ic)
    T_prime = Int(p["T_prime"])

    # Initialize standard agents to get actual model values
    rotw_std = Bit.RestOfTheWorld(p, ic)
    gov_std = Bit.Government(p, ic)

    println("=" ^ 60)
    println("AR(1) Parameter Comparison: Log-Level vs Growth-Rate")
    println("=" ^ 60)

    # Y_EA
    Y_EA_series = haskey(ic, "Y_EA_series") ? vec(ic["Y_EA_series"]) : vec(ic["Y"])
    alpha_log, beta_log, _ = Bit.estimate(log.(Y_EA_series))
    alpha_gr, beta_gr, sigma_gr = estimate_gr_ar1(Y_EA_series)
    Y_EA_scale = rotw_std.Y_EA / Y_EA_series[end]
    println("\nY_EA (Euro Area GDP):")
    println("  Log-level AR(1): alpha=$(round(alpha_log, digits=4)), beta=$(round(beta_log, digits=4))")
    println("  Growth-rate AR(1): alpha=$(round(alpha_gr, digits=4)), beta=$(round(beta_gr, digits=4)), sigma=$(round(sigma_gr, digits=4))")
    println("  Scale factor: $(round(Y_EA_scale, digits=4)) (actual=$(round(rotw_std.Y_EA, digits=0)), series_end=$(round(Y_EA_series[end], digits=0)))")

    # C_G
    C_G_series = vec(ic["C_G"])[1:T_prime]
    alpha_log, beta_log, _ = Bit.estimate(log.(C_G_series))
    alpha_gr, beta_gr, sigma_gr = estimate_gr_ar1(C_G_series)
    C_G_scale = gov_std.C_G / C_G_series[end]
    println("\nC_G (Government Consumption):")
    println("  Log-level AR(1): alpha=$(round(alpha_log, digits=4)), beta=$(round(beta_log, digits=4))")
    println("  Growth-rate AR(1): alpha=$(round(alpha_gr, digits=4)), beta=$(round(beta_gr, digits=4)), sigma=$(round(sigma_gr, digits=4))")
    println("  Scale factor: $(round(C_G_scale, digits=4)) (actual=$(round(gov_std.C_G, digits=0)), series_end=$(round(C_G_series[end], digits=0)))")

    # C_E
    C_E_series = vec(ic["C_E"])[1:T_prime]
    alpha_log, beta_log, _ = Bit.estimate(log.(C_E_series))
    alpha_gr, beta_gr, sigma_gr = estimate_gr_ar1(C_E_series)
    C_E_scale = rotw_std.C_E / C_E_series[end]
    println("\nC_E (Exports):")
    println("  Log-level AR(1): alpha=$(round(alpha_log, digits=4)), beta=$(round(beta_log, digits=4))")
    println("  Growth-rate AR(1): alpha=$(round(alpha_gr, digits=4)), beta=$(round(beta_gr, digits=4)), sigma=$(round(sigma_gr, digits=4))")
    println("  Scale factor: $(round(C_E_scale, digits=4)) (actual=$(round(rotw_std.C_E, digits=0)), series_end=$(round(C_E_series[end], digits=0)))")

    # Y_I
    Y_I_series = vec(ic["Y_I"])[1:T_prime]
    alpha_log, beta_log, _ = Bit.estimate(log.(Y_I_series))
    alpha_gr, beta_gr, sigma_gr = estimate_gr_ar1(Y_I_series)
    Y_I_scale = rotw_std.Y_I / Y_I_series[end]
    println("\nY_I (Imports):")
    println("  Log-level AR(1): alpha=$(round(alpha_log, digits=4)), beta=$(round(beta_log, digits=4))")
    println("  Growth-rate AR(1): alpha=$(round(alpha_gr, digits=4)), beta=$(round(beta_gr, digits=4)), sigma=$(round(sigma_gr, digits=4))")
    println("  Scale factor: $(round(Y_I_scale, digits=4)) (actual=$(round(rotw_std.Y_I, digits=0)), series_end=$(round(Y_I_series[end], digits=0)))")

    println("\n" * "=" ^ 60)
end

# =====================================================
# TEST
# =====================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using Plots, JLD2

    T = 12

    # Use precomputed Austria 2010Q1 data from BeforeIT package
    p = Bit.AUSTRIA2010Q1.parameters
    ic = Bit.AUSTRIA2010Q1.initial_conditions

    println("="^60)
    println("Growth-Rate AR(1) Extension Test")
    println("="^60)

    # Show parameter comparison
    compare_gr_params(p, ic)

    # Create both models
    println("\nCreating standard model (log-level AR1)...")
    model_std = Bit.Model(p, ic)

    println("Creating growth-rate AR1 model...")
    model_gr = create_model(p, ic)

    # Run ensemble simulations
    n_sims = 8
    println("\nRunning $(n_sims) simulations for each model (T=$T quarters)...")

    results_std = Bit.ensemblerun(model_std, T, n_sims)
    results_gr = Bit.ensemblerun(model_gr, T, n_sims)

    # Extract GDP
    gdp_std = [r.data.real_gdp for r in results_std]
    gdp_gr = [r.data.real_gdp for r in results_gr]

    # Compute mean and std
    gdp_std_mean = mean(gdp_std)
    gdp_std_std = std(gdp_std)
    gdp_gr_mean = mean(gdp_gr)
    gdp_gr_std = std(gdp_gr)

    println("\nResults:")
    println("  Standard (log-level AR1) - Final GDP: $(round(gdp_std_mean[end], digits=2)) +/- $(round(gdp_std_std[end], digits=2))")
    println("  Growth-rate AR1 - Final GDP: $(round(gdp_gr_mean[end], digits=2)) +/- $(round(gdp_gr_std[end], digits=2))")

    # Plot comparison
    p1 = plot(1:T, gdp_std_mean, ribbon=2*gdp_std_std, label="Log-level AR(1)",
              xlabel="Quarter", ylabel="Real GDP", title="Model Comparison",
              fillalpha=0.2, linewidth=2)
    plot!(p1, 1:T, gdp_gr_mean, ribbon=2*gdp_gr_std, label="Growth-rate AR(1)",
          fillalpha=0.2, linewidth=2)

    display(p1)
    println("\n" * "="^60)
    println("Test completed successfully!")
    println("="^60)
end
