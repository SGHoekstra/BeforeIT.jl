"""
Shared helpers for DDGABM pricing models (CANVAS, CATS, EUBI, KS).

Extracted from the individual model files to eliminate duplication.
Each model file includes this via:
    if !@isdefined(growth_inflation_expectations_matlab_core)
        include("shared_pricing_helpers.jl")
    end
"""

import BeforeIT as Bit
using Statistics, LinearAlgebra

# Include DDGABM exogenous extension (6x6 covariance + P_G/P_E/P_I)
if !@isdefined(GovernmentDDGABM)
    include("DDGABM_exogenous_extension.jl")
end

# =====================================================
# CORRELATED EPSILON DRAWS (MATLAB epsilon_.m)
# =====================================================

"""
    draw_correlated_epsilons(u_gamma, u_pi) -> (epsilon_gamma, epsilon_pi)

Draw correlated shocks from Cholesky decomposition of residual covariance.
MATLAB (epsilon_.m): returns (0,0) for degenerate covariance.
"""
function draw_correlated_epsilons(u_gamma::Vector, u_pi::Vector)
    n = min(length(u_gamma), length(u_pi))
    u_gamma = u_gamma[end-n+1:end]
    u_pi = u_pi[end-n+1:end]

    C = cov([u_gamma u_pi])

    if all(C .== 0) || !isposdef(C)
        return 0.0, 0.0
    end

    L = cholesky(C).L
    z = randn(2)

    epsilon_gamma = L[1, 1] * z[1] + L[1, 2] * z[2]
    epsilon_pi = L[2, 1] * z[1] + L[2, 2] * z[2]

    return epsilon_gamma, epsilon_pi
end

# =====================================================
# AR(1) ESTIMATION WITH RESIDUALS
# =====================================================

"""
    estimate_with_residuals(ydata) -> (alpha, beta, residuals)

Estimate AR(1) and return parameters plus residuals for covariance computation.
"""
function estimate_with_residuals(ydata::Vector)
    if length(ydata) < 3
        return 0.0, 0.0, zeros(1)
    end
    ydata_mat = ydata[:, :]
    var = Bit.rfvar3(ydata_mat, 1, ones(size(ydata_mat, 1), 1))
    alpha = var.By[1]
    beta = var.Bx[1]
    u = vec(var.u)
    return alpha, beta, u
end

# =====================================================
# DOMESTIC PRICE AVERAGE (MATLAB line 252)
# =====================================================

"""
    compute_P_bar_dom_g(firms, G) -> Vector

P_bar_dom_g(g) = sum(P_i .* Q_i) / sum(Q_i) for sector g.
Used by CANVAS, CATS, and EUBI for domestic price weighting.
"""
function compute_P_bar_dom_g(firms, G::Int)
    P_bar_dom_g = zeros(G)
    for g in 1:G
        sector_mask = firms.G_i .== g
        sector_P = firms.P_i[sector_mask]
        sector_Q = firms.Q_i[sector_mask]
        total_Q = sum(sector_Q)
        if total_Q > 0
            P_bar_dom_g[g] = sum(sector_P .* sector_Q) / total_Q
        else
            P_bar_dom_g[g] = mean(sector_P)
        end
    end
    return P_bar_dom_g
end

# =====================================================
# MATLAB-STYLE GROWTH EXPECTATIONS (AR on gamma)
# =====================================================

"""
    growth_inflation_expectations_matlab_core(model) -> (Y_e, gamma_e, pi_e)

Compute growth/inflation expectations using AR on growth rates (MATLAB style).
Works on any firms type that has `supply_history` and `last_supply_update_t` fields.

MATLAB (abm.m lines 90, 104, 307-308):
    gamma_Y = exp(alpha*gamma_{t-1} + beta + epsilon) - 1
    Y = sum(Q_i + S_i + DS_DISP_i)
"""
function growth_inflation_expectations_matlab_core(model)
    pi = model.agg.pi_
    T_prime = model.prop.T_prime
    t = model.agg.t
    firms = model.firms

    supply_history = firms.supply_history
    last_update_t = firms.last_supply_update_t[]

    # Update supply_history with previous step's supply
    if t > 1 && last_update_t < t
        supply_available = sum(firms.Q_i .+ firms.S_i .+ firms.DS_i)
        push!(supply_history, supply_available)
        firms.last_supply_update_t[] = t
    end

    n_history = length(supply_history)

    # Need at least 3 values for AR estimation
    if n_history < 3
        return Bit.growth_inflation_expectations(model)
    end

    # Compute gamma from supply_history
    gamma_slice = diff(log.(supply_history))
    pi_slice = pi[1:(T_prime + t - 1)]

    if length(gamma_slice) < 3 || length(pi_slice) < 3
        return Bit.growth_inflation_expectations(model)
    end

    # AR(1) on gamma (MATLAB style)
    alpha_gamma, beta_gamma, u_gamma = estimate_with_residuals(gamma_slice)
    alpha_pi, beta_pi, u_pi = estimate_with_residuals(pi_slice)

    # Draw correlated epsilons
    epsilon_gamma, epsilon_pi = draw_correlated_epsilons(u_gamma, u_pi)

    # MATLAB formula: gamma_Y = exp(alpha*gamma_{t-1} + beta + epsilon) - 1
    gamma_prev = gamma_slice[end]
    gamma_e = exp(alpha_gamma * gamma_prev + beta_gamma + epsilon_gamma) - 1

    Y_prev = supply_history[end]
    Y_e = Y_prev * (1 + gamma_e)

    pi_prev = pi_slice[end]
    pi_e = exp(alpha_pi * pi_prev + beta_pi + epsilon_pi) - 1

    return Y_e, gamma_e, pi_e
end

# =====================================================
# FIRMS_STOCKS: delta_S=1 (no inventory accumulation)
# =====================================================

"""
    firms_stocks_delta_s1(firms) -> (K_i, M_i, DS_i, S_i)

MATLAB (abm.m lines 262-264) with delta_S=1:
  DS_i = Y_i - Q_i  (one-period disposable stock)
  S_i unchanged (no accumulation)
"""
function firms_stocks_delta_s1(firms)
    K_i = firms.K_i - firms.delta_i ./ firms.kappa_i .* firms.Y_i + firms.I_i
    M_i = firms.M_i - firms.Y_i ./ firms.beta_i + firms.DM_i
    DS_i = firms.Y_i - firms.Q_i
    S_i = firms.S_i
    return K_i, M_i, DS_i, S_i
end

# =====================================================
# FIRMS PROFITS: delta_S=1 (DS_i=0 in profit calculation)
# =====================================================

"""
    firms_profits_delta_s1(firms, model)

MATLAB abm.m line 266: Pi_i = P_i.*Q_i + P_i.*DS_i - costs
With delta_S=1: DS_i = 0, so profits = P_i*Q_i - costs only.
Julia's DS_i field holds Y_i-Q_i (= MATLAB's DS_DISP_i), which must NOT be in profits.
"""
function firms_profits_delta_s1(firms, model)
    P_bar_HH = model.agg.P_bar_HH
    tau_SIF = model.prop.tau_SIF
    r = model.bank.r
    r_bar = model.cb.r_bar

    # DS_i = 0 for delta_S=1 (MATLAB: only Q_i contributes to sales revenue)
    in_sales = firms.P_i .* firms.Q_i
    in_deposits = r_bar .* Bit.pos(firms.D_i)
    out_wages = (1.0 + tau_SIF) .* firms.w_i .* firms.N_i .* P_bar_HH
    out_expenses = 1.0 ./ firms.beta_i .* firms.P_bar_i .* firms.Y_i
    out_depreciation = firms.delta_i ./ firms.kappa_i .* firms.P_CF_i .* firms.Y_i
    out_taxes_prods = firms.tau_Y_i .* firms.P_i .* firms.Y_i
    out_taxes_capital = firms.tau_K_i .* firms.P_i .* firms.Y_i
    out_loans = r .* (firms.L_i .+ Bit.pos(-firms.D_i))

    return in_sales + in_deposits - out_wages - out_expenses - out_depreciation -
           out_taxes_prods - out_taxes_capital - out_loans
end

# =====================================================
# INFLATION/PRICE INDEX: MATLAB weighting (Q_i+S_i+DS_DISP_i)
# =====================================================

"""
    inflation_priceindex_matlab(firms, P_bar)

MATLAB abm.m:247-248: P = sum(P_i.*(Q_i+S_i+DS_DISP_i)) / sum(Q_i+S_i+DS_DISP_i)
With delta_S=1: Q_i+S_i+DS_DISP_i = Y_i+S_i
"""
function inflation_priceindex_matlab(firms, P_bar)
    weights = firms.Y_i .+ firms.S_i
    price_index = sum(firms.P_i .* weights) / sum(weights)
    inflation = log(price_index / P_bar)
    return inflation, price_index
end

# =====================================================
# GDP MEASURE: MATLAB (sum of Y_i + S_i)
# =====================================================

"""
    compute_GDP_matlab(firms)

MATLAB abm.m:307: Y = sum(Q_i + S_i + DS_DISP_i) = sum(Y_i + S_i) with delta_S=1
"""
compute_GDP_matlab(firms) = sum(firms.Y_i .+ firms.S_i)

# =====================================================
# INITIALIZATION HELPERS
# =====================================================

"""
    init_markup_firms(firms_st, tau_SIF) -> (initial_AC_e, initial_mu)

MATLAB initialization (abm.m lines 32-33):
    AC_e_i = (1+tau_SIF).*w_bar_i./alpha_bar_i + delta_i./kappa_i + 1./beta_i
    mu_i = 1./AC_e_i - 1
"""
function init_markup_firms(firms_st, tau_SIF)
    I = length(firms_st.G_i)
    initial_AC_e = zeros(I)
    for i in 1:I
        initial_AC_e[i] = (1 + tau_SIF) * firms_st.w_bar_i[i] / firms_st.alpha_bar_i[i] +
                          firms_st.delta_i[i] / firms_st.kappa_i[i] +
                          1 / firms_st.beta_i[i]
    end
    initial_mu = 1.0 ./ initial_AC_e .- 1.0
    return initial_AC_e, initial_mu
end

"""
    init_supply_history(p, ic) -> Vector

Initialize supply_history from calibration data. Uses ic["Y"] if available
(EUBI-style), otherwise falls back to agg.Y[1:T_prime].
"""
function init_supply_history(p, ic)
    if haskey(ic, "Y")
        return Vector{Bit.typeFloat}(vec(ic["Y"]))
    else
        agg = Bit.Aggregates(p, ic)
        T_prime = p["T_prime"]
        return copy(agg.Y[1:T_prime])
    end
end

"""
    init_Q_Y_fallback!(firms)

Set Q_i, Y_i to Q_d_i if they are all zeros (needed for supply/market-share calcs).
"""
function init_Q_Y_fallback!(firms)
    if all(firms.Q_i .== 0)
        firms.Q_i .= firms.Q_d_i
    end
    if all(firms.Y_i .== 0)
        firms.Y_i .= firms.Q_d_i
    end
end

"""
    assemble_model(firms, p, ic) -> Bit.Model

Create a complete Model from firms + standard agents.
Uses DDGABM gov/rotw if 6x6 covariance params are available.
"""
function assemble_model(firms, p, ic)
    w_act, w_inact = Bit.Workers(p, ic)
    bank = Bit.Bank(p, ic)
    central_bank = Bit.CentralBank(p, ic)

    if haskey(p, "C6")
        gov, rotw = create_ddgabm_gov_rotw(p, ic)
    else
        gov = Bit.Government(p, ic)
        rotw = Bit.RestOfTheWorld(p, ic)
    end

    agg = Bit.Aggregates(p, ic)
    prop = Bit.Properties(p, ic)
    data = Bit.Data(p)

    return Bit.Model(w_act, w_inact, firms, bank, central_bank, gov, rotw, agg, prop, data)
end
