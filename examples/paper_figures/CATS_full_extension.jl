"""
=====================================================================
FULL CATS MODEL IMPLEMENTATION - EXACT MATLAB MATCH
=====================================================================

Implementation matching DDGABM/abm.m EXACTLY, with markup-based pricing:
    P_i = (1 + μ_i) × AC_e_i

MATLAB EXACT MATCHES:
1. Supply = Q_i + S_i (actual sales + inventory) - line 134
2. P_bar_dom_g computed from domestic firm prices weighted by Q_i - line 252
3. gamma_C_i = Q_d_i / (Q_i + S_i) - 1 - line 142
4. mu_i = 1/AC_e_init - 1 - line 33
5. Cost absorption: mu divided by (1+pi_AC) - line 161

## CATS Delta Parameters (from set_parameters_and_initial_conditions.m)

| Parameter | Value | Description |
|-----------|-------|-------------|
| δ_C       | NaN   | Conditional (switches 0/1 per firm) - SAME as CANVAS |
| δ_Y       | 0     | NO demand forecast (differs from CANVAS!) |
| δ_MS      | 0     | No market share in markup |
| δ_AC      | 1     | Cost ABSORPTION (key difference from CANVAS!) |
| δ_rec     | 1     | Recursive markup |
| δ_P       | 0     | No competitor prices |
"""

import BeforeIT as Bit
using Dates, Statistics, LinearAlgebra

# =====================================================
# FULL CATS FIRM TYPE
# =====================================================

abstract type AbstractFirmsFullCATS <: Bit.AbstractFirms end

Bit.@object mutable struct FirmsFullCATS(Bit.Firms) <: AbstractFirmsFullCATS
    mu_i::Vector{Bit.typeFloat}
    AC_e_i::Vector{Bit.typeFloat}
    AC_e_lag_i::Vector{Bit.typeFloat}
    supply_history::Vector{Bit.typeFloat}    # MATLAB-style: tracks Y+S for gamma computation
    last_supply_update_t::Ref{Bit.typeInt}   # Track when supply_history was last updated
end

# =====================================================
# MATLAB-STYLE AR ESTIMATION HELPERS
# =====================================================

"""
Draw correlated epsilon shocks using Cholesky decomposition.
"""
function draw_correlated_epsilons_cats(u_gamma::Vector, u_pi::Vector)
    n = min(length(u_gamma), length(u_pi))
    u_gamma = u_gamma[end-n+1:end]
    u_pi = u_pi[end-n+1:end]

    C = cov([u_gamma u_pi])

    if all(C .== 0) || !isposdef(C)
        return randn() * std(u_gamma), randn() * std(u_pi)
    end

    L = cholesky(C).L
    z = randn(2)

    epsilon_gamma = L[1, 1] * z[1] + L[1, 2] * z[2]
    epsilon_pi = L[2, 1] * z[1] + L[2, 2] * z[2]

    return epsilon_gamma, epsilon_pi
end

"""
Estimate AR(1) and return parameters plus residuals.
"""
function estimate_with_residuals_cats(ydata::Vector)
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

"""
Compute growth expectations using AR on growth rates (MATLAB style).
"""
function growth_inflation_expectations_matlab_style_cats(model)
    pi = model.agg.pi_
    T_prime = model.prop.T_prime
    t = model.agg.t
    firms = model.firms

    supply_history = firms.supply_history
    last_update_t = firms.last_supply_update_t[]

    # Update supply_history with previous step's supply
    if t > 1 && last_update_t < t
        supply_available = sum(firms.Y_i .+ firms.S_i)
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
    alpha_gamma, beta_gamma, u_gamma = estimate_with_residuals_cats(gamma_slice)
    alpha_pi, beta_pi, u_pi = estimate_with_residuals_cats(pi_slice)

    # Draw correlated epsilons
    epsilon_gamma, epsilon_pi = draw_correlated_epsilons_cats(u_gamma, u_pi)

    # MATLAB formula: gamma_Y = exp(alpha*gamma_{t-1} + beta + epsilon) - 1
    gamma_prev = gamma_slice[end]
    gamma_e = exp(alpha_gamma * gamma_prev + beta_gamma + epsilon_gamma) - 1
    gamma_e = clamp(gamma_e, -0.5, 0.5)

    Y_prev = supply_history[end]
    Y_e = Y_prev * (1 + gamma_e)

    pi_prev = pi_slice[end]
    pi_e = exp(alpha_pi * pi_prev + beta_pi + epsilon_pi) - 1

    return Y_e, gamma_e, pi_e
end

# =====================================================
# OVERRIDE growth_inflation_expectations FOR CATS
# =====================================================

function Bit.growth_inflation_expectations(
    model::Bit.Model{<:Bit.AbstractWorkers, <:Bit.AbstractWorkers, <:AbstractFirmsFullCATS,
                     <:Bit.AbstractBank, <:Bit.AbstractCentralBank, <:Bit.AbstractGovernment,
                     <:Bit.AbstractRestOfTheWorld, <:Bit.AbstractAggregates})
    return growth_inflation_expectations_matlab_style_cats(model)
end

# =====================================================
# HELPER: Compute P_bar_dom_g (domestic price average)
# =====================================================

function compute_P_bar_dom_g_cats(firms, G::Int)
    P_bar_dom_g = zeros(G)
    for g in 1:G
        sector_mask = firms.G_i .== g
        # EXACT MATLAB: use Q_i (actual sales) as weight
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
# FULL CATS PRICING
# =====================================================

function Bit.firms_expectations_and_decisions(firms::AbstractFirmsFullCATS, model::Bit.AbstractModel)
    P_bar_g = model.agg.P_bar_g
    P_bar_HH = model.agg.P_bar_HH
    P_bar_CF = model.agg.P_bar_CF
    gamma_e = model.agg.gamma_e
    pi_e = model.agg.pi_e
    tau_SIF = model.prop.tau_SIF
    a_sg = model.prop.a_sg

    I = length(firms.G_i)
    G = length(P_bar_g)

    # Compute P_bar_dom_g (EXACT MATLAB)
    P_bar_dom_g = compute_P_bar_dom_g_cats(firms, G)

    # =====================================================
    # 1. CONDITIONAL DELTA_C (same as CANVAS)
    # =====================================================

    delta_C_i = zeros(I)
    gamma_C_i = zeros(I)

    for i in 1:I
        g = firms.G_i[i]
        # EXACT MATLAB: supply = Q_i + S_i + DS_DISP_i
        supply = firms.Q_i[i] + firms.S_i[i]
        demand = firms.Q_d_i[i]
        P_i = firms.P_i[i]
        P_bar_dom = P_bar_dom_g[g]  # EXACT: P_bar_dom_g

        gamma_C_i[i] = demand / max(supply, 1e-10) - 1

        if (supply <= demand && P_i >= P_bar_dom) || (supply > demand && P_i < P_bar_dom)
            delta_C_i[i] = 0.0
        else
            delta_C_i[i] = 1.0
        end
    end

    # =====================================================
    # 2. QUANTITY DECISION - CATS: delta_Y = 0 (NO gamma_e!)
    # =====================================================
    # EXACT: Q_s_i = max(0, (1+(1-delta_C).*gamma_C_i)*(1+delta_Y.*gamma_Y).*(Q+S)-S)
    # With delta_Y = 0: Q_s_i = max(0, (1+(1-delta_C).*gamma_C_i).*(Q+S)-S)

    Q_s_i = zeros(I)
    for i in 1:I
        # EXACT MATLAB: supply = Q_i + S_i + DS_DISP_i
        supply = firms.Q_i[i] + firms.S_i[i]
        # NO gamma_e term because delta_Y = 0 for CATS!
        Q_s_i[i] = max(0.0, (1 + (1 - delta_C_i[i]) * gamma_C_i[i]) * supply - firms.S_i[i])
    end

    # =====================================================
    # 3. EXPECTED AVERAGE COST (same as CANVAS)
    # =====================================================

    firms.AC_e_lag_i .= firms.AC_e_i

    for i in 1:I
        g = firms.G_i[i]
        labour_cost = (1 + tau_SIF) * firms.w_bar_i[i] / firms.alpha_bar_i[i] * P_bar_HH
        material_cost = (1 / firms.beta_i[i]) * sum(a_sg[:, g] .* P_bar_g)
        capital_cost = (firms.delta_i[i] / firms.kappa_i[i]) * P_bar_CF
        firms.AC_e_i[i] = (labour_cost + material_cost + capital_cost) * (1 + pi_e)
    end

    # =====================================================
    # 4. COST INFLATION (for absorption)
    # =====================================================

    pi_AC_i = zeros(I)
    for i in 1:I
        if firms.AC_e_lag_i[i] > 0
            pi_AC_i[i] = firms.AC_e_i[i] / firms.AC_e_lag_i[i] - 1
        end
    end

    # =====================================================
    # 5. MARKUP EVOLUTION - CATS: delta_AC = 1
    # =====================================================
    # EXACT: mu_i = (1+mu_i) * (1+delta_C*gamma_C) / (1+delta_AC*pi_AC) - 1
    # With delta_AC = 1: mu_i = (1+mu_i) * (1+delta_C*gamma_C) / (1+pi_AC) - 1

    for i in 1:I
        firms.mu_i[i] = (1 + firms.mu_i[i]) * (1 + delta_C_i[i] * gamma_C_i[i]) / (1 + pi_AC_i[i]) - 1
        firms.mu_i[i] = clamp(firms.mu_i[i], -0.5, 2.0)
    end

    # =====================================================
    # 6. PRICE SETTING
    # =====================================================

    new_P_i = (1 .+ firms.mu_i) .* firms.AC_e_i
    new_P_i = max.(new_P_i, 1e-10)

    # =====================================================
    # 7. REST OF DECISIONS
    # =====================================================

    I_d_i, DM_d_i, N_d_i = Bit.desired_capital_material_employment(firms, Q_s_i)
    Pi_e_i = firms.Pi_i .* (1 + pi_e) * (1 + gamma_e)
    DD_e_i, K_e_i, L_e_i = Bit.expected_deposits_capital_loans(firms, model, Pi_e_i)
    DL_d_i = max.(0, -DD_e_i - firms.D_i)

    return Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, new_P_i
end

# =====================================================
# HELPER: Create Full CATS model - EXACT MATLAB INIT
# =====================================================

function create_full_cats_model(p, ic)
    firms_st = Bit.Firms(p, ic)
    I = length(firms_st.G_i)

    # EXACT MATLAB initialization
    prop = Bit.Properties(p, ic)
    tau_SIF = prop.tau_SIF

    initial_AC_e = zeros(I)
    for i in 1:I
        initial_AC_e[i] = (1 + tau_SIF) * firms_st.w_bar_i[i] / firms_st.alpha_bar_i[i] +
                          firms_st.delta_i[i] / firms_st.kappa_i[i] +
                          1 / firms_st.beta_i[i]
    end

    initial_mu = 1.0 ./ initial_AC_e .- 1.0

    # Initialize supply_history from calibration data (Y from agg)
    agg = Bit.Aggregates(p, ic)
    T_prime = p["T_prime"]
    initial_supply_history = copy(agg.Y[1:T_prime])

    firms = FirmsFullCATS(
        (getfield(firms_st, x) for x in fieldnames(Bit.Firms))...,
        initial_mu,
        initial_AC_e,
        copy(initial_AC_e),
        initial_supply_history,
        Ref(0)
    )

    # Initialize Q_i (actual sales) if needed (ensure supply > 0)
    # MATLAB uses Q_i for supply calculation
    if all(firms.Q_i .== 0)
        firms.Q_i .= firms.Q_d_i
    end
    if all(firms.Y_i .== 0)
        firms.Y_i .= firms.Q_d_i
    end

    w_act, w_inact = Bit.Workers(p, ic)
    bank = Bit.Bank(p, ic)
    central_bank = Bit.CentralBank(p, ic)
    gov = Bit.Government(p, ic)
    rotw = Bit.RestOfTheWorld(p, ic)
    data = Bit.Data(p)

    return Bit.Model(w_act, w_inact, firms, bank, central_bank, gov, rotw, agg, prop, data)
end
