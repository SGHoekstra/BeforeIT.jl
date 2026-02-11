"""
=====================================================================
CATS MODEL IMPLEMENTATION - EXACT MATLAB MATCH
=====================================================================

Markup-based pricing: P_i = (1 + μ_i) × AC_e_i

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
using Dates

# Include shared helpers
if !@isdefined(growth_inflation_expectations_matlab_core)
    include("shared_pricing_helpers.jl")
end

# =====================================================
# CATS FIRM TYPE
# =====================================================

abstract type AbstractFirmsCATS <: Bit.AbstractFirms end

Bit.@object mutable struct FirmsCATS(Bit.Firms) <: AbstractFirmsCATS
    mu_i::Vector{Bit.typeFloat}
    AC_e_i::Vector{Bit.typeFloat}
    AC_e_lag_i::Vector{Bit.typeFloat}
    supply_history::Vector{Bit.typeFloat}
    last_supply_update_t::Ref{Bit.typeInt}
end

# =====================================================
# DISPATCH: growth_inflation_expectations
# =====================================================

function Bit.growth_inflation_expectations(
    model::Bit.Model{<:Bit.AbstractWorkers, <:Bit.AbstractWorkers, <:AbstractFirmsCATS,
                     <:Bit.AbstractBank, <:Bit.AbstractCentralBank, <:Bit.AbstractGovernment,
                     <:Bit.AbstractRestOfTheWorld, <:Bit.AbstractAggregates})
    return growth_inflation_expectations_matlab_core(model)
end

# =====================================================
# DISPATCH: firms_stocks (delta_S=1)
# =====================================================

Bit.firms_stocks(firms::AbstractFirmsCATS) = firms_stocks_delta_s1(firms)

# =====================================================
# DISPATCH: firms_profits (delta_S=1, DS_i=0 in revenue)
# =====================================================

Bit.firms_profits(firms::AbstractFirmsCATS, model::Bit.AbstractModel) = firms_profits_delta_s1(firms, model)

# =====================================================
# CATS PRICING: P = (1+μ) × AC_e
# =====================================================
# δ_C = conditional, δ_Y = 0, δ_MS = 0, δ_AC = 1, δ_rec = 1, δ_P = 0

function Bit.firms_expectations_and_decisions(firms::AbstractFirmsCATS, model::Bit.AbstractModel)
    P_bar_g = model.agg.P_bar_g
    P_bar_HH = model.agg.P_bar_HH
    P_bar_CF = model.agg.P_bar_CF
    gamma_e = model.agg.gamma_e
    pi_e = model.agg.pi_e
    tau_SIF = model.prop.tau_SIF
    a_sg = model.prop.a_sg

    I = length(firms.G_i)
    G = length(P_bar_g)

    # 0. P_bar_dom_g (MATLAB line 252)
    P_bar_dom_g = compute_P_bar_dom_g(firms, G)

    # 1. CONDITIONAL DELTA_C (same as CANVAS)
    delta_C_i = zeros(I)
    gamma_C_i = zeros(I)

    for i in 1:I
        g = firms.G_i[i]
        supply = firms.Q_i[i] + firms.S_i[i] + firms.DS_i[i]
        demand = firms.Q_d_i[i]
        P_i = firms.P_i[i]
        P_bar_dom = P_bar_dom_g[g]

        gamma_C_i[i] = demand / max(supply, 1e-10) - 1

        if (supply <= demand && P_i >= P_bar_dom) || (supply > demand && P_i < P_bar_dom)
            delta_C_i[i] = 0.0
        else
            delta_C_i[i] = 1.0
        end
    end

    # 2. QUANTITY DECISION - CATS: delta_Y = 0 (NO gamma_e!)
    # Q_s = max(0, (1+(1-δ_C)×γ_C) × supply - S)
    Q_s_i = zeros(I)
    for i in 1:I
        supply = firms.Q_i[i] + firms.S_i[i] + firms.DS_i[i]
        Q_s_i[i] = max(0.0, (1 + (1 - delta_C_i[i]) * gamma_C_i[i]) * supply - firms.S_i[i])
    end

    # 3. EXPECTED AVERAGE COST (MATLAB line 152)
    firms.AC_e_lag_i .= firms.AC_e_i
    for i in 1:I
        g = firms.G_i[i]
        labour_cost = (1 + tau_SIF) * firms.w_bar_i[i] / firms.alpha_bar_i[i] * P_bar_HH
        material_cost = (1 / firms.beta_i[i]) * sum(a_sg[:, g] .* P_bar_g)
        capital_cost = (firms.delta_i[i] / firms.kappa_i[i]) * P_bar_CF
        firms.AC_e_i[i] = (labour_cost + material_cost + capital_cost) * (1 + pi_e)
    end

    # 4. COST INFLATION (for absorption)
    pi_AC_i = zeros(I)
    for i in 1:I
        if firms.AC_e_lag_i[i] > 0
            pi_AC_i[i] = firms.AC_e_i[i] / firms.AC_e_lag_i[i] - 1
        end
    end

    # 5. MARKUP EVOLUTION: δ_rec=1, δ_AC=1
    # mu = (1+mu) × (1+δ_C×γ_C) / (1+π_AC) - 1
    for i in 1:I
        firms.mu_i[i] = (1 + firms.mu_i[i]) * (1 + delta_C_i[i] * gamma_C_i[i]) / (1 + pi_AC_i[i]) - 1
    end

    # 6. PRICE SETTING: P = (1+μ) × AC_e
    new_P_i = (1 .+ firms.mu_i) .* firms.AC_e_i
    new_P_i = max.(new_P_i, 1e-10)

    # 7. REST OF DECISIONS (standard)
    I_d_i, DM_d_i, N_d_i = Bit.desired_capital_material_employment(firms, Q_s_i)
    Pi_e_i = firms.Pi_i .* (1 + pi_e) * (1 + gamma_e)
    DD_e_i, K_e_i, L_e_i = Bit.expected_deposits_capital_loans(firms, model, Pi_e_i)
    DL_d_i = max.(0, -DD_e_i - firms.D_i)

    return Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, new_P_i
end

# =====================================================
# MODEL CREATOR
# =====================================================

function create_cats_model(p, ic)
    firms_st = Bit.Firms(p, ic)
    prop = Bit.Properties(p, ic)
    initial_AC_e, initial_mu = init_markup_firms(firms_st, prop.tau_SIF)
    supply_history = init_supply_history(p, ic)

    firms = FirmsCATS(
        (getfield(firms_st, x) for x in fieldnames(Bit.Firms))...,
        initial_mu, initial_AC_e, copy(initial_AC_e),
        supply_history, Ref(0)
    )
    init_Q_Y_fallback!(firms)

    return assemble_model(firms, p, ic)
end
