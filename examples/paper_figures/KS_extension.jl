"""
=====================================================================
KS MODEL IMPLEMENTATION - EXACT MATLAB MATCH
=====================================================================

Markup-based pricing: P_i = (1 + μ_i) × AC_e_i

## KS Delta Parameters (from set_parameters_and_initial_conditions.m)

| Parameter | Value | Description |
|-----------|-------|-------------|
| δ_C       | 0     | NO demand-pull adjustment |
| δ_Y       | 0     | NO demand forecast |
| δ_MS      | 0.04  | Small market share effect |
| δ_AC      | 0     | NO cost absorption (costs pass through!) |
| δ_rec     | 0     | STATIC markup (not recursive) |
| δ_P       | 0     | NO competitor prices |

## Key Characteristics

KS (Keynes-Schumpeter) is the simplest pricing model.
Because δ_AC=0, ALL cost changes flow through to prices:
  Productivity shock: α↑ → AC_e↓ → P↓ (full pass-through)
  Cost-push shock: material costs↑ → AC_e↑ → P↑ (full pass-through)
"""

import BeforeIT as Bit
using Dates

# Include shared helpers
if !@isdefined(growth_inflation_expectations_matlab_core)
    include("shared_pricing_helpers.jl")
end

# =====================================================
# KS FIRM TYPE
# =====================================================

abstract type AbstractFirmsKS <: Bit.AbstractFirms end

Bit.@object mutable struct FirmsKS(Bit.Firms) <: AbstractFirmsKS
    mu_i::Vector{Bit.typeFloat}
    AC_e_i::Vector{Bit.typeFloat}
    AC_e_lag_i::Vector{Bit.typeFloat}
    MS_i::Vector{Bit.typeFloat}
    MS_lag_i::Vector{Bit.typeFloat}
    supply_history::Vector{Bit.typeFloat}
    last_supply_update_t::Ref{Bit.typeInt}
end

# =====================================================
# DISPATCH: growth_inflation_expectations
# =====================================================

function Bit.growth_inflation_expectations(
    model::Bit.Model{<:Bit.AbstractWorkers, <:Bit.AbstractWorkers, <:AbstractFirmsKS,
                     <:Bit.AbstractBank, <:Bit.AbstractCentralBank, <:Bit.AbstractGovernment,
                     <:Bit.AbstractRestOfTheWorld, <:Bit.AbstractAggregates})
    return growth_inflation_expectations_matlab_core(model)
end

# =====================================================
# DISPATCH: firms_stocks (delta_S=1)
# =====================================================

Bit.firms_stocks(firms::AbstractFirmsKS) = firms_stocks_delta_s1(firms)

# =====================================================
# DISPATCH: firms_profits (delta_S=1, DS_i=0 in revenue)
# =====================================================

Bit.firms_profits(firms::AbstractFirmsKS, model::Bit.AbstractModel) = firms_profits_delta_s1(firms, model)

# =====================================================
# KS PRICING: P = (1+μ) × AC_e with market share
# =====================================================
# δ_C = 0, δ_Y = 0, δ_MS = 0.04, δ_AC = 0, δ_rec = 0, δ_P = 0

function Bit.firms_expectations_and_decisions(firms::AbstractFirmsKS, model::Bit.AbstractModel)
    P_bar_g = model.agg.P_bar_g
    P_bar_HH = model.agg.P_bar_HH
    P_bar_CF = model.agg.P_bar_CF
    gamma_e = model.agg.gamma_e
    pi_e = model.agg.pi_e
    tau_SIF = model.prop.tau_SIF
    a_sg = model.prop.a_sg

    I = length(firms.G_i)
    G = length(P_bar_g)

    # 1. UPDATE MARKET SHARES
    # MATLAB: MS_i = Q_i / sum(Q_i in sector)
    firms.MS_lag_i .= firms.MS_i
    for g in 1:G
        sector_mask = firms.G_i .== g
        sector_Q = firms.Q_i[sector_mask]
        total_Q = sum(sector_Q)
        if total_Q > 0
            sector_indices = findall(sector_mask)
            for idx in sector_indices
                firms.MS_i[idx] = firms.Q_i[idx] / total_Q
            end
        end
    end

    # Market share growth
    gamma_MS_i = zeros(I)
    for i in 1:I
        if firms.MS_lag_i[i] > 0
            gamma_MS_i[i] = firms.MS_i[i] / firms.MS_lag_i[i] - 1
        end
    end

    # 2. QUANTITY DECISION: δ_Y=0, δ_C=0
    # Simplifies to Q_s = Q_d - S
    Q_s_i = max.(firms.Q_d_i .- firms.S_i, 0.0)

    # 3. EXPECTED AVERAGE COST (MATLAB line 152)
    firms.AC_e_lag_i .= firms.AC_e_i
    for i in 1:I
        g = firms.G_i[i]
        labour_cost = (1 + tau_SIF) * firms.w_bar_i[i] / firms.alpha_bar_i[i] * P_bar_HH
        material_cost = (1 / firms.beta_i[i]) * sum(a_sg[:, g] .* P_bar_g)
        capital_cost = (firms.delta_i[i] / firms.kappa_i[i]) * P_bar_CF
        firms.AC_e_i[i] = (labour_cost + material_cost + capital_cost) * (1 + pi_e)
    end

    # 4. MARKUP EVOLUTION: δ_rec=0, δ_MS=0.04
    # mu = mu × (1 + 0.04×γ_MS)
    for i in 1:I
        firms.mu_i[i] = firms.mu_i[i] * (1 + 0.04 * gamma_MS_i[i])
    end

    # 5. PRICE SETTING: P = (1+μ) × AC_e
    new_P_i = (1 .+ firms.mu_i) .* firms.AC_e_i
    new_P_i = max.(new_P_i, 1e-10)

    # 6. REST OF DECISIONS (standard)
    I_d_i, DM_d_i, N_d_i = Bit.desired_capital_material_employment(firms, Q_s_i)
    Pi_e_i = firms.Pi_i .* (1 + pi_e) * (1 + gamma_e)
    DD_e_i, K_e_i, L_e_i = Bit.expected_deposits_capital_loans(firms, model, Pi_e_i)
    DL_d_i = max.(0, -DD_e_i - firms.D_i)

    return Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, new_P_i
end

# =====================================================
# MODEL CREATOR
# =====================================================

function create_ks_model(p, ic)
    firms_st = Bit.Firms(p, ic)
    I = length(firms_st.G_i)
    G = maximum(firms_st.G_i)

    prop = Bit.Properties(p, ic)
    initial_AC_e, initial_mu = init_markup_firms(firms_st, prop.tau_SIF)

    # Initialize market shares (equal within sector)
    initial_MS = zeros(I)
    for g in 1:G
        sector_mask = firms_st.G_i .== g
        n_in_sector = sum(sector_mask)
        if n_in_sector > 0
            initial_MS[sector_mask] .= 1.0 / n_in_sector
        end
    end

    supply_history = init_supply_history(p, ic)

    firms = FirmsKS(
        (getfield(firms_st, x) for x in fieldnames(Bit.Firms))...,
        initial_mu, initial_AC_e, copy(initial_AC_e),
        initial_MS, copy(initial_MS),
        supply_history, Ref(0)
    )
    init_Q_Y_fallback!(firms)

    return assemble_model(firms, p, ic)
end
