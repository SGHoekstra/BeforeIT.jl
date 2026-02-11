"""
=====================================================================
EUBI MODEL IMPLEMENTATION - EXACT MATLAB MATCH
=====================================================================

Markup-based pricing: P_i = (1 + μ_i) × AC_e_i

## EUBI Delta Parameters (from set_parameters_and_initial_conditions.m)

| Parameter | Value | Description |
|-----------|-------|-------------|
| δ_C       | 0     | NO demand-pull adjustment |
| δ_Y       | 1     | Uses demand forecast |
| δ_MS      | 1     | Market share affects markup |
| δ_AC      | 1     | Cost absorption |
| δ_rec     | 0     | STATIC markup (not recursive!) |
| δ_P       | 1     | Considers competitor prices |
| δ_S       | 1     | Full disequilibrium inventory (DS_DISP = Y - Q) |

## Key Differences from CANVAS/CATS

EUBI has the richest pricing: markups respond to market share changes (γ_MS),
competitor price inflation (π_P), AND cost inflation absorption (π_AC).
With δ_rec=0, the markup is STATIC (no demand-pull recursive feedback).
"""

import BeforeIT as Bit
using Dates

# Include shared helpers
if !@isdefined(growth_inflation_expectations_matlab_core)
    include("shared_pricing_helpers.jl")
end

# =====================================================
# EUBI FIRM TYPE
# =====================================================

abstract type AbstractFirmsEUBI <: Bit.AbstractFirms end

Bit.@object mutable struct FirmsEUBI(Bit.Firms) <: AbstractFirmsEUBI
    mu_i::Vector{Bit.typeFloat}
    AC_e_i::Vector{Bit.typeFloat}
    AC_e_lag_i::Vector{Bit.typeFloat}
    MS_i::Vector{Bit.typeFloat}           # Current market share
    MS_lag_i::Vector{Bit.typeFloat}       # Previous market share
    P_bar_e_g::Vector{Bit.typeFloat}      # Expected sector prices (for pi_P)
    P_bar_e_lag_g::Vector{Bit.typeFloat}  # Previous expected sector prices
    supply_history::Vector{Bit.typeFloat}
    last_supply_update_t::Ref{Bit.typeInt}
end

# =====================================================
# DISPATCH: growth_inflation_expectations
# =====================================================

function Bit.growth_inflation_expectations(
    model::Bit.Model{W1, W2, F, B, C, G, R, A, P, D}
) where {W1, W2, F<:AbstractFirmsEUBI, B, C, G, R, A, P, D}
    return growth_inflation_expectations_matlab_core(model)
end

# =====================================================
# DISPATCH: firms_stocks (delta_S=1)
# =====================================================

Bit.firms_stocks(firms::AbstractFirmsEUBI) = firms_stocks_delta_s1(firms)

# =====================================================
# DISPATCH: firms_profits (delta_S=1, DS_i=0 in revenue)
# =====================================================

Bit.firms_profits(firms::AbstractFirmsEUBI, model::Bit.AbstractModel) = firms_profits_delta_s1(firms, model)

# =====================================================
# EUBI PRICING: P = (1+μ) × AC_e
# =====================================================
# δ_C = 0, δ_Y = 1, δ_MS = 1, δ_AC = 1, δ_rec = 0, δ_P = 1

function Bit.firms_expectations_and_decisions(firms::AbstractFirmsEUBI, model::Bit.AbstractModel)
    P_bar_g = model.agg.P_bar_g
    P_bar_HH = model.agg.P_bar_HH
    P_bar_CF = model.agg.P_bar_CF
    gamma_e = model.agg.gamma_e
    pi_e = model.agg.pi_e
    tau_SIF = model.prop.tau_SIF
    a_sg = model.prop.a_sg

    I = length(firms.G_i)
    G = length(P_bar_g)

    # 0. UPDATE P_bar_e_g (expected sector prices for pi_P calculation)
    P_bar_dom_g = compute_P_bar_dom_g(firms, G)
    firms.P_bar_e_lag_g .= firms.P_bar_e_g
    firms.P_bar_e_g .= P_bar_dom_g .* (1 + pi_e)

    # 1. UPDATE MARKET SHARES (MATLAB lines 40-42)
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

    # Market share growth (MATLAB line 155)
    gamma_MS_i = zeros(I)
    for i in 1:I
        if firms.MS_lag_i[i] > 0
            gamma_MS_i[i] = firms.MS_i[i] / firms.MS_lag_i[i] - 1
        end
    end

    # 2. QUANTITY DECISION: δ_Y=1, δ_C=0
    # Q_s = Q_d × (1 + γ_e) - S
    Q_s_i = max.(firms.Q_d_i .* (1 + gamma_e) .- firms.S_i, 0.0)

    # 3. EXPECTED AVERAGE COST (MATLAB line 152)
    firms.AC_e_lag_i .= firms.AC_e_i
    for i in 1:I
        g = firms.G_i[i]
        labour_cost = (1 + tau_SIF) * firms.w_bar_i[i] / firms.alpha_bar_i[i] * P_bar_HH
        material_cost = (1 / firms.beta_i[i]) * sum(a_sg[:, g] .* P_bar_g)
        capital_cost = (firms.delta_i[i] / firms.kappa_i[i]) * P_bar_CF
        firms.AC_e_i[i] = (labour_cost + material_cost + capital_cost) * (1 + pi_e)
    end

    # 4. COST INFLATION
    pi_AC_i = zeros(I)
    for i in 1:I
        if firms.AC_e_lag_i[i] > 0
            pi_AC_i[i] = firms.AC_e_i[i] / firms.AC_e_lag_i[i] - 1
        end
    end

    # 5. COMPETITOR PRICE INFLATION (MATLAB line 153)
    pi_P_i = zeros(I)
    for i in 1:I
        g = firms.G_i[i]
        if firms.P_bar_e_lag_g[g] > 0
            pi_P_i[i] = firms.P_bar_e_g[g] / firms.P_bar_e_lag_g[g] - 1
        else
            pi_P_i[i] = pi_e
        end
    end

    # 6. MARKUP EVOLUTION: δ_rec=0, δ_MS=1, δ_P=1, δ_AC=1
    # mu = mu × (1+γ_MS) × (1+π_P) / (1+π_AC)
    for i in 1:I
        firms.mu_i[i] = firms.mu_i[i] * (1 + gamma_MS_i[i]) * (1 + pi_P_i[i]) / (1 + pi_AC_i[i])
    end

    # 7. PRICE SETTING: P = (1+μ) × AC_e
    new_P_i = (1 .+ firms.mu_i) .* firms.AC_e_i
    new_P_i = max.(new_P_i, 1e-10)

    # 8. REST OF DECISIONS (standard)
    I_d_i, DM_d_i, N_d_i = Bit.desired_capital_material_employment(firms, Q_s_i)
    Pi_e_i = firms.Pi_i .* (1 + pi_e) * (1 + gamma_e)
    DD_e_i, K_e_i, L_e_i = Bit.expected_deposits_capital_loans(firms, model, Pi_e_i)
    DL_d_i = max.(0, -DD_e_i - firms.D_i)

    return Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, new_P_i
end

# =====================================================
# MODEL CREATOR
# =====================================================

function create_eubi_model(p, ic)
    firms_st = Bit.Firms(p, ic)
    I = length(firms_st.G_i)
    G = maximum(firms_st.G_i)

    prop = Bit.Properties(p, ic)
    initial_AC_e, initial_mu = init_markup_firms(firms_st, prop.tau_SIF)

    # Initialize market shares from firm sizes
    initial_MS = zeros(I)
    for g in 1:G
        sector_mask = firms_st.G_i .== g
        sector_Q = firms_st.Q_d_i[sector_mask]
        total_Q = sum(sector_Q)
        if total_Q > 0
            sector_indices = findall(sector_mask)
            for idx in sector_indices
                initial_MS[idx] = firms_st.Q_d_i[idx] / total_Q
            end
        else
            n_in_sector = sum(sector_mask)
            if n_in_sector > 0
                initial_MS[sector_mask] .= 1.0 / n_in_sector
            end
        end
    end

    initial_P_bar_e_g = ones(G)
    supply_history = init_supply_history(p, ic)

    firms = FirmsEUBI(
        (getfield(firms_st, x) for x in fieldnames(Bit.Firms))...,
        initial_mu, initial_AC_e, copy(initial_AC_e),
        initial_MS, copy(initial_MS),
        initial_P_bar_e_g, copy(initial_P_bar_e_g),
        supply_history, Ref{Bit.typeInt}(0)
    )
    init_Q_Y_fallback!(firms)

    return assemble_model(firms, p, ic)
end
