"""
=====================================================================
FULL KS MODEL IMPLEMENTATION
=====================================================================

Implementation matching DDGABM/abm.m exactly, with markup-based pricing:
    P_i = (1 + μ_i) × AC_e_i

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

KS (Keynes-Schumpeter) is the simplest pricing model:
- Quantity: Pure market share allocation
- Markup: μ = μ × (1 + 0.04×γ_MS)
- NO cost absorption (δ_AC=0), so costs pass through directly
- NO demand-pull, NO competitor consideration

Because δ_AC=0, KS passes through ALL cost changes:
- Productivity shock: α↑ → AC_e↓ → P↓ (full pass-through)
- Cost-push shock: material costs↑ → AC_e↑ → P↑ (full pass-through)

This creates STRONG price responses to cost shocks!
"""

import BeforeIT as Bit
using Dates, Statistics, LinearAlgebra

# =====================================================
# FULL KS FIRM TYPE
# =====================================================

abstract type AbstractFirmsFullKS <: Bit.AbstractFirms end

# Extended firms struct with markup and market share tracking
Bit.@object mutable struct FirmsFullKS(Bit.Firms) <: AbstractFirmsFullKS
    mu_i::Vector{Bit.typeFloat}          # Current markup per firm
    AC_e_i::Vector{Bit.typeFloat}        # Expected average cost per firm
    AC_e_lag_i::Vector{Bit.typeFloat}    # Previous period's AC_e (not used for absorption, but for consistency)
    MS_i::Vector{Bit.typeFloat}          # Current market share
    MS_lag_i::Vector{Bit.typeFloat}      # Previous market share
    supply_history::Vector{Bit.typeFloat}    # MATLAB-style: tracks Y+S for gamma computation
    last_supply_update_t::Ref{Bit.typeInt}   # Track when supply_history was last updated
end

# =====================================================
# MATLAB-STYLE AR ESTIMATION HELPERS
# =====================================================

"""
Draw correlated epsilon shocks using Cholesky decomposition.
"""
function draw_correlated_epsilons_ks(u_gamma::Vector, u_pi::Vector)
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
function estimate_with_residuals_ks(ydata::Vector)
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
function growth_inflation_expectations_matlab_style_ks(model)
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
    alpha_gamma, beta_gamma, u_gamma = estimate_with_residuals_ks(gamma_slice)
    alpha_pi, beta_pi, u_pi = estimate_with_residuals_ks(pi_slice)

    # Draw correlated epsilons
    epsilon_gamma, epsilon_pi = draw_correlated_epsilons_ks(u_gamma, u_pi)

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
# OVERRIDE growth_inflation_expectations FOR KS
# =====================================================

function Bit.growth_inflation_expectations(
    model::Bit.Model{<:Bit.AbstractWorkers, <:Bit.AbstractWorkers, <:AbstractFirmsFullKS,
                     <:Bit.AbstractBank, <:Bit.AbstractCentralBank, <:Bit.AbstractGovernment,
                     <:Bit.AbstractRestOfTheWorld, <:Bit.AbstractAggregates})
    return growth_inflation_expectations_matlab_style_ks(model)
end

# =====================================================
# FULL KS PRICING: P = (1+μ) × AC_e with market share
# =====================================================

function Bit.firms_expectations_and_decisions(firms::AbstractFirmsFullKS, model::Bit.AbstractModel)
    # Unpack model variables
    P_bar_g = model.agg.P_bar_g
    P_bar_HH = model.agg.P_bar_HH
    P_bar_CF = model.agg.P_bar_CF
    gamma_e = model.agg.gamma_e
    pi_e = model.agg.pi_e
    tau_SIF = model.prop.tau_SIF
    a_sg = model.prop.a_sg

    I = length(firms.G_i)
    G = length(P_bar_g)

    # =====================================================
    # 1. UPDATE MARKET SHARES (for γ_MS calculation)
    # =====================================================
    # EXACT MATLAB: MS_i = Q_i / sum(Q_i in sector)
    # Use Q_i (actual sales), not Y_i (production)

    firms.MS_lag_i .= firms.MS_i

    for g in 1:G
        sector_mask = firms.G_i .== g
        sector_Q = firms.Q_i[sector_mask]  # EXACT MATLAB: use Q_i (actual sales)
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

    # =====================================================
    # 2. QUANTITY DECISION (line 146)
    # =====================================================
    # KS: δ_Y = 0, δ_C = 0
    # This means NO growth forecast and NO demand-supply adjustment!
    # Pure market share allocation based on current demand.

    # From MATLAB (line 146):
    # Q_s = max(0, (1 + (1-δ_C)×γ_C) × (1 + δ_Y×γ_e) × (Q+S+DS_DISP) - S)
    #
    # With δ_Y=0, δ_C=0:
    # γ_C = Q_d/(Q+S+DS_DISP) - 1
    # Q_s = (1 + γ_C) × (Q+S+DS_DISP) - S
    #     = Q_d/(Q+S+DS_DISP) × (Q+S+DS_DISP) - S
    #     = Q_d - S
    #
    # So for KS, quantity planned = demand minus existing inventory

    Q_s_i = max.(firms.Q_d_i .- firms.S_i, 0.0)

    # =====================================================
    # 3. EXPECTED AVERAGE COST (line 152)
    # =====================================================

    firms.AC_e_lag_i .= firms.AC_e_i

    for i in 1:I
        g = firms.G_i[i]

        # Labor cost per unit output
        labour_cost = (1 + tau_SIF) * firms.w_bar_i[i] / firms.alpha_bar_i[i] * P_bar_HH

        # Material cost per unit output
        material_cost = (1 / firms.beta_i[i]) * sum(a_sg[:, g] .* P_bar_g)

        # Capital depreciation cost per unit output
        capital_cost = (firms.delta_i[i] / firms.kappa_i[i]) * P_bar_CF

        # Expected average cost
        firms.AC_e_i[i] = (labour_cost + material_cost + capital_cost) * (1 + pi_e)
    end

    # =====================================================
    # 4. MARKUP EVOLUTION (line 161)
    # =====================================================
    # For KS: δ_rec=0, δ_MS=0.04, δ_P=0, δ_AC=0, δ_C=0
    # μ = (δ_rec + μ) × (1 + δ_MS×γ_MS) × (1 + δ_P×π_P) × (1 + δ_C×γ_C) / (1 + δ_AC×π_AC) - δ_rec
    # = (0 + μ) × (1 + 0.04×γ_MS) × 1 × 1 / 1 - 0
    # = μ × (1 + 0.04×γ_MS)
    #
    # δ_AC=0 means NO cost absorption - costs pass through via AC_e directly!
    # This creates strong price responses to productivity and cost-push shocks.

    for i in 1:I
        firms.mu_i[i] = firms.mu_i[i] * (1 + 0.04 * gamma_MS_i[i])

        # Ensure markup stays reasonable
        firms.mu_i[i] = clamp(firms.mu_i[i], -0.5, 2.0)
    end

    # =====================================================
    # 5. PRICE SETTING (line 163)
    # =====================================================
    # P_i = (1 + μ_i) × AC_e_i
    #
    # Because δ_AC=0, ALL cost changes flow through AC_e to price!

    new_P_i = (1 .+ firms.mu_i) .* firms.AC_e_i
    new_P_i = max.(new_P_i, 1e-10)

    # =====================================================
    # 6. REST OF DECISIONS (same as standard model)
    # =====================================================

    I_d_i, DM_d_i, N_d_i = Bit.desired_capital_material_employment(firms, Q_s_i)
    Pi_e_i = firms.Pi_i .* (1 + pi_e) * (1 + gamma_e)
    DD_e_i, K_e_i, L_e_i = Bit.expected_deposits_capital_loans(firms, model, Pi_e_i)
    DL_d_i = max.(0, -DD_e_i - firms.D_i)

    return Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, new_P_i
end

# =====================================================
# HELPER: Create Full KS model
# =====================================================

"""
    create_full_ks_model(p, ic)

Create a Full KS model with markup-based pricing P = (1+μ)×AC_e.

KS is the simplest model with:
- Small market share effect: μ = μ × (1 + 0.04×γ_MS)
- FULL cost pass-through (δ_AC=0) - no cost absorption

MATLAB initialization (abm.m lines 32-33):
    AC_e_i = (1+tau_SIF).*w_bar_i./alpha_bar_i + delta_i./kappa_i + 1./beta_i
    mu_i = 1./AC_e_i - 1

This typically produces the strongest CPI responses to cost shocks.
"""
function create_full_ks_model(p, ic)
    # Initialize standard firms
    firms_st = Bit.Firms(p, ic)

    # Get number of firms
    I = length(firms_st.G_i)
    G = maximum(firms_st.G_i)

    # =====================================================
    # EXACT MATLAB INITIALIZATION (abm.m lines 32-33)
    # =====================================================
    # AC_e_i = (1+tau_SIF).*w_bar_i./alpha_bar_i + delta_i./kappa_i + 1./beta_i
    # mu_i = 1./AC_e_i - 1

    # Get tau_SIF from properties
    prop = Bit.Properties(p, ic)
    tau_SIF = prop.tau_SIF

    # Compute initial AC_e exactly like MATLAB (without price terms - real cost units)
    initial_AC_e = zeros(I)
    for i in 1:I
        initial_AC_e[i] = (1 + tau_SIF) * firms_st.w_bar_i[i] / firms_st.alpha_bar_i[i] +
                          firms_st.delta_i[i] / firms_st.kappa_i[i] +
                          1 / firms_st.beta_i[i]
    end

    # EXACT MATLAB: mu_i = 1./AC_e_i - 1
    initial_mu = 1.0 ./ initial_AC_e .- 1.0

    # Initialize market shares
    initial_MS = zeros(I)
    for g in 1:G
        sector_mask = firms_st.G_i .== g
        n_in_sector = sum(sector_mask)
        if n_in_sector > 0
            initial_MS[sector_mask] .= 1.0 / n_in_sector
        end
    end

    # Initialize supply_history from calibration data (Y from agg)
    agg = Bit.Aggregates(p, ic)
    T_prime = p["T_prime"]
    initial_supply_history = copy(agg.Y[1:T_prime])

    # Create Full KS firms
    firms = FirmsFullKS(
        (getfield(firms_st, x) for x in fieldnames(Bit.Firms))...,
        initial_mu,
        initial_AC_e,
        copy(initial_AC_e),
        initial_MS,
        copy(initial_MS),
        initial_supply_history,
        Ref(0)
    )

    # Initialize Q_i (actual sales) if needed for market share calculation
    # MATLAB uses Q_i for market share
    if all(firms.Q_i .== 0)
        firms.Q_i .= firms.Q_d_i
    end
    if all(firms.Y_i .== 0)
        firms.Y_i .= firms.Q_d_i
    end

    # Standard agent initializations
    w_act, w_inact = Bit.Workers(p, ic)
    bank = Bit.Bank(p, ic)
    central_bank = Bit.CentralBank(p, ic)
    gov = Bit.Government(p, ic)
    rotw = Bit.RestOfTheWorld(p, ic)
    # prop already created above for tau_SIF
    data = Bit.Data(p)

    return Bit.Model(w_act, w_inact, firms, bank, central_bank, gov, rotw, agg, prop, data)
end
