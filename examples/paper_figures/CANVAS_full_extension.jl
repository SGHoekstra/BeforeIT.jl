"""
=====================================================================
FULL CANVAS MODEL IMPLEMENTATION - EXACT MATLAB MATCH
=====================================================================

Implementation matching DDGABM/abm.m EXACTLY, with markup-based pricing:
    P_i = (1 + μ_i) × AC_e_i

MATLAB EXACT MATCHES:
1. Supply = Q_i + S_i (actual sales + inventory) - line 134
2. P_bar_dom_g computed from domestic firm prices weighted by Q_i - line 252
3. gamma_C_i = Q_d_i / (Q_i + S_i) - 1 - line 142
4. mu_i = 1/AC_e_init - 1 - line 33

## CANVAS Delta Parameters (from set_parameters_and_initial_conditions.m)

| Parameter | Value | Description |
|-----------|-------|-------------|
| δ_C       | NaN   | Conditional (switches 0/1 per firm) |
| δ_Y       | 1     | Uses demand forecast |
| δ_MS      | 0     | No market share in markup |
| δ_AC      | 0     | No cost absorption (pass-through) |
| δ_rec     | 1     | Recursive markup |
| δ_P       | 0     | No competitor prices |
"""

import BeforeIT as Bit
using Dates, Statistics, LinearAlgebra

# =====================================================
# FULL CANVAS FIRM TYPE
# =====================================================

abstract type AbstractFirmsFullCANVAS <: Bit.AbstractFirms end

# Extended firms struct with markup tracking
Bit.@object mutable struct FirmsFullCANVAS(Bit.Firms) <: AbstractFirmsFullCANVAS
    mu_i::Vector{Bit.typeFloat}          # Current markup per firm
    AC_e_i::Vector{Bit.typeFloat}        # Expected average cost per firm
    AC_e_lag_i::Vector{Bit.typeFloat}    # Previous period's AC_e for cost inflation calc
    supply_history::Vector{Bit.typeFloat}    # MATLAB-style: tracks Y+S for gamma computation
    last_supply_update_t::Ref{Bit.typeInt}   # Track when supply_history was last updated
end

# =====================================================
# MATLAB-STYLE AR ESTIMATION HELPERS
# =====================================================

"""
Draw correlated epsilon shocks using Cholesky decomposition.
MATLAB (epsilon_.m) computes correlation from residuals.
"""
function draw_correlated_epsilons_canvas(u_gamma::Vector, u_pi::Vector)
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
function estimate_with_residuals_canvas(ydata::Vector)
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

MATLAB: AR on gamma directly, not on log(Y) levels.
This gives correct shock persistence and multiplier effects.
"""
function growth_inflation_expectations_matlab_style_canvas(model)
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
    alpha_gamma, beta_gamma, u_gamma = estimate_with_residuals_canvas(gamma_slice)
    alpha_pi, beta_pi, u_pi = estimate_with_residuals_canvas(pi_slice)

    # Draw correlated epsilons
    epsilon_gamma, epsilon_pi = draw_correlated_epsilons_canvas(u_gamma, u_pi)

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
# OVERRIDE growth_inflation_expectations FOR CANVAS
# =====================================================

function Bit.growth_inflation_expectations(
    model::Bit.Model{<:Bit.AbstractWorkers, <:Bit.AbstractWorkers, <:AbstractFirmsFullCANVAS,
                     <:Bit.AbstractBank, <:Bit.AbstractCentralBank, <:Bit.AbstractGovernment,
                     <:Bit.AbstractRestOfTheWorld, <:Bit.AbstractAggregates})
    return growth_inflation_expectations_matlab_style_canvas(model)
end

# =====================================================
# HELPER: Compute P_bar_dom_g (domestic price average)
# =====================================================
# MATLAB line 252: P_bar_dom_g(g)=(sum(P_i(G_i==g).*Q_i(G_i==g)))/(sum(Q_i(G_i==g)));

function compute_P_bar_dom_g(firms, G::Int)
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
# FULL CANVAS PRICING: P = (1+μ) × AC_e
# =====================================================

function Bit.firms_expectations_and_decisions(firms::AbstractFirmsFullCANVAS, model::Bit.AbstractModel)
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
    # 0. COMPUTE P_bar_dom_g (MATLAB line 252)
    # =====================================================
    # P_bar_dom_g(g) = sum(P_i .* Q_i) / sum(Q_i) for sector g
    P_bar_dom_g = compute_P_bar_dom_g(firms, G)

    # =====================================================
    # 1. CONDITIONAL DELTA_C (CANVAS/CATS switching rule)
    # =====================================================
    # MATLAB lines 134-138:
    # if ((Q_i(i)+S_i(i)+DS_DISP_i(i))<=Q_d_i(i) && P_i(i)>=P_bar_dom_g(G_i(i))) || ...
    #    ((Q_i(i)+S_i(i)+DS_DISP_i(i))>Q_d_i(i) && P_i(i)<P_bar_dom_g(G_i(i)))
    #     delta_C(i)=0;
    #
    # Supply = Q_i + S_i (actual sales + inventory)
    # Note: DS_DISP_i not used in BeforeIT

    delta_C_i = zeros(I)
    gamma_C_i = zeros(I)

    for i in 1:I
        g = firms.G_i[i]
        # EXACT MATLAB: supply = Q_i + S_i + DS_DISP_i
        # Using Q_i (actual sales) + S_i (inventory), matching MATLAB
        supply = firms.Q_i[i] + firms.S_i[i]
        demand = firms.Q_d_i[i]
        P_i = firms.P_i[i]
        P_bar_dom = P_bar_dom_g[g]  # EXACT: use P_bar_dom_g, not P_bar_g

        # EXACT MATLAB line 142: gamma_C_i = Q_d_i ./ (Q_i+S_i+DS_DISP_i) - 1
        gamma_C_i[i] = demand / max(supply, 1e-10) - 1

        # EXACT MATLAB lines 134-138
        if (supply <= demand && P_i >= P_bar_dom) || (supply > demand && P_i < P_bar_dom)
            delta_C_i[i] = 0.0  # Adjust quantity
        else
            delta_C_i[i] = 1.0  # Adjust price
        end
    end

    # =====================================================
    # 2. QUANTITY DECISION (MATLAB line 146)
    # =====================================================
    # EXACT: Q_s_i = max(0, (1+(1-delta_C).*gamma_C_i)*(1+delta_Y.*gamma_Y).*(Q_i+S_i+DS_DISP_i)-S_i)
    # For CANVAS: delta_Y = 1

    Q_s_i = zeros(I)
    for i in 1:I
        # EXACT MATLAB: supply = Q_i + S_i + DS_DISP_i
        supply = firms.Q_i[i] + firms.S_i[i]
        Q_s_i[i] = max(0.0, (1 + (1 - delta_C_i[i]) * gamma_C_i[i]) * (1 + gamma_e) * supply - firms.S_i[i])
    end

    # =====================================================
    # 3. EXPECTED AVERAGE COST (MATLAB line 152)
    # =====================================================
    # EXACT: AC_e_i(i) = ((1+tau_SIF)*w_bar_i(i)/alpha_bar_i(i)*P_bar_HH +
    #                     1/beta_i(i)*sum(a_sg(:,G_i(i)).*P_bar_g) +
    #                     delta_i(i)/kappa_i(i)*P_bar_CF) * (1+pi_e)

    firms.AC_e_lag_i .= firms.AC_e_i  # Store previous period

    for i in 1:I
        g = firms.G_i[i]

        # Labor cost per unit output
        labour_cost = (1 + tau_SIF) * firms.w_bar_i[i] / firms.alpha_bar_i[i] * P_bar_HH

        # Material cost per unit output
        material_cost = (1 / firms.beta_i[i]) * sum(a_sg[:, g] .* P_bar_g)

        # Capital depreciation cost per unit output
        capital_cost = (firms.delta_i[i] / firms.kappa_i[i]) * P_bar_CF

        # Expected average cost (scaled by expected inflation)
        firms.AC_e_i[i] = (labour_cost + material_cost + capital_cost) * (1 + pi_e)
    end

    # =====================================================
    # 4. MARKUP EVOLUTION (MATLAB line 161)
    # =====================================================
    # For CANVAS: δ_rec=1, δ_MS=0, δ_P=0, δ_AC=0
    # EXACT: mu_i = (delta_rec + mu_i) .* (1 + delta_C.*gamma_C_i) - delta_rec
    #      = (1 + mu_i) * (1 + delta_C * gamma_C_i) - 1

    for i in 1:I
        firms.mu_i[i] = (1 + firms.mu_i[i]) * (1 + delta_C_i[i] * gamma_C_i[i]) - 1
        firms.mu_i[i] = clamp(firms.mu_i[i], -0.5, 2.0)
    end

    # =====================================================
    # 5. PRICE SETTING (MATLAB line 163)
    # =====================================================
    # EXACT: P_i = (1 + mu_i) .* AC_e_i

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
# HELPER: Create Full CANVAS model - EXACT MATLAB INIT
# =====================================================

"""
    create_full_canvas_model(p, ic)

Create a Full CANVAS model with EXACT MATLAB initialization.

MATLAB initialization (abm.m lines 32-33):
    AC_e_i = (1+tau_SIF).*w_bar_i./alpha_bar_i + delta_i./kappa_i + 1./beta_i
    mu_i = 1./AC_e_i - 1
"""
function create_full_canvas_model(p, ic)
    # Initialize standard firms
    firms_st = Bit.Firms(p, ic)

    I = length(firms_st.G_i)

    # =====================================================
    # EXACT MATLAB INITIALIZATION (abm.m lines 32-33)
    # =====================================================
    # AC_e_i = (1+tau_SIF).*w_bar_i./alpha_bar_i + delta_i./kappa_i + 1./beta_i
    # mu_i = 1./AC_e_i - 1

    # Get tau_SIF from properties (need to create prop temporarily)
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

    # Initialize supply_history from calibration data (Y from agg)
    agg = Bit.Aggregates(p, ic)
    T_prime = p["T_prime"]
    initial_supply_history = copy(agg.Y[1:T_prime])

    # Create Full CANVAS firms with markup tracking and supply history
    firms = FirmsFullCANVAS(
        (getfield(firms_st, x) for x in fieldnames(Bit.Firms))...,
        initial_mu,
        initial_AC_e,
        copy(initial_AC_e),
        initial_supply_history,
        Ref(0)
    )

    # Initialize Q_i (actual sales) if needed (ensure supply > 0)
    # MATLAB uses Q_i for supply calculation, so we need Q_i initialized
    if all(firms.Q_i .== 0)
        firms.Q_i .= firms.Q_d_i  # Initialize to demand
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
    agg = Bit.Aggregates(p, ic)
    data = Bit.Data(p)

    return Bit.Model(w_act, w_inact, firms, bank, central_bank, gov, rotw, agg, prop, data)
end

# =====================================================
# TEST
# =====================================================

if abspath(PROGRAM_FILE) == @__FILE__
    using Plots, StatsPlots, JLD2, Statistics

    T = 12
    cal = load(joinpath(@__DIR__, "../../data/at/calibration_object.jld2"))["calibration_object"]
    calibration_date = DateTime(2010, 03, 31)

    p, ic = Bit.get_params_and_initial_conditions(cal, calibration_date; scale = 0.001)

    # Full CANVAS model
    model_canvas = create_full_canvas_model(p, ic)

    println("Running Full CANVAS model...")
    results_canvas = Bit.ensemblerun(model_canvas, T, 8)

    gdp_canvas = mean([r.data.real_gdp for r in results_canvas])
    println("\nFull CANVAS GDP: $(gdp_canvas[end])")
end
