"""
=====================================================================
FULL CANVAS MODEL IMPLEMENTATION (Wage-Price-Spiral Compatible)
=====================================================================

This file demonstrates how to extend BeforeIT.jl using Julia's multiple
dispatch (method overloading) WITHOUT modifying the base library source files.

## THREE PRICING IMPLEMENTATIONS IN BeforeIT.jl

| Implementation      | File                      | Description                        |
|---------------------|---------------------------|------------------------------------|
| Original (Poledna)  | Base BeforeIT.jl          | Multiplicative pricing             |
| Simplified CANVAS   | CANVAS_extension.jl       | Demand-pull with Q_s_i, P_bar_g    |
| Full CANVAS         | CANVAS_extension_full.jl  | Wage-price-spiral compatible       |

## HOW METHOD OVERLOADING WORKS

Julia's multiple dispatch allows us to define specialized behavior for
new types without modifying the original code:

1. Define abstract subtypes:
   `abstract type AbstractFirmsCANVAS <: Bit.AbstractFirms end`

2. Create extended structs with additional fields:
   `Bit.@object mutable struct FirmsCANVAS(Bit.Firms) <: AbstractFirmsCANVAS`

3. Overload methods for the new types:
   `function Bit.firms_expectations_and_decisions(firms::AbstractFirmsCANVAS, model)`

When the simulation runs with CANVAS types, Julia automatically dispatches
to our specialized methods instead of the base implementations.

## FULL CANVAS MODEL (Hommes et al., 2025)

This implementation includes:
1) Markup-based pricing: P = (1 + μ) × AC_e (not multiplicative)
2) Wage price index (P_bar_W) for labour cost calculation
3) Domestic sector price index (P_bar_g_dom) for demand-pull logic
4) Offered quantity (Q_o_i) = Y_i + S_i at period start
5) Dynamic markup adjustment based on demand conditions
6) Adaptive learning for central bank Taylor rule parameters
7) Configurable phi parameters for pass-through effects

## KEY DIFFERENCES FROM POLEDNA ET AL. (2023)

| Aspect         | Poledna (2023)                    | CANVAS (2025)                       |
|----------------|-----------------------------------|-------------------------------------|
| Pricing        | P × (1 + π_c) × (1 + π_e)         | P = (1 + μ) × AC_e                  |
| Wage costs     | Uses P_bar_HH (consumer CPI)      | Uses P_bar_W (wage price index)     |
| Demand-pull    | Uses P_bar_g (includes imports)   | Uses P_bar_g_dom (domestic only)    |
| Quantity base  | Q_s_i (supply)                    | Q_o_i = Y_i + S_i (offered)         |
| Cost returns   | Relative inflation                | Absolute costs                      |

Reference: Bank of Canada Staff Working Paper 2022-51
"""

import BeforeIT as Bit
using Dates

# =====================================================
# CANVAS PHI PARAMETERS (configurable defaults)
# =====================================================
# These control the pass-through of various effects
const DEFAULT_PHI_DP = 1.0      # Demand-pull effect on markup
const DEFAULT_PHI_CP = 1.0      # Cost-push effect on markup
const DEFAULT_PHI_AE = 1.0      # Anticipated effect on expected average cost
const DEFAULT_PHI_DP_Q = 1.0    # Demand-pull effect on quantity
const DEFAULT_PHI_G_Q = 1.0     # Growth expectation effect on quantity
const DEFAULT_PHI_UNION = 1.0   # Union wage indexation to inflation

# =====================================================
# CANVAS AGENT DEFINITIONS
# =====================================================

# Abstract types for CANVAS agents
abstract type AbstractFirmsCANVAS <: Bit.AbstractFirms end
abstract type AbstractCentralBankCANVAS <: Bit.AbstractCentralBank end
abstract type AbstractRestOfTheWorldCANVAS <: Bit.AbstractRestOfTheWorld end
abstract type AbstractAggregatesCANVAS <: Bit.AbstractAggregates end

# Extended Firms struct with CANVAS-specific fields
Bit.@object mutable struct FirmsCANVAS(Bit.Firms) <: AbstractFirmsCANVAS
    Q_o_i::Vector{Float64}              # Offered quantity (base for demand-pull)
    AC_i::Vector{Float64}               # Current average cost
    AC_e_i::Vector{Float64}             # Expected average cost
    mu_i::Vector{Float64}               # Markup rate
    labour_costs_i::Vector{Float64}     # Labour cost component
    material_costs_i::Vector{Float64}   # Material cost component
    capital_costs_i::Vector{Float64}    # Capital cost component
end

# Extended CentralBank for adaptive Taylor rule learning
Bit.@object mutable struct CentralBankCANVAS(Bit.CentralBank) <: AbstractCentralBankCANVAS
    r_bar_series::Vector{Float64}
end

# Extended RestOfTheWorld for tracking EA series and P_bar_W
Bit.@object mutable struct RestOfTheWorldCANVAS(Bit.RestOfTheWorld) <: AbstractRestOfTheWorldCANVAS
    Y_EA_series::Vector{Float64}
    pi_EA_series::Vector{Float64}
    P_bar_W::Float64                    # Wage price index (CANVAS addition)
end

# Extended Aggregates for P_bar_g_dom
Bit.@object mutable struct AggregatesCANVAS(Bit.Aggregates) <: AbstractAggregatesCANVAS
    P_bar_g_dom::Vector{Float64}        # Domestic sector price index
end

# =====================================================
# HELPER: Compute domestic sector price index
# (excludes imports, used for demand-pull logic)
# =====================================================
function compute_P_bar_g_dom(firms::AbstractFirmsCANVAS, G::Integer)
    P_bar_g_dom = zeros(Float64, G)
    for g in 1:G
        sector_mask = firms.G_i .== g
        sector_P = firms.P_i[sector_mask]
        sector_Q = firms.Q_i[sector_mask]
        total_quantity = sum(sector_Q)
        if total_quantity > 0
            P_bar_g_dom[g] = sum(sector_P .* sector_Q) / total_quantity
        else
            P_bar_g_dom[g] = 1.0  # Default if no production
        end
    end
    return P_bar_g_dom
end

# =====================================================
# CANVAS COST CALCULATION
# Returns absolute average costs using P_bar_W for wages
# =====================================================
function canvas_cost_push_inflation(firms::AbstractFirmsCANVAS, model::Bit.AbstractModel)
    # Unpack model variables
    P_bar_W = model.rotw.P_bar_W        # Wage price index (CANVAS)
    P_bar_CF = model.agg.P_bar_CF
    P_bar_g = model.agg.P_bar_g
    tau_SIF = model.prop.tau_SIF
    a_sg = model.prop.a_sg

    # Compute material price index for each firm's sector
    term = dropdims(sum(a_sg[:, firms.G_i] .* P_bar_g, dims=1), dims=1)

    # CANVAS: Compute ABSOLUTE cost components
    # KEY FIX: Use P_bar_W (wage price index) for labour costs, not P_bar_HH
    labour_costs = (1 + tau_SIF) .* firms.w_bar_i ./ firms.alpha_bar_i .* P_bar_W
    material_costs = 1.0 ./ firms.beta_i .* term
    capital_costs = firms.delta_i ./ firms.kappa_i .* P_bar_CF
    average_cost = labour_costs .+ material_costs .+ capital_costs

    return average_cost, labour_costs, material_costs, capital_costs
end

# =====================================================
# CANVAS FIRM EXPECTATIONS AND DECISIONS
# Implements markup-based pricing and demand-pull logic
# =====================================================
function Bit.firms_expectations_and_decisions(firms::AbstractFirmsCANVAS, model::Bit.AbstractModel)
    # Unpack model variables
    gamma_e = model.agg.gamma_e
    pi_e = model.agg.pi_e
    G = model.prop.G

    # Get phi parameters (use defaults if not in model.prop)
    phi_DP = hasproperty(model.prop, :phi_DP) ? model.prop.phi_DP : DEFAULT_PHI_DP
    phi_CP = hasproperty(model.prop, :phi_CP) ? model.prop.phi_CP : DEFAULT_PHI_CP
    phi_AE = hasproperty(model.prop, :phi_AE) ? model.prop.phi_AE : DEFAULT_PHI_AE
    phi_DP_Q = hasproperty(model.prop, :phi_DP_Q) ? model.prop.phi_DP_Q : DEFAULT_PHI_DP_Q
    phi_G_Q = hasproperty(model.prop, :phi_G_Q) ? model.prop.phi_G_Q : DEFAULT_PHI_G_Q

    # =====================================================
    # KEY FIX #1: Update Q_o_i at START of period
    # Q_o_i = Y_i + S_i (production + inventory from previous period)
    # This is the correct timing - after previous period's production
    # =====================================================
    firms.Q_o_i .= firms.Y_i .+ firms.S_i

    # =====================================================
    # KEY FIX #2: Compute domestic sector price index
    # Use P_bar_g_dom (domestic only) for demand-pull logic
    # =====================================================
    P_bar_g_dom = compute_P_bar_g_dom(firms, G)

    # Number of firms
    I = length(firms.G_i)
    gamma_d_i = zeros(I)
    pi_d_i = zeros(I)

    # =====================================================
    # DEMAND-PULL LOGIC using Q_o_i (offered quantity)
    # Firms decide whether to adjust price OR quantity
    # based on supply-demand imbalance and price vs DOMESTIC sector average
    # =====================================================
    for i in 1:I
        Q_avail = firms.Q_o_i[i]

        # Protect against division by zero
        if Q_avail <= 0
            Q_avail = max(Q_avail, 1e-10)
        end

        # KEY FIX #3: Compare to P_bar_g_dom (domestic), not P_bar_g
        # Case A: Adjust QUANTITY (not price)
        # - Excess demand AND price above sector average, OR
        # - Excess supply AND price below sector average
        if (Q_avail <= firms.Q_d_i[i] && firms.P_i[i] >= P_bar_g_dom[firms.G_i[i]]) ||
           (Q_avail > firms.Q_d_i[i] && firms.P_i[i] < P_bar_g_dom[firms.G_i[i]])
            gamma_d_i[i] = firms.Q_d_i[i] / Q_avail - 1
            pi_d_i[i] = 0.0
        # Case B: Adjust PRICE (not quantity)
        # - Excess demand AND price below sector average, OR
        # - Excess supply AND price above sector average
        else
            gamma_d_i[i] = 0.0
            pi_d_i[i] = firms.Q_d_i[i] / Q_avail - 1
        end
    end

    # =====================================================
    # CANVAS QUANTITY SETTING
    # Q_s = max(0, Q_o × (1 + φ_G_Q × γ_e) × (1 + φ_DP_Q × γ_d) - S)
    # =====================================================
    Q_s_i = max.(0, firms.Q_o_i .* (1 .+ phi_G_Q * gamma_e) .* (1 .+ phi_DP_Q .* gamma_d_i) .- firms.S_i)

    # =====================================================
    # CANVAS PRICING: Markup × Expected Average Cost
    # =====================================================

    # Get absolute average costs (using P_bar_W for wages)
    AC_i, labour_costs_i, material_costs_i, capital_costs_i = canvas_cost_push_inflation(firms, model)

    # Cost-push inflation from change in average cost
    # Protect against division by zero for expected AC
    AC_e_safe = max.(firms.AC_e_i, 1e-10)
    pi_c_i = (AC_i .- firms.AC_e_i) ./ AC_e_safe

    # Dynamic markup adjustment based on demand conditions
    # μ_new = (1 + μ) × (1 + φ_DP × π_d) / (1 + (1 - φ_CP) × π_c) - 1
    denominator = 1 .+ (1 - phi_CP) .* pi_c_i
    denominator = max.(denominator, 0.1)  # Prevent extreme values
    mu_new_i = (1 .+ firms.mu_i) .* (1 .+ phi_DP .* pi_d_i) ./ denominator .- 1


    # Expected average cost with inflation anticipation
    # AC_e = AC × (1 + φ_AE × π_e)
    AC_e_new_i = AC_i .* (1 .+ phi_AE .* pi_e)

    # CANVAS pricing: P = (1 + markup) × expected_average_cost
    # Uses OLD markup (firms.mu_i) for current price, then update markup for next period
    new_P_i = (1 .+ firms.mu_i) .* AC_e_new_i

    # Ensure positive prices
    new_P_i = max.(new_P_i, 1e-10)

    # =====================================================
    # UPDATE FIRM STATE for next period
    # NOTE: Q_o_i is NOT updated here anymore - it's updated at START of next period
    # =====================================================
    firms.AC_i .= AC_i
    firms.AC_e_i .= AC_e_new_i
    firms.mu_i .= mu_new_i
    firms.labour_costs_i .= labour_costs_i
    firms.material_costs_i .= material_costs_i
    firms.capital_costs_i .= capital_costs_i

    # =====================================================
    # STANDARD DOWNSTREAM CALCULATIONS
    # =====================================================
    # Target investments, materials, employment
    I_d_i, DM_d_i, N_d_i = Bit.desired_capital_material_employment(firms, Q_s_i)

    # Expected profits
    Pi_e_i = firms.Pi_i .* (1 + pi_e) * (1 + gamma_e)

    # Expected deposits, capital, loans
    DD_e_i, K_e_i, L_e_i = Bit.expected_deposits_capital_loans(firms, model, Pi_e_i)

    # Target loans
    DL_d_i = max.(0, -DD_e_i - firms.D_i)

    return Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, new_P_i
end

# =====================================================
# ADAPTIVE CENTRAL BANK TAYLOR RULE LEARNING
# =====================================================
function Bit.central_bank_rate(cb::AbstractCentralBankCANVAS, model::Bit.AbstractModel)
    # Unpack arguments
    gamma_EA = model.rotw.gamma_EA
    pi_EA = model.rotw.pi_EA
    T_prime = model.prop.T_prime
    t = model.agg.t

    # Historical series for adaptive learning
    a1 = cb.r_bar_series[1:(T_prime + t - 1)]
    a2 = model.rotw.Y_EA_series[1:(T_prime + t - 1)]
    a3 = model.rotw.pi_EA_series[1:(T_prime + t - 1)]

    # Update central bank parameters via adaptive learning
    rho, r_star, xi_pi, xi_gamma, pi_star = Bit.estimate_taylor_rule(a1, a2, a3)
    model.cb.rho = rho
    model.cb.r_star = r_star
    model.cb.xi_pi = xi_pi
    model.cb.xi_gamma = xi_gamma
    model.cb.pi_star = pi_star

    # Compute new policy rate using Taylor rule
    r_bar = Bit.taylor_rule(cb.rho, cb.r_bar, cb.r_star, cb.pi_star, cb.xi_pi, cb.xi_gamma, gamma_EA, pi_EA)

    # Store for next period's learning
    cb.r_bar_series[T_prime + t] = r_bar

    return r_bar
end

# =====================================================
# REST OF THE WORLD: Track EA series for CB learning
# AND update P_bar_W (wage price index)
# =====================================================
function Bit.growth_inflation_EA(rotw::AbstractRestOfTheWorldCANVAS, model::Bit.AbstractModel)
    # Unpack model variables
    epsilon_Y_EA = model.agg.epsilon_Y_EA
    T_prime = model.prop.T_prime
    t = model.agg.t

    # Compute EA GDP and growth
    Y_EA = exp(rotw.alpha_Y_EA * log(rotw.Y_EA) + rotw.beta_Y_EA + epsilon_Y_EA)
    gamma_EA = Y_EA / rotw.Y_EA - 1

    # Compute EA inflation
    epsilon_pi_EA = randn() * rotw.sigma_pi_EA
    pi_EA = exp(rotw.alpha_pi_EA * log(1 + rotw.pi_EA) + rotw.beta_pi_EA + epsilon_pi_EA) - 1

    # Store in series for CB learning
    rotw.Y_EA_series[T_prime + t] = Y_EA
    rotw.pi_EA_series[T_prime + t] = pi_EA

    # =====================================================
    # KEY FIX #4: Update P_bar_W (wage price index)
    # P_bar_W = (1 + φ_UNION × max(π, 0)) × P_bar_W
    # Wages are indexed to inflation with union pass-through
    # =====================================================
    phi_UNION = hasproperty(model.prop, :phi_UNION) ? model.prop.phi_UNION : DEFAULT_PHI_UNION

    # Get current inflation from aggregates
    current_pi = t > 1 ? model.agg.pi_[T_prime + t - 1] : 0.0

    # Update wage price index
    rotw.P_bar_W = (1 + phi_UNION * max(current_pi, 0.0)) * rotw.P_bar_W

    return Y_EA, gamma_EA, pi_EA
end

# =====================================================
# MAIN SCRIPT
# =====================================================

# Simulation parameters
T = 12
cal = Bit.ITALY_CALIBRATION
calibration_date = DateTime(2010, 03, 31)

# Get parameters and initial conditions
p, ic = Bit.get_params_and_initial_conditions(cal, calibration_date; scale = 0.001)

# Expand series with T new time steps for CB learning
Y_EA_series = vec(vcat(ic["Y_EA_series"], zeros(Float64, T)))
pi_EA_series = vec(vcat(ic["pi_EA_series"], zeros(Float64, T)))
r_bar_series = vec(vcat(ic["r_bar_series"], zeros(Float64, T)))

# =====================================================
# INITIALIZE CANVAS FIRMS
# Following wage-price-spiral initialization
# =====================================================
firms_st = Bit.Firms(p, ic)
I = length(firms_st.G_i)

# Get tax rate for labour cost calculation
tau_SIF = p["tau_SIF"]

# Initialize CANVAS-specific fields using correct formulas from wage-price-spiral
# Offered quantity = production (Y_i) - will be updated to Y_i + S_i at start of each period
Q_o_i = copy(firms_st.Y_i)

# Average cost = labour unit cost + capital unit cost + material unit cost
# AC = (1+tau_SIF) * w_bar / alpha_bar + delta / kappa + 1 / beta
# NOTE: At initialization, P_bar_W = 1.0, so this is correct
labour_costs_i = (1 + tau_SIF) .* firms_st.w_bar_i ./ firms_st.alpha_bar_i
material_costs_i = 1.0 ./ firms_st.beta_i
capital_costs_i = firms_st.delta_i ./ firms_st.kappa_i
AC_i = labour_costs_i .+ material_costs_i .+ capital_costs_i

# Expected AC = current AC initially
AC_e_i = copy(AC_i)

# Markup derived from P = (1+mu)*AC, so mu = 1/AC - 1 (since P_i = 1.0)
# Allow negative markups for high-cost firms - this is mathematically necessary
mu_i = 1.0 ./ AC_i .- 1

# Create CANVAS firms with extended fields
firms = FirmsCANVAS(
    (getfield(firms_st, x) for x in fieldnames(Bit.Firms))...,
    Q_o_i, AC_i, AC_e_i, mu_i, labour_costs_i, material_costs_i, capital_costs_i
)

# =====================================================
# INITIALIZE CANVAS CENTRAL BANK
# =====================================================
central_bank_st = Bit.CentralBank(p, ic)
central_bank = CentralBankCANVAS(
    (getfield(central_bank_st, x) for x in fieldnames(Bit.CentralBank))...,
    r_bar_series
)

# =====================================================
# INITIALIZE CANVAS REST OF THE WORLD
# With P_bar_W (wage price index) starting at 1.0
# =====================================================
rotw_st = Bit.RestOfTheWorld(p, ic)
P_bar_W_init = 1.0  # Initial wage price index
rotw = RestOfTheWorldCANVAS(
    (getfield(rotw_st, x) for x in fieldnames(Bit.RestOfTheWorld))...,
    Y_EA_series, pi_EA_series, P_bar_W_init
)

# =====================================================
# STANDARD AGENT INITIALIZATIONS
# =====================================================
w_act, w_inact = Bit.Workers(p, ic)
bank = Bit.Bank(p, ic)
agg = Bit.Aggregates(p, ic)
gov = Bit.Government(p, ic)
prop = Bit.Properties(p, ic)
data = Bit.Data(p)

# =====================================================
# CREATE MODELS
# =====================================================
# Standard model (Poledna et al. 2023)
model_std = Bit.Model(w_act, w_inact, firms_st, bank, central_bank_st, gov, rotw_st, agg, prop, data)

# CANVAS model (Hommes et al. 2025)
model_canvas = Bit.Model(w_act, w_inact, firms, bank, central_bank, gov, rotw, agg, prop, data)

# =====================================================
# RUN SIMULATIONS
# =====================================================
println("Running standard model (Poledna et al. 2023)...")
model_vector_std = Bit.ensemblerun(model_std, T, 8)

println("Running CANVAS model (Hommes et al. 2025)...")
println("  - Using P_bar_W for wage costs (initially 1.0)")
println("  - Using P_bar_g_dom for demand-pull logic")
println("  - Q_o_i updated at start of each period as Y_i + S_i")
println("  - Initial markup range: $(extrema(mu_i))")
println("  - Initial AC range: $(extrema(AC_i))")
model_vector_canvas = Bit.ensemblerun(model_canvas, T, 8)

# =====================================================
# PLOT RESULTS
# =====================================================
using Plots, StatsPlots
println("Plotting comparison...")
ps = Bit.plot_data_vectors([model_vector_std, model_vector_canvas])
p_final = plot(ps..., layout = (3, 3), size=(1200, 900),
     plot_title="Blue=Poledna 2023 | Orange=CANVAS 2025 (fixed)")

# Save the plot
savefig(p_final, "examples/canvas_comparison_plot.png")
println("Plot saved to examples/canvas_comparison_plot.png")

println("\nDone! Fixed CANVAS implementation includes:")
println("  ✓ P_bar_W (wage price index) for labour costs")
println("  ✓ P_bar_g_dom (domestic sector prices) for demand-pull")
println("  ✓ Correct Q_o_i update timing (Y_i + S_i at period start)")
println("  ✓ Configurable phi parameters with defaults")
