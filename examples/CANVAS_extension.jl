"""
--------- CANVAS model by overwriting ---------

There are 4 major changes from the Poledna et al. (2023) to the CANVAS model (Hommes et al., 2025)

1) Increased heterogeneity with respect to consumer behaviour and initiasation
2) Increased heterogeniety with respect to firm initialisation
3) Demand pull firm level price and quanitity setting
4) Adaptive learning for the central bank to learn the parameters of the Taylor rule

This script implements changes 3 and 4 by overwriting the methods that govern that behaviour. 
To introduce changes 1 and 2 we need dissagregated data at the household and firm level.
"""

import BeforeIT as Bit

# define a new central bank for the CANVAS model
abstract type AbstractCentralBankCANVAS <: Bit.AbstractCentralBank end
Bit.@object mutable struct CentralBankCANVAS(Bit.CentralBank) <: AbstractCentralBankCANVAS
    r_bar_series::Vector{Float64}
end

# define new firms for the CANVAS model
abstract type AbstractFirmsCANVAS <: Bit.AbstractFirms end
Bit.@object struct FirmsCANVAS(Firms) <: AbstractFirmsCANVAS
    delta_S_i::Vector{Float64}   # inventory depreciation rate (paper Eq. 32)
end

# define a new rest of the world for the CANVAS model
abstract type AbstractRestOfTheWorldCANVAS <: Bit.AbstractRestOfTheWorld end
Bit.@object mutable struct RestOfTheWorldCANVAS(Bit.RestOfTheWorld) <: AbstractRestOfTheWorldCANVAS
    Y_EA_series::Vector{Float64}
    pi_EA_series::Vector{Float64}
end

# CANVAS paper Eq. 15a-15b: AR(1) on growth rates (not log-levels)
# γ^e(t) = exp(α^γ · γ(t-1) + β^γ + ε^γ) - 1
# π^e(t) = exp(α^π · π(t-1) + β^π + ε^π) - 1
function growth_inflation_expectations_canvas(model)
    Y = model.agg.Y
    pi = model.agg.pi_
    T_prime = model.prop.T_prime
    t = model.agg.t

    Y_slice = Y[1:(T_prime + t - 1)]
    pi_slice = pi[1:(T_prime + t - 1)]

    # Need enough data for AR(1) on growth rates (diff reduces length by 1)
    if length(Y_slice) < 4 || length(pi_slice) < 3
        # Fallback: inline base model logic (cannot call Bit.growth_inflation_expectations
        # because the dispatch override would cause infinite recursion)
        lY_e = Bit.estimate_next_value(log.(Y_slice))
        Y_e = exp(lY_e)
        gamma_e = Y_e / Y_slice[end] - 1
        lpi = Bit.estimate_next_value(pi_slice)
        pi_e = exp(lpi) - 1
        return Y_e, gamma_e, pi_e
    end

    # Paper Eq. 15a: AR(1) on growth rates γ (log first differences of real GDP)
    gamma_slice = diff(log.(Y_slice))
    gamma_hat = Bit.estimate_next_value(gamma_slice)
    gamma_e = exp(gamma_hat) - 1
    Y_e = Y_slice[end] * (1 + gamma_e)

    # Paper Eq. 15b: AR(1) on inflation π (log first difference of GDP deflator)
    pi_hat = Bit.estimate_next_value(pi_slice)
    pi_e = exp(pi_hat) - 1

    return Y_e, gamma_e, pi_e
end

function Bit.growth_inflation_expectations(
    model::Bit.Model{<:Bit.AbstractWorkers, <:Bit.AbstractWorkers, <:AbstractFirmsCANVAS,
                     <:Bit.AbstractBank, <:Bit.AbstractCentralBank, <:Bit.AbstractGovernment,
                     <:Bit.AbstractRestOfTheWorld, <:Bit.AbstractAggregates})
    return growth_inflation_expectations_canvas(model)
end

# define new functions for the CANVAS-specific agents
function Bit.firms_expectations_and_decisions(firms::AbstractFirmsCANVAS, model::Bit.AbstractModel)
    # unpack non-firm variables
    P_bar_g = model.agg.P_bar_g
    gamma_e = model.agg.gamma_e
    pi_e = model.agg.pi_e

    # CANVAS switching rule: conditional delta_C per firm
    I = length(firms.G_i)
    delta_C_i = zeros(I)
    gamma_C_i = zeros(I)

    for i in 1:I
        g = firms.G_i[i]
        supply = firms.Q_i[i] + firms.S_i[i] + firms.DS_i[i]
        demand = firms.Q_d_i[i]

        # gamma_C = demand/supply ratio - 1
        gamma_C_i[i] = demand / max(supply, 1e-10) - 1

        # Switching: quantity adjustment (delta_C=0) vs price adjustment (delta_C=1)
        if (supply <= demand && firms.P_i[i] >= P_bar_g[g]) || (supply > demand && firms.P_i[i] < P_bar_g[g])
            delta_C_i[i] = 0.0  # adjust quantity
        else
            delta_C_i[i] = 1.0  # adjust price (markup)
        end
    end

    # Production target: desired total supply minus existing inventory
    Q_s_i = zeros(I)
    for i in 1:I
        supply = firms.Q_i[i] + firms.S_i[i] + firms.DS_i[i]
        Q_s_i[i] = max(0.0, (1 + (1 - delta_C_i[i]) * gamma_C_i[i]) * (1 + gamma_e) * supply - firms.S_i[i])
    end

    # cost push inflation
    pi_c_i = Bit.cost_push_inflation(firms, model)
    # demand-pull inflation: only when delta_C=1 (price adjustment mode)
    pi_d_i = delta_C_i .* gamma_C_i
    # price setting
    new_P_i = firms.P_i .* (1 .+ pi_c_i) .* (1 + pi_e) .* (1 .+ pi_d_i)
    # target investments in capital, intermediate goods to purchase and employment
    I_d_i, DM_d_i, N_d_i = Bit.desired_capital_material_employment(firms, Q_s_i)
    # expected profits 
    Pi_e_i = firms.Pi_i .* (1 + pi_e) * (1 + gamma_e)
    # expected deposits, capital and loans
    DD_e_i, K_e_i, L_e_i = Bit.expected_deposits_capital_loans(firms, model, Pi_e_i)
    # target loans
    DL_d_i = max.(0, -DD_e_i - firms.D_i)

    return Q_s_i, I_d_i, DM_d_i, N_d_i, Pi_e_i, DL_d_i, K_e_i, L_e_i, new_P_i
end

# Inventory dynamics with delta_S=1 (MATLAB abm.m lines 262-264):
# DS_i = Y_i - Q_i  (one-period disposable stock)
# S_i unchanged (frozen at initial value, no accumulation)
function Bit.firms_stocks(firms::AbstractFirmsCANVAS)
    K_i = firms.K_i - firms.delta_i ./ firms.kappa_i .* firms.Y_i + firms.I_i
    M_i = firms.M_i - firms.Y_i ./ firms.beta_i + firms.DM_i
    DS_i = firms.Y_i - firms.Q_i
    S_i = firms.S_i            # frozen — delta_S = 1
    return K_i, M_i, DS_i, S_i
end

function Bit.central_bank_rate(cb::AbstractCentralBankCANVAS, model::Bit.AbstractModel)
    # unpack arguments
    gamma_EA = model.rotw.gamma_EA
    pi_EA = model.rotw.pi_EA
    T_prime = model.prop.T_prime
    t = model.agg.t

    a1 = cb.r_bar_series[1:(T_prime + t - 1)]
    a2 = model.rotw.Y_EA_series[1:(T_prime + t - 1)]
    a3 = model.rotw.pi_EA_series[1:(T_prime + t - 1)]

    # update central bank parameters
    rho, r_star, xi_pi, xi_gamma, pi_star = Bit.estimate_taylor_rule(a1, a2, a3)
    model.cb.rho = rho
    model.cb.r_star = r_star
    model.cb.xi_pi = xi_pi
    model.cb.xi_gamma = xi_gamma
    model.cb.pi_star = pi_star

    r_bar = Bit.taylor_rule(cb.rho, cb.r_bar, cb.r_star, cb.pi_star, cb.xi_pi, cb.xi_gamma, gamma_EA, pi_EA)

    cb.r_bar_series[T_prime + t] = r_bar
    return r_bar
end

function Bit.growth_inflation_EA(rotw::AbstractRestOfTheWorldCANVAS, model::Bit.AbstractModel)
    # unpack model variables
    epsilon_Y_EA = model.agg.epsilon_Y_EA
    T_prime = model.prop.T_prime
    t = model.agg.t

    Y_EA = exp(rotw.alpha_Y_EA * log(rotw.Y_EA) + rotw.beta_Y_EA + epsilon_Y_EA) # GDP EA
    gamma_EA = Y_EA / rotw.Y_EA - 1                                              # growht EA
    epsilon_pi_EA = randn() * rotw.sigma_pi_EA
    pi_EA = exp(rotw.alpha_pi_EA * log(1 + rotw.pi_EA) + rotw.beta_pi_EA + epsilon_pi_EA) - 1   # inflation EA

    rotw.pi_EA_series[T_prime + t] = pi_EA
    rotw.Y_EA_series[T_prime + t] = Y_EA

    return Y_EA, gamma_EA, pi_EA
end

# =====================================================
# FACTORY: Create CANVAS Model from parameters and initial conditions
# =====================================================

"""
    create_model(p, ic)

Create a CANVAS model from parameter and initial condition dictionaries.

This factory function is the standard interface for `save_all_simulations`:
    include("examples/CANVAS_extension.jl")
    Bit.save_all_simulations(folder; model_factory=create_model, output_suffix="canvas")

Creates FirmsCANVAS (with delta_S_i=0), CentralBankCANVAS (with r_bar_series),
and RestOfTheWorldCANVAS (with Y_EA_series, pi_EA_series).
"""
function create_model(p, ic)
    # Standard agent initialisation
    w_act, w_inact = Bit.Workers(p, ic)
    firms_st = Bit.Firms(p, ic)
    bank = Bit.Bank(p, ic)
    cb_st = Bit.CentralBank(p, ic)
    gov = Bit.Government(p, ic)
    rotw_st = Bit.RestOfTheWorld(p, ic)
    agg = Bit.Aggregates(p, ic)
    prop = Bit.Properties(p, ic)
    data = Bit.Data(p)

    I = length(firms_st.G_i)

    # CANVAS firms: add inventory depreciation field (delta_S = 0 by default)
    delta_S_i = zeros(I)
    firms = FirmsCANVAS((getfield(firms_st, x) for x in fieldnames(Bit.Firms))..., delta_S_i)

    # Initialize Q_i and Y_i from Q_d_i if they are all zeros (needed for supply calc)
    if all(firms.Q_i .== 0)
        firms.Q_i .= firms.Q_d_i
    end
    if all(firms.Y_i .== 0)
        firms.Y_i .= firms.Q_d_i
    end

    # CANVAS central bank: add r_bar_series (historical + buffer for forecast horizon)
    r_bar_hist = vec(ic["r_bar_series"])
    r_bar_series = vcat(r_bar_hist, zeros(Float64, 50))
    cb = CentralBankCANVAS((getfield(cb_st, x) for x in fieldnames(Bit.CentralBank))..., r_bar_series)

    # CANVAS rest of the world: add Y_EA_series and pi_EA_series
    Y_EA_series = vec(vcat(ic["Y_EA_series"], zeros(Float64, 50)))
    pi_EA_series = vec(vcat(ic["pi_EA_series"], zeros(Float64, 50)))
    rotw = RestOfTheWorldCANVAS((getfield(rotw_st, x) for x in fieldnames(Bit.RestOfTheWorld))...,
        Y_EA_series, pi_EA_series)

    return Bit.Model(w_act, w_inact, firms, bank, cb, gov, rotw, agg, prop, data)
end

# =====================================================
# DEMO: Only runs when this file is executed directly
# =====================================================

if abspath(PROGRAM_FILE) == @__FILE__

using Plots, StatsPlots, Dates

# get parameters and initial conditions
T = 12
cal = Bit.ITALY_CALIBRATION
calibration_date = DateTime(2010, 03, 31)

p, ic = Bit.get_params_and_initial_conditions(cal, calibration_date; scale = 0.001)

# expand series with T new time steps in the future
Y_EA_series = vec(vcat(ic["Y_EA_series"], zeros(Float64, T)))
pi_EA_series = vec(vcat(ic["pi_EA_series"], zeros(Float64, T)))
r_bar_series = vec(vcat(ic["r_bar_series"], zeros(Float64, T)))

# new firms initialisation
firms_st = Bit.Firms(p, ic)
delta_S_i = zeros(length(firms_st.G_i))  # default: no inventory depreciation (δ^S = 0)
firms = FirmsCANVAS((getfield(firms_st, x) for x in fieldnames(Bit.Firms))..., delta_S_i)

# new central bank initialisation
central_bank_st = Bit.CentralBank(p, ic)
central_bank = CentralBankCANVAS((getfield(central_bank_st, x) for x in fieldnames(Bit.CentralBank))...,
    r_bar_series) # add new variables to the aggregates

# new rotw initialisation
rotw_st = Bit.RestOfTheWorld(p, ic)
rotw = RestOfTheWorldCANVAS((getfield(rotw_st, x) for x in fieldnames(Bit.RestOfTheWorld))...,
    Y_EA_series, pi_EA_series) # add new variables to the aggregates

# standard initialisations: workers, bank, aggregats, government, properties and data
w_act, w_inact = Bit.Workers(p, ic)
bank = Bit.Bank(p, ic)
agg = Bit.Aggregates(p, ic)
gov = Bit.Government(p, ic)
prop = Bit.Properties(p, ic)
data = Bit.Data(p)

# define a standard model
model_std = Bit.Model(w_act, w_inact, firms_st, bank, central_bank_st, gov, rotw_st, agg, prop, data)

# define a CANVAS model
model_canvas = Bit.Model(w_act, w_inact, firms, bank, central_bank, gov, rotw, agg, prop, data)

# run the model(s)
model_vector_std = Bit.ensemblerun(model_std, T, 8)
model_vector_canvas = Bit.ensemblerun(model_canvas, T, 8)

# plot the results
ps = Bit.plot_data_vectors([model_vector_std, model_vector_canvas])
plot(ps..., layout = (3, 3))

end # if PROGRAM_FILE
