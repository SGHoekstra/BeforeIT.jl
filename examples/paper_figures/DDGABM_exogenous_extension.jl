"""
=====================================================================
DDGABM EXOGENOUS PROCESSES EXTENSION
=====================================================================

Implements MATLAB DDGABM's 6-dimensional correlated exogenous shocks
and separate price indices (P_G, P_E, P_I) for government consumption,
exports, and imports.

MATLAB source: DDGABM/model/simulate_abm.m lines 274-302
- 6D Cholesky: epsilon(C) draws (eps_gG, eps_pG, eps_gE, eps_pE, eps_gI, eps_pI)
- Growth-rate AR: gamma_G(t) = alpha_gamma_G * gamma_G(t-1) + beta_gamma_G + eps_gG
- Price index AR: pi_G(t) = alpha_pi_G * pi_G(t-1) + beta_pi_G + eps_pG
- Level update: C_G(t) = C_G(t-1) * exp(gamma_G(t-1))  [LAGGED gamma]
- Price update: P_G(t) = P_G(t-1) * exp(pi_G(t-1))     [LAGGED pi]

MATLAB source: DDGABM/model/abm.m lines 225, 227, 229-230
- C_d_j = P_G * C_G / J                        (gov consumption demand; P_G replaces price basket)
- C_d_l = P_E * C_E / L                        (export demand; P_E replaces price basket)
- P_m = ones(1,G) * P_I                        (import prices)

MATLAB source: DDGABM/model/epsilon.m
- Cholesky with `any(any(C~=0))` check (not isposdef)
"""

import BeforeIT as Bit
using Statistics, LinearAlgebra

# =====================================================
# 6D EPSILON FUNCTION (MATLAB epsilon.m)
# =====================================================

"""
    epsilon6(C::Matrix)

Draw 6 correlated shocks from joint distribution via Cholesky.
Order: (gamma_G, pi_G, gamma_E, pi_E, gamma_I, pi_I)

MATLAB (DDGABM/model/epsilon.m):
    if any(any(C~=0))
        L = chol(C, 'lower');
        z = randn(6,1);
        eps = L * z;
    else
        eps = zeros(6,1);
    end
"""
function epsilon6(C::Matrix)
    if !any(C .!= 0)
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    end
    L = cholesky(Symmetric(C)).L
    z = randn(6)
    eps = L * z
    return eps[1], eps[2], eps[3], eps[4], eps[5], eps[6]
end

# =====================================================
# EXTENDED GOVERNMENT TYPE
# =====================================================

abstract type AbstractGovernmentDDGABM <: Bit.AbstractGovernment end

Bit.@object mutable struct GovernmentDDGABM(Bit.Government) <: AbstractGovernmentDDGABM
    # Growth-rate AR params (MATLAB: alpha_gamma_G, beta_gamma_G)
    alpha_gamma_G::Bit.typeFloat
    beta_gamma_G::Bit.typeFloat
    # Deflator AR params (MATLAB: alpha_pi_G, beta_pi_G)
    alpha_pi_G::Bit.typeFloat
    beta_pi_G::Bit.typeFloat
    # Lagged growth/inflation rates
    g_prev_G::Bit.typeFloat
    pi_prev_G::Bit.typeFloat
    # Price index (MATLAB: P_G)
    P_G::Bit.typeFloat
    # 6x6 covariance matrix (shared with RotW)
    C6::Matrix{Bit.typeFloat}
    # Storage for last drawn epsilons (shared with RotW within same step)
    eps_gG::Ref{Bit.typeFloat}
    eps_pG::Ref{Bit.typeFloat}
    eps_gE::Ref{Bit.typeFloat}
    eps_pE::Ref{Bit.typeFloat}
    eps_gI::Ref{Bit.typeFloat}
    eps_pI::Ref{Bit.typeFloat}
    eps_drawn_t::Ref{Bit.typeInt}  # Track which step the eps were drawn for
end

# =====================================================
# EXTENDED REST OF THE WORLD TYPE
# =====================================================

abstract type AbstractRestOfTheWorldDDGABM <: Bit.AbstractRestOfTheWorld end

Bit.@object mutable struct RestOfTheWorldDDGABM(Bit.RestOfTheWorld) <: AbstractRestOfTheWorldDDGABM
    # Growth-rate AR params
    alpha_gamma_E::Bit.typeFloat
    beta_gamma_E::Bit.typeFloat
    alpha_gamma_I::Bit.typeFloat
    beta_gamma_I::Bit.typeFloat
    # Deflator AR params
    alpha_pi_E::Bit.typeFloat
    beta_pi_E::Bit.typeFloat
    alpha_pi_I::Bit.typeFloat
    beta_pi_I::Bit.typeFloat
    # Lagged growth/inflation rates
    g_prev_E::Bit.typeFloat
    pi_prev_E::Bit.typeFloat
    g_prev_I::Bit.typeFloat
    pi_prev_I::Bit.typeFloat
    # Price indices (MATLAB: P_E, P_I)
    P_E::Bit.typeFloat
    P_I::Bit.typeFloat
    # Reference to government's epsilon storage (shared 6D draw)
    gov_ref::Ref{GovernmentDDGABM}
end

# =====================================================
# HELPER: Draw or retrieve 6D epsilons for current step
# =====================================================

"""
    get_epsilons!(gov::GovernmentDDGABM, t::Int)

Draw 6D correlated epsilons if not already drawn for step t.
Returns (eps_gG, eps_pG, eps_gE, eps_pE, eps_gI, eps_pI).
"""
function get_epsilons!(gov::GovernmentDDGABM, t::Int)
    if gov.eps_drawn_t[] != t
        e1, e2, e3, e4, e5, e6 = epsilon6(gov.C6)
        gov.eps_gG[] = e1
        gov.eps_pG[] = e2
        gov.eps_gE[] = e3
        gov.eps_pE[] = e4
        gov.eps_gI[] = e5
        gov.eps_pI[] = e6
        gov.eps_drawn_t[] = t
    end
    return gov.eps_gG[], gov.eps_pG[], gov.eps_gE[], gov.eps_pE[], gov.eps_gI[], gov.eps_pI[]
end

# =====================================================
# OVERRIDE gov_social_benefits FOR DDGABM GOVERNMENT
# =====================================================
# MATLAB abm.m lines 197-198: social benefits update is COMMENTED OUT
# (sb stays constant at calibrated level — no growth by gamma_e)

function Bit.gov_social_benefits(gov::AbstractGovernmentDDGABM, model)
    return gov.sb_other, gov.sb_inact  # no growth
end

# =====================================================
# OVERRIDE gov_expenditure FOR DDGABM GOVERNMENT
# =====================================================
# MATLAB (simulate_abm.m lines 280-284):
#   gamma_G(T_prime+t) = alpha_gamma_G * gamma_G(T_prime+t-1) + beta_gamma_G + eps_gG
#   C_G(T_prime+t) = C_G(T_prime+t-1) * exp(gamma_G(T_prime+t-1))   # LAGGED gamma
#   pi_G(T_prime+t) = alpha_pi_G * pi_G(T_prime+t-1) + beta_pi_G + eps_pG
#   P_G(T_prime+t) = P_G(T_prime+t-1) * exp(pi_G(T_prime+t-1))     # LAGGED pi
#
# MATLAB (abm.m line 225):
#   C_d_j = P_G(T_prime+t) * C_G(T_prime+t) / J   (P_G replaces price basket)
#
# Key: MATLAB uses LAGGED gamma/pi for level updates, then AR for next period's gamma/pi.

# NOTE: This MUST be called before rotw_import_export in step! so that
# the 6D epsilon draw is available for both gov and rotw.
function Bit.gov_expenditure(gov::AbstractGovernmentDDGABM, model)
    c_G_g = model.prop.c_G_g
    P_bar_g = model.agg.P_bar_g
    pi_e = model.agg.pi_e
    t = model.agg.t

    # Draw 6D epsilons (or retrieve if already drawn this step)
    eps_gG, eps_pG, _, _, _, _ = get_epsilons!(gov, t)

    # 1. Update LEVELS using LAGGED growth/inflation (MATLAB: exp(gamma_{t-1}))
    C_G = gov.C_G * exp(gov.g_prev_G)
    P_G = gov.P_G * exp(gov.pi_prev_G)

    # 2. AR step: compute NEW growth/inflation for this period
    g_new_G = gov.alpha_gamma_G * gov.g_prev_G + gov.beta_gamma_G + eps_gG
    pi_new_G = gov.alpha_pi_G * gov.pi_prev_G + gov.beta_pi_G + eps_pG

    # 3. Store updated values
    gov.C_G = C_G
    gov.P_G = P_G
    gov.g_prev_G = g_new_G
    gov.pi_prev_G = pi_new_G

    # 4. Compute demand (MATLAB abm.m line 225: C_d_j = P_G * C_G / J)
    # P_G REPLACES the base model's sum(c_G_g.*P_bar_g)*(1+pi_e) — do NOT include both!
    J = size(gov.C_d_j, 1)
    C_d_j = P_G * C_G ./ J .* ones(J)

    return C_G, C_d_j
end

# =====================================================
# OVERRIDE rotw_import_export FOR DDGABM ROTW
# =====================================================
# MATLAB (simulate_abm.m lines 286-302):
#   gamma_E, pi_E, gamma_I, pi_I updated via AR with correlated epsilons
#   C_E, P_E, Y_I, P_I updated using LAGGED growth/inflation
#
# MATLAB (abm.m lines 227, 229-230):
#   C_d_l = P_E * C_E / L   (P_E replaces price basket)
#   P_m = ones(1,G) * P_I
#   Y_m = c_I_g * Y_I

function Bit.rotw_import_export(rotw::AbstractRestOfTheWorldDDGABM, model)
    c_E_g = model.prop.c_E_g
    c_I_g = model.prop.c_I_g
    P_bar_g = model.agg.P_bar_g
    pi_e = model.agg.pi_e
    t = model.agg.t

    # Get epsilons from government's shared draw
    gov = rotw.gov_ref[]
    _, _, eps_gE, eps_pE, eps_gI, eps_pI = get_epsilons!(gov, t)

    # --- EXPORTS ---
    # 1. Update levels using LAGGED growth/inflation
    C_E = rotw.C_E * exp(rotw.g_prev_E)
    P_E = rotw.P_E * exp(rotw.pi_prev_E)

    # 2. AR step: compute NEW growth/inflation
    g_new_E = rotw.alpha_gamma_E * rotw.g_prev_E + rotw.beta_gamma_E + eps_gE
    pi_new_E = rotw.alpha_pi_E * rotw.pi_prev_E + rotw.beta_pi_E + eps_pE

    # 3. Store updated values
    rotw.C_E = C_E
    rotw.P_E = P_E
    rotw.g_prev_E = g_new_E
    rotw.pi_prev_E = pi_new_E

    # 4. Export demand (MATLAB abm.m line 227: C_d_l = P_E * C_E / L)
    # P_E REPLACES the base model's sum(c_E_g.*P_bar_g)*(1+pi_e) — do NOT include both!
    L = size(rotw.C_d_l, 1)
    C_d_l = P_E * C_E ./ L .* ones(L)

    # --- IMPORTS ---
    # 1. Update levels using LAGGED growth/inflation
    Y_I = rotw.Y_I * exp(rotw.g_prev_I)
    P_I = rotw.P_I * exp(rotw.pi_prev_I)

    # 2. AR step: compute NEW growth/inflation
    g_new_I = rotw.alpha_gamma_I * rotw.g_prev_I + rotw.beta_gamma_I + eps_gI
    pi_new_I = rotw.alpha_pi_I * rotw.pi_prev_I + rotw.beta_pi_I + eps_pI

    # 3. Store updated values
    rotw.Y_I = Y_I
    rotw.P_I = P_I
    rotw.g_prev_I = g_new_I
    rotw.pi_prev_I = pi_new_I

    # 4. Import supply and prices (MATLAB abm.m lines 229-230)
    Y_m = c_I_g * Y_I
    G = length(P_bar_g)
    P_m = ones(G) .* P_I  # MATLAB: P_m = ones(1,G) * P_I

    return C_E, Y_I, C_d_l, Y_m, P_m
end

# =====================================================
# CALIBRATION WRAPPER
# =====================================================

"""
    add_ddgabm_params!(params, ic, cal, calibration_date)

Add DDGABM 6D exogenous parameters to params/ic dicts.
Estimates growth-rate AR and deflator AR for G, E, I,
computes 6x6 covariance from residuals.

MATLAB source: DDGABM/calibration/set_parameters_and_initial_conditions.m lines 185-193
"""
function add_ddgabm_params!(params, ic, cal, calibration_date)
    data = cal.data
    estimation_date = cal.estimation_date

    T_estimation_exo = findall(data["quarters_num"] .== Bit.date2num(estimation_date))[1][1]
    T_calibration_exo = findall(data["quarters_num"] .== Bit.date2num(calibration_date))[1][1]

    # Range matches MATLAB: diff(log(data.X(T_estimation_exo-1:T_calibration_exo)))
    # gives (T_calibration_exo - T_estimation_exo + 1) growth rate observations
    idx_range = (T_estimation_exo - 1):T_calibration_exo

    # --- Growth-rate AR residuals ---
    G_growth = diff(log.(data["real_government_consumption_quarterly"][idx_range]))
    alpha_gamma_G, beta_gamma_G, _, epsilon_G = Bit.estimate_for_calibration_script(G_growth)

    E_growth = diff(log.(data["real_exports_quarterly"][idx_range]))
    alpha_gamma_E, beta_gamma_E, _, epsilon_E = Bit.estimate_for_calibration_script(E_growth)

    I_growth = diff(log.(data["real_imports_quarterly"][idx_range]))
    alpha_gamma_I, beta_gamma_I, _, epsilon_I = Bit.estimate_for_calibration_script(I_growth)

    # --- Deflator AR residuals ---
    pi_G_series = diff(log.(data["government_consumption_deflator_quarterly"][idx_range]))
    alpha_pi_G, beta_pi_G, _, epsilon_pi_G = Bit.estimate_for_calibration_script(pi_G_series)

    pi_E_series = diff(log.(data["exports_deflator_quarterly"][idx_range]))
    alpha_pi_E, beta_pi_E, _, epsilon_pi_E = Bit.estimate_for_calibration_script(pi_E_series)

    pi_I_series = diff(log.(data["imports_deflator_quarterly"][idx_range]))
    alpha_pi_I, beta_pi_I, _, epsilon_pi_I = Bit.estimate_for_calibration_script(pi_I_series)

    # --- 6x6 covariance: (gamma_G, pi_G, gamma_E, pi_E, gamma_I, pi_I) ---
    # Match MATLAB ordering from set_parameters_and_initial_conditions.m line 193
    # epsilon vectors from estimate_for_calibration_script are matrices; vec them
    C6 = cov([vec(epsilon_G) vec(epsilon_pi_G) vec(epsilon_E) vec(epsilon_pi_E) vec(epsilon_I) vec(epsilon_pi_I)])

    # Store growth-rate AR params
    params["alpha_gamma_G"] = alpha_gamma_G
    params["beta_gamma_G"] = beta_gamma_G
    params["alpha_gamma_E"] = alpha_gamma_E
    params["beta_gamma_E"] = beta_gamma_E
    params["alpha_gamma_I"] = alpha_gamma_I
    params["beta_gamma_I"] = beta_gamma_I

    # Store deflator AR params
    params["alpha_pi_G"] = alpha_pi_G
    params["beta_pi_G"] = beta_pi_G
    params["alpha_pi_E"] = alpha_pi_E
    params["beta_pi_E"] = beta_pi_E
    params["alpha_pi_I"] = alpha_pi_I
    params["beta_pi_I"] = beta_pi_I

    # Store 6x6 covariance
    params["C6"] = C6

    # Initial conditions: lagged values at T_prime
    ic["g_prev_G"] = G_growth[end]
    ic["g_prev_E"] = E_growth[end]
    ic["g_prev_I"] = I_growth[end]
    ic["pi_prev_G"] = pi_G_series[end]
    ic["pi_prev_E"] = pi_E_series[end]
    ic["pi_prev_I"] = pi_I_series[end]

    return params, ic
end

# =====================================================
# FACTORY: Create DDGABM Government + RestOfTheWorld
# =====================================================

"""
    create_ddgabm_gov_rotw(p, ic)

Create GovernmentDDGABM and RestOfTheWorldDDGABM from params/ic
that have been augmented by add_ddgabm_params!.

Returns (gov, rotw) tuple.
"""
function create_ddgabm_gov_rotw(p, ic)
    # Create base agents
    gov_base = Bit.Government(p, ic)
    rotw_base = Bit.RestOfTheWorld(p, ic)

    C6 = p["C6"]

    # Shared epsilon storage refs
    eps_gG = Ref(0.0)
    eps_pG = Ref(0.0)
    eps_gE = Ref(0.0)
    eps_pE = Ref(0.0)
    eps_gI = Ref(0.0)
    eps_pI = Ref(0.0)
    eps_drawn_t = Ref(0)

    # Create DDGABM Government
    gov = GovernmentDDGABM(
        (getfield(gov_base, x) for x in fieldnames(Bit.Government))...,
        p["alpha_gamma_G"],
        p["beta_gamma_G"],
        p["alpha_pi_G"],
        p["beta_pi_G"],
        ic["g_prev_G"],
        ic["pi_prev_G"],
        1.0,  # P_G initial = 1.0 (normalized)
        C6,
        eps_gG, eps_pG, eps_gE, eps_pE, eps_gI, eps_pI,
        eps_drawn_t
    )

    # Create DDGABM RestOfTheWorld
    rotw = RestOfTheWorldDDGABM(
        (getfield(rotw_base, x) for x in fieldnames(Bit.RestOfTheWorld))...,
        p["alpha_gamma_E"],
        p["beta_gamma_E"],
        p["alpha_gamma_I"],
        p["beta_gamma_I"],
        p["alpha_pi_E"],
        p["beta_pi_E"],
        p["alpha_pi_I"],
        p["beta_pi_I"],
        ic["g_prev_E"],
        ic["pi_prev_E"],
        ic["g_prev_I"],
        ic["pi_prev_I"],
        1.0,  # P_E initial = 1.0
        1.0,  # P_I initial = 1.0
        Ref(gov)
    )

    return gov, rotw
end

# =====================================================
# CALIBRATION WRAPPER: Auto-include DDGABM params
# =====================================================

"""
    DDGABMCalibration(cal)

Wrapper around a calibration object that automatically augments
params/initial_conditions with DDGABM 6D exogenous parameters
when passed to `Bit.get_params_and_initial_conditions`.

Usage:
    dcal = DDGABMCalibration(cal)
    p, ic = Bit.get_params_and_initial_conditions(dcal, date)
    # p now has "C6", deflator AR params, etc. — no manual add_ddgabm_params! needed
"""
struct DDGABMCalibration
    cal  # The underlying calibration object
end

function Bit.get_params_and_initial_conditions(dcal::DDGABMCalibration, calibration_date; scale=0.001)
    p, ic = Bit.get_params_and_initial_conditions(dcal.cal, calibration_date; scale=scale)
    add_ddgabm_params!(p, ic, dcal.cal, calibration_date)
    return p, ic
end
