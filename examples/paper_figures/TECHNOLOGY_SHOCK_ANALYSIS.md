# Technology Shock Discrepancy Analysis

## Executive Summary

The Julia (BeforeIT.jl) implementation shows a ~2.5% GDP response to a 10% productivity shock, while the MATLAB (DDGABM) implementation shows ~7% GDP response for the EUBI model. This document explains the root cause and proposed fix.

## Root Cause: Different Growth Expectations Mechanisms

### MATLAB Approach (abm.m lines 90, 104)
```matlab
% AR estimation on growth RATES directly
[alpha_gamma, beta_gamma, u_gamma] = estimate(gamma(1:T_prime+t-1));

% Expected growth from AR on gamma
gamma_Y = exp(alpha_gamma * gamma(T_prime+t-1) + beta_gamma + epsilon_gamma) - 1;

% Update actual gamma after each period (line 308)
gamma(T_prime+t) = log(sum(Q_i+S_i+DS_DISP_i) / Y(T_prime+t-1));
```

**Key characteristic**: AR is performed on **growth rates** directly. When a shock causes higher-than-expected growth, that high growth rate persists through the AR coefficient, creating a **multiplier effect**.

### Julia Approach (estimations.jl lines 40-47)
```julia
# AR estimation on log GDP LEVELS
lY_e = estimate_next_value(log.(Y_slice))
Y_e = exp(lY_e)
gamma_e = Y_e / Y[T_prime + t - 1] - 1
```

**Key characteristic**: AR is performed on **log GDP levels**, then growth is derived. This approach captures level persistence but not growth rate persistence.

## Mathematical Difference

### MATLAB Growth Rate AR
```
gamma_e = exp(α_γ · γ_{t-1} + β_γ + ε) - 1
```

If γ_{t-1} = 5% (due to shock), and α_γ = 0.8:
- γ_e ≈ exp(0.8 × 0.05 + β + ε) - 1 ≈ 4% + trend

The shock creates **high growth persistence** through the AR coefficient.

### Julia Level AR
```
log(Y_e) = α_Y · log(Y_{t-1}) + β_Y + ε
Y_e = Y_{t-1}^α · exp(β + ε)
γ_e = Y_e/Y_{t-1} - 1 = Y_{t-1}^{α-1} · exp(β + ε) - 1
```

If α ≈ 1 (typical for GDP), then γ_e ≈ exp(β + ε) - 1, which is approximately constant regardless of past growth.

The shock affects the **level** but growth expectations return to trend faster.

## Empirical Verification

Using Octave to trace the MATLAB calibration (2010Q1):

| Metric | Value |
|--------|-------|
| Number of firms | 6,113 |
| Labor cost share | 36.2% |
| Material cost share | 54.2% |
| Capital cost share | 9.6% |
| Initial markup | 22.0% |
| Cost reduction (10% productivity) | -3.3% |
| Markup increase (cost absorption) | +3.5% |
| **Net price change** | **-2.7%** |

The price response is small due to EUBI's cost absorption (δ_AC=1). The large GDP response must come from the **quantity rule multiplier** via gamma_e persistence.

## EUBI Quantity Rule
```
Q_s = Q_d × (1 + γ_e) - S
```

With δ_Y=1, the expected growth rate directly multiplies demand. When gamma_e persists at high levels (MATLAB) vs returning to trend (Julia), this creates:
- **MATLAB**: Strong multiplier effect → ~7% GDP response
- **Julia**: Weaker multiplier → ~2.5% GDP response

## Proposed Fix

Modify BeforeIT.jl to track gamma history and use AR on growth rates:

1. **Track actual growth rates** in Aggregates:
```julia
mutable struct Aggregates
    ...
    gamma_history::Vector{Float64}  # Add this field
end
```

2. **Update gamma_history** after each step:
```julia
actual_gamma = log(Y[t] / Y[t-1])
push!(gamma_history, actual_gamma)
```

3. **Use AR on gamma** for expectations:
```julia
alpha, beta, epsilon = estimate(gamma_history)
gamma_e = exp(alpha * gamma_history[end] + beta + epsilon) - 1
```

## Files Created

- `EUBI_matlab_gamma.jl` - Prototype implementation with MATLAB-style gamma AR
- `trace_productivity_shock.m` - Octave script for MATLAB value verification

## Conclusion

The technology shock magnitude discrepancy is caused by different growth expectations mechanisms:
- MATLAB: AR on growth rates → strong persistence → large GDP multiplier
- Julia: AR on log levels → weak growth persistence → smaller GDP multiplier

Both implementations are mathematically valid but produce different dynamics. To match MATLAB's behavior exactly, Julia needs to switch to AR on growth rates.
