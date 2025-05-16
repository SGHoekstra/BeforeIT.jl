

"""
    init_aggregates(parameters, initial_conditions, T; typeInt = Int64, typeFloat = Float64)

Initialize aggregates for the model.

# Arguments
- `parameters`: The model parameters.
- `initial_conditions`: The initial conditions.
- `T`: The total simulation time.
- `typeInt`: The integer type to use (default: `Int64`).
- `typeFloat`: The floating-point type to use (default: `Float64`).

# Returns
- `agg`: The initialized aggregates.
- `agg_args`: The arguments used to initialize the aggregates.

"""
function init_aggregates(parameters, initial_conditions, T; typeInt = Int64, typeFloat = Float64, conditional_forecast = false)
    
    Y = initial_conditions["Y"]
    pi_ = initial_conditions["pi"]
    Y = Vector{typeFloat}(vec(vcat(Y, zeros(typeFloat, T))))
    pi_ = Vector{typeFloat}(vec(vcat(pi_, zeros(typeFloat, T))))
    G = typeInt(parameters["G"])

    if !conditional_forecast
        C_G = initial_conditions["C_G"]
        Y_I = initial_conditions["Y_I"]
        C_E = initial_conditions["C_E"]
        
        if T > 12 
            C_G = Vector{typeFloat}(vec(vcat(C_G, zeros(typeFloat, T-12))))
            Y_I = Vector{typeFloat}(vec(vcat(Y_I, zeros(typeFloat, T-12))))
            C_E = Vector{typeFloat}(vec(vcat(C_E, zeros(typeFloat, T-12))))
        end
    else #load all available data for conditional forecasts
        C_G = initial_conditions["C_G_full"]
        Y_I = initial_conditions["Y_I_full"]
        C_E = initial_conditions["C_E_full"]
    end

    P_bar = one(typeFloat)
    P_bar_g = ones(typeFloat, G)
    P_bar_HH = one(typeFloat)
    P_bar_CF = one(typeFloat)

    P_bar_h = zero(typeFloat)
    P_bar_CF_h = zero(typeFloat)
    t = typeInt(1)
    Y_e = zero(typeFloat)
    gamma_e = zero(typeFloat)
    pi_e = zero(typeFloat)
    epsilon_Y_EA = zero(typeFloat)
    epsilon_E = zero(typeFloat)
    epsilon_I = zero(typeFloat)

    agg_args = (
        Y,
        pi_,
        P_bar,
        P_bar_g,
        P_bar_HH,
        P_bar_CF,
        P_bar_h,
        P_bar_CF_h,
        Y_e,
        gamma_e,
        pi_e,
        epsilon_Y_EA,
        epsilon_E,
        epsilon_I,
        t,
        C_G,
        C_E,
        Y_I,
    )

    agg = Aggregates(agg_args...)

    return agg, agg_args
end