using LinearAlgebra, Random, Distributions

function estimate_next_value(data, type = nothing)
    alpha, beta, epsilon = estimate(data)
    return alpha * data[end] + beta + epsilon
end

function estimate_next_value_VAR(data, type = nothing)
    alpha, beta, epsilon = estimate_VAR(data)
    return alpha ⋅ data[end,:] .+ beta .+ epsilon
end

function forecast_k_steps_VAR(data, n_forecasts; intercept = false, lags = 1, stochastic = false)
    """
    
    forecast_k_steps_VAR(data, k; intercept = false, lags = 1)

    Forecasts the next `k` steps of a Vector Autoregression (VAR) model.

    # Arguments
    - `data::Matrix{Float64}`: The input time series data, where rows represent time steps and columns represent variables.
    - `k::Int`: The number of steps to forecast.
    - `intercept::Bool`: Whether to include an intercept in the VAR model. Default is `false`.
    - `lags::Int`: The number of lags to include in the VAR model. Default is `1`.

    # Returns
    - `forecasted_values::Matrix{Float64}`: A matrix containing the forecasted values for the next `k` steps.

    # Example
    ```julia
    N = 500  # Number of time steps
    p = 2  # Number of lags
    alpha = [0.9 -0.2; 0.5 -0.1]  # Coefficient matrix (2 variables, 1 lag)
    alpha = [0.9 -0.2 0.1 0.05; # Coefficient matrix   (2 variables, 2 lags)
            0.5 -0.1 0.2 -0.3]
    beta = [0.1, -0.2]  # Intercept vector
    sigma = [0.1 0.05; 0.05 0.1]  # Covariance matrix for the noise
    timeseries = generate_var_timeseries(N, p, alpha, beta, sigma)
    alpha_hat, beta_hat, epsilon_hat = BIT.estimate_VAR(timeseries, intercept = false, lags = p)
    # Run the forecast function
    forecast = BIT.forecast_k_steps_VAR(timeseries, 10, intercept = true, lags = p)
    """

    forecasted_values = Matrix{Float64}(undef, 0, size(data, 2))
    
    alpha, beta, epsilon, sigma = estimate_VAR(data; intercept = intercept , lags = lags)

    alpha = dropdims(alpha, dims = tuple(findall(size(alpha) .== 1)...))

    if lags > 1
        alpha = reshape(alpha,  size(data,2), lags *size(data,2))
    end

    if length(alpha) == 1
        alpha = only(alpha)
    end

    if !stochastic
        epsilon .=0
    end
    
    
    for i in 1:n_forecasts

        if stochastic
            epsilon = zeros(size(data, 2))
            for i = 1:size(data, 2)
                epsilon[i] = rand(Normal(0, sqrt(sigma[i, i])))
            end 
        end

        if lags ==1
            if intercept
                next_value = alpha * data[end,:] .+ beta .+ epsilon
            else
                next_value = alpha * data[end,:] .+ epsilon
            end
        else
            if intercept
                next_value = alpha * vec(data[end:-1:end-lags+1,:]') .+ beta .+ epsilon
            else
                next_value = alpha * vec(data[end:-1:end-lags+1,:]') .+ epsilon
            end
        end
        forecasted_values = vcat(forecasted_values, next_value')
        data = vcat(data, next_value')
    end

    return forecasted_values
end

function forecast_k_steps_VARX(data, exogenous, n_forecasts; intercept = false, lags = 1, stochastic = false)
    """
    
    forecast_k_steps_VARX(data, k; intercept = false, lags = 1)

    Forecasts the next `k` steps of a Vector Autoregression with exogenous predictors (VARX) model.

    # Arguments
    - `data::Matrix{Float64}`: The input time series data, where rows represent time steps and columns represent variables.
    - `exogenous::Matrix{Float64}`: The exogenous time series data, where rows represent time steps and columns represent variables.
    - `k::Int`: The number of steps to forecast.
    - `intercept::Bool`: Whether to include an intercept in the VAR model. Default is `false`.
    - `lags::Int`: The number of lags to include in the VAR model. Default is `1`.

    # Returns
    - `forecasted_values::Matrix{Float64}`: A matrix containing the forecasted values for the next `k` steps.
    """

    alpha, beta, gamma, epsilon, sigma = estimate_VARX(data, exogenous[2:size(data,1)+1,:]; intercept = intercept , lags = lags)

    alpha = dropdims(alpha, dims = tuple(findall(size(alpha) .== 1)...))

    if lags > 1
        alpha = reshape(alpha,  size(data,2), lags *size(data,2))
    end

    if length(alpha) == 1
        alpha = only(alpha)
    end

    if !stochastic
        epsilon .=0
    end

    forecasted_values = Matrix{Float64}(undef, 0, size(data, 2))
    
    for i in 1:n_forecasts
        
        if stochastic
            epsilon = zeros(size(data, 2))
            for i = 1:size(data, 2)
                epsilon[i] = rand(Normal(0, sqrt(sigma[i, i])))
            end 
        end

        if lags ==1
            if intercept
                next_value = alpha * data[end,:] .+ gamma * exogenous[end - n_forecasts + i ,:] .+ beta .+ epsilon
            else
                next_value = alpha * data[end,:] .+ gamma * exogenous[end - n_forecasts + i ,:] .+ epsilon
            end
        else
            if intercept
                next_value = alpha * vec(data[end:-1:end-lags+1,:]') .+ gamma * exogenous[end - n_forecasts + i ,:] .+ beta .+ epsilon
            else
                next_value = alpha * vec(data[end:-1:end-lags+1,:]') .+ gamma * exogenous[end - n_forecasts + i ,:] .+ epsilon
            end
        end
        forecasted_values = vcat(forecasted_values, next_value')
        data = vcat(data, next_value')
    end

    return forecasted_values
end

function estimate_next_value_VARX(data, exo, type = nothing)
    alpha_1, alpha_2, beta, gamma_1, gamma_2, gamma_3, epsilon = estimate_with_predictors_VARX(data, exo)
    val = alpha_1 .* data[end,1] .+ alpha_2 .* data[end,2] .+ gamma_1 .* exo[end,1] .+ gamma_2 .* exo[end,2] .+ gamma_3 .* exo[end,3] .+ beta .+ epsilon
    return val
end

function estimate_with_predictors_VARX(ydata::Matrix{Float64}, exo::Matrix{Float64})

    if typeof(ydata) == Vector{Float64} 
        ydata = ydata[:, :]
    end

    var = rfvar3(ydata, 1, hcat(ones(size(ydata, 1)), exo[1:size(ydata,1), :]))
    alpha_1 = var.By[:,1]
    alpha_2 = var.By[:,2]
    beta = var.Bx[:,1]
    gamma_1 = var.Bx[:,2]
    gamma_2 = var.Bx[:,3]
    gamma_3 = var.Bx[:,4]
    epsilon = zeros(2,1)
    epsilon[1] = rand(Normal(0, sqrt(cov(var.u[:,1]))))
    epsilon[2] = rand(Normal(0, sqrt(cov(var.u[:,2]))))
    return alpha_1, alpha_2, beta, gamma_1, gamma_2, gamma_3, epsilon
end

function estimate(ydata::Union{Matrix{Float64}, Vector{Float64}})
    if typeof(ydata) == Vector{Float64}
        ydata = ydata[:, :]
    end
    var = rfvar3(ydata, 1, ones(size(ydata, 1), 1))
    alpha = var.By[1]
    beta = var.Bx[1]
    epsilon = rand(Normal(0, sqrt(cov(var.u))[1, 1]))
    sigma = sqrt(cov(var.u))[1, 1]
    return alpha, beta, epsilon, sigma
end

function estimate_VAR(ydata::Union{Matrix{Float64}, Vector{Float64}}; intercept = false, lags = 1)    
    if typeof(ydata) == Vector{Float64}
        ydata = ydata[:, :]
    end

    if intercept
        var = rfvar3(ydata, lags, ones(size(ydata, 1), 1))
    else
        var = rfvar3(ydata, lags,Array{Float64}(undef, size(ydata, 1), 0))
    end
    
    alpha = var.By
    beta = var.Bx
    sigma = cov(var.u)

    epsilon = zeros(size(ydata, 2))
    for i = 1:size(ydata, 2)
        epsilon[i] = rand(Normal(0, sqrt(sigma[i, i])))
    end 

    return alpha, beta, epsilon, sigma, var.u
end

function estimate_VARX(ydata::Union{Matrix{Float64}, Vector{Float64}}, xdata::Union{Matrix{Float64}, Vector{Float64}}; intercept = false, lags = 1)    

    if typeof(ydata) == Vector{Float64}
        ydata = ydata[:, :]
    end
    if typeof(xdata) == Vector{Float64}
        xdata = xdata[:, :]
    end

    if intercept
        xdata = hcat(ones(size(xdata, 1), 1), xdata)
    end
        
    var = rfvar3(ydata, lags, xdata)
    
    
    alpha = var.By
    beta = var.Bx[:,1]
    gamma = var.Bx[:,2:end]
    sigma = cov(var.u)

    epsilon = zeros(size(ydata, 2))
    for i = 1:size(ydata, 2)
        epsilon[i] = rand(Normal(0, sqrt(sigma[i, i])))
    end 

    return alpha, beta, gamma, epsilon, sigma, var.u
end


function estimate_for_calibration_script(ydata::Union{Matrix{Float64}, Vector{Float64}})
    if typeof(ydata) == Vector{Float64}
        ydata = ydata[:, :]
    end
    var = rfvar3(ydata, 1, ones(size(ydata, 1), 1))
    alpha = var.By[1]
    beta = var.Bx[1]
    sigma = sqrt(cov(var.u))[1, 1]
    epsilon = var.u
    return alpha, beta, sigma, epsilon
end


# function estimate_with_predictors(ydata::Union{Matrix{Float64}, Vector{Float64}}, exo::Matrix)

#     if typeof(ydata) == Vector{Float64} 
#         ydata = ydata[:, :]
#     end

#     var = rfvar3(ydata, 1, [ones(size(ydata, 1)), exo[1:length(ydata), :]])
#     alpha = var.By[1]
#     beta = var.Bx[1]
#     gamma_1 = var.Bx[2]
#     gamma_2 = var.Bx[3]
#     gamma_3 = var.Bx[4]
#     epsilon = rand(Normal(0, sqrt(cov(var.u))))
#     return alpha, beta, gamma_1, gamma_2, gamma_3, epsilon
# end



function estimate_taylor_rule(
    r_bar::Union{Matrix{Float64}, Vector{Float64}},
    pi_EA::Vector{Float64},
    gamma_EA::Vector{Float64},
)
    ydata = r_bar
    if typeof(ydata) == Vector{Float64}
        ydata = ydata[:, :]
    end

    exo = [pi_EA gamma_EA]
    var = rfvar3(ydata, 1, exo[1:length(ydata), :])
    alpha = var.By[1]
    gamma_1 = var.Bx[1]
    gamma_2 = var.Bx[2]

    rho = alpha
    xi_pi = gamma_1 ./ (1 .- rho)
    xi_gamma = gamma_2 ./ (1 .- rho)
    pi_star = (0.02 + 1)^(1 / 4) - 1
    r_star = pi_star .* (xi_pi .- 1)

    return rho, r_star, xi_pi, xi_gamma, pi_star
end