# # Scenario analysis via wage shock

import BeforeIT as Bit
import StatsBase: mean, std
using Plots, FileIO

year_i = 2016
quarter = 4

parameters = load(pwd() * "/src/utils/parameters_initial_conditions_data/netherlands/parameters/"* string(year_i) *"Q"* string(quarter) *".jld2");
initial_conditions = load(pwd() * "/src/utils/parameters_initial_conditions_data/netherlands/initial_conditions/"* string(year_i) *"Q"* string(quarter) *".jld2");

# Initialise the model and the data collector

T = 16
model = Bit.init_model(parameters, initial_conditions, T);

# Simulate the baseline model for T quarters, N_reps times, and collect the data

N_reps = 64
data_vec_baseline = Bit.ensemblerun(model, N_reps)

# Now, apply a shock to the model and simulate it again.
# A shock is simply a function that takes the model and changes some of
# its parameters for a specific time period.
# We do this by first defining a "struct" with useful attributes.
# For example, we can define an productivity and a consumption shock with the following structs


struct WageShock
    wage_multiplier::Float64    # wage multiplier
    shock_time::Int
end

# and then by making the structs callable functions that change the parameters of the model,
# this is done in Julia using the syntax below

# A permanent change in the wage by the factor s.wage_multiplier

function (s::WageShock)(model::Bit.Model)
   if model.agg.t == s.shock_time
        model.firms.w_bar_i = model.firms.w_bar_i .* s.wage_multiplier
   end
end


wage_shock = WageShock(1.02,4)


# Simulate the model with the shock

data_vec_shocked = Bit.ensemblerun(model, N_reps; shock = wage_shock)

# Compute mean and standard error of GDP for the baseline and shocked simulations

mean_gdp_baseline = mean(data_vec_baseline.real_gdp, dims = 2)
mean_gdp_shocked = mean(data_vec_shocked.real_gdp, dims = 2)
sem_gdp_baseline = std(data_vec_baseline.real_gdp, dims = 2) / sqrt(N_reps)
sem_gdp_shocked = std(data_vec_shocked.real_gdp, dims = 2) / sqrt(N_reps)

mean_inflation_baseline = mean(data_vec_baseline.nominal_gdp./data_vec_baseline.real_gdp, dims = 2)
mean_inflation_shocked = mean(data_vec_shocked.nominal_gdp./data_vec_shocked.real_gdp, dims = 2)
sem_inflation_baseline = std(data_vec_baseline.nominal_gdp./data_vec_baseline.real_gdp, dims = 2) / sqrt(N_reps)
sem_inflation_shocked = std(data_vec_shocked.nominal_gdp./data_vec_shocked.real_gdp, dims = 2) / sqrt(N_reps)

mean_wages_baseline = mean(data_vec_baseline.wages, dims = 2)
mean_wages_shocked = mean(data_vec_shocked.wages, dims = 2)
sem_wages_baseline = std(data_vec_baseline.wages, dims = 2) / sqrt(N_reps)
sem_wages_shocked = std(data_vec_shocked.wages, dims = 2) / sqrt(N_reps)

# Compute the ratio of shocked to baseline GDP

gdp_ratio = mean_gdp_shocked ./ mean_gdp_baseline

inflation_ratio = mean_inflation_shocked ./ mean_inflation_baseline

wages_ratio = mean_wages_shocked ./ mean_wages_baseline


# the standard error on a ratio of two variables is computed with the error propagation formula

sem_gdp_ratio = gdp_ratio .* ((sem_gdp_baseline ./ mean_gdp_baseline).^2 .+ (sem_gdp_shocked ./ mean_gdp_shocked).^2).^0.5

sem_inflation_ratio = inflation_ratio .* ((sem_inflation_baseline ./ mean_gdp_baseline).^2 .+ (sem_inflation_shocked ./ mean_inflation_shocked).^2).^0.5

sem_wages_ratio = inflation_ratio .* ((sem_wages_baseline ./ mean_gdp_baseline).^2 .+ (sem_wages_shocked ./ mean_wages_shocked).^2).^0.5

# Finally, we can plot the impulse response curve

plot(
    1:T+1,
    gdp_ratio,
    ribbon = sem_gdp_ratio,
    fillalpha = 0.2,
    label = "",
    xlabel = "quarters",
    ylabel = "GDP change",
)

plot(
    1:T+1,
    inflation_ratio,
    ribbon = sem_inflation_ratio,
    fillalpha = 0.2,
    label = "",
    xlabel = "quarters",
    ylabel = "GDP change",
)

plot(
    1:T+1,
    wages_ratio,
    ribbon = sem_wages_ratio,
    fillalpha = 0.2,
    label = "",
    xlabel = "quarters",
    ylabel = "GDP change",
)

# We can save the figure using: savefig("gdp_shock.png")

