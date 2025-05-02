# # Essential use of BeforeIT

# We start by importing the BeforeIT library and other useful libraries.

import BeforeIT as Bit

using FileIO, Plots

# We then initialise the model loading some precomputed set of parameters and by specifying a number of epochs.
# In another tutorial we will illustrate how to compute parameters and initial conditions.

year_i = 2019
quarter = 4

country = "netherlands"
#country = "italy"

parameters = load(pwd() * "/src/utils/parameters_initial_conditions_data/$(country)/parameters/"* string(year_i) *"Q"* string(quarter) *".jld2");
initial_conditions = load(pwd() * "/src/utils/parameters_initial_conditions_data/$(country)/initial_conditions/"* string(year_i) *"Q"* string(quarter) *".jld2");

# To run mu
# We can now initialise the model, by specifying in advance the maximum number of epochs.

T = 16
model = Bit.init_model(parameters, initial_conditions, T)

# Note that the it is very simple to inspect the model by typing

fieldnames(typeof(model))

# and to inspect the specific attributes of one agent type by typing

fieldnames(typeof(model.bank))

# We can now define a data tracker, which will store the time series of the model.

data = Bit.init_data(model);

# We can run now the model for a number of epochs and progressively update the data tracker.

for t in 1:T
    Bit.step!(model; multi_threading = true)
    Bit.update_data!(data, model)
end

# Note that we can equivalently run the model for a number of epochs in the single command 
# `data = Bit.run!(model)`, but writing the loop explicitely is more instructive.

# We can then plot any time series stored in the data tracker, for example

plot(data.real_gdp, title = "gdp", titlefont = 10)

# Or we can plot multiple time series at once using the function `plot_data`

ps = Bit.plot_data(data, quantities = [:real_gdp, :real_household_consumption, :real_government_consumption, :real_capitalformation, :real_exports, :real_imports, :wages, :euribor, :gdp_deflator])
plot(ps..., layout = (3, 3))

# To run multiple monte-carlo repetitions in parallel we can use

model = Bit.init_model(parameters, initial_conditions, T)

model.prop.theta_UNION = 0.8

data_vector = Bit.ensemblerun(model, 20)

# Note that this will use the number of threads specified when activating the Julia environment.
# To discover the number of threads available, you can use the command 

Threads.nthreads()

# To activate Julia with a specific number of threads, say 8, you can use the command
# `julia -t 8` in the terminal.

# We can then plot the results of the monte-carlo repetitions using the function `plot_data_vector`

ps = Bit.plot_data_vector(data_vector)
plot(ps..., layout = (3, 3))


# income accounting and production accounting should be equal
zero = sum(data.nominal_gva - data.compensation_employees - data.operating_surplus - data.taxes_production)
@test isapprox(zero, 0.0, atol = 1e-8)

# compare nominal_gdp to total expenditure
zero = sum(
    data.nominal_gdp - data.nominal_household_consumption - data.nominal_government_consumption -
    data.nominal_capitalformation - data.nominal_exports + data.nominal_imports,
)
@test isapprox(zero, 0.0, atol = 1e-8)

zero = sum(
    data.real_gdp - data.real_household_consumption - data.real_government_consumption -
    data.real_capitalformation - data.real_exports + data.real_imports,
)    
@test isapprox(zero, 0.0, atol = 1e-8)

# accounting identity of balance sheet of central bank
zero = model.cb.E_CB + model.rotw.D_RoW - model.gov.L_G + model.bank.D_k
@test isapprox(zero, 0.0, atol = 1e-8)

# accounting identity of balance sheet of commercial bank
tot_D_h = sum(model.w_act.D_h) + sum(model.w_inact.D_h) + sum(model.firms.D_h) + model.bank.D_h
zero = sum(model.firms.D_i) + tot_D_h + sum(model.bank.E_k) - sum(model.firms.L_i) - model.bank.D_k
@test isapprox(zero, 0.0, atol = 1e-8)