import BeforeIT as Bit
using Dates
include("get_calibrator_data.jl")
include("get_calibrator_model_data.jl")

start_date = DateTime(2010, 03, 31)
end_date = DateTime(2013, 12, 31)

# Load historical data
real_data = Bit.load_timeseries_for_calibrator(start_date, end_date; repetitions = 10)

# Create parameter dictionary for simulations
par_dict = Dict(
    "theta_UNION" => 0.5  # Example parameter value, adjust as needed
)

# Get models for the date range
models = Bit.get_models(start_date, end_date)

# Run simulations and get model data
@time simulated_data = run_abm_simulations_with_parameters(
    models,
    par_dict,
    start_date,
    end_date;
    num_simulations = 10,  # Set to same as repetitions in real_data
    multi_threading = false
)

# Compare the first time step of both datasets
println("First time step comparison:")
println("Real data shape: ", size(real_data))
println("Simulated data shape: ", size(simulated_data))

# Display first time step of both datasets (for all variables of first quarter)
num_vars = 5  # Number of variables
println("\nFirst quarter comparison:")
println("Real data (t=1):")
println(real_data[1, 1:num_vars])
println("Simulated data (t=1):")
println(simulated_data[1, 1:num_vars])