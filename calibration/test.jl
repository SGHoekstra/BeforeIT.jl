using Test
using Dates

using BeforeIT

# This tests the run_abm_simulations_with_parameters function with the new signature
# that takes theta_UNION directly as a parameter

# Mock data setup
models = BeforeIT.get_models(DateTime("2010-03-31"), DateTime("2013-12-31"))
start_date = DateTime("2010-03-31")
end_date = DateTime("2013-12-31")


result = BeforeIT.run_abm_simulations_with_parameters(
    models,
    0.8,
    start_date,
    end_date;
    num_simulations=10,
    multi_threading=true
)

