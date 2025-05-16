# Here we show how to get simulations for all quarters
# from `2010Q1` to `2019Q4`, and for all years from 2010 to 2019.

import BeforeIT as Bit

using MAT, FileIO

# The following code loads the parameters and initial conditions,
# it initialises the model, runs the model `n_sims` times, and finally
# saves the `data_vector` into a `.jld2` file with an appropriate name.
# The whole process is repeatead for all quarters from `2010Q1` to `2019Q4`,
# and for all years from 2010 to 2019.

function get_simulations(country = "netherlands"; abmx = false, long_run = false, empirical_distribution = false, unconditional_forecasts = false)
    for year in 2010:2019
        for quarter in 1:4
            println("Y: ", year, " Q: ", quarter)
            parameters = load("data/netherlands/parameters/" * string(year) * "Q" * string(quarter) * ".jld2")
            initial_conditions = load("data/netherlands/initial_conditions/" * string(year) * "Q" * string(quarter) * ".jld2")

            if long_run
                T = 30
            else
                T = 12
            end

            model = Bit.init_model(parameters, initial_conditions, T; conditional_forecast = !unconditional_forecasts)
            n_sims = 100

            if abmx
                model.prop.theta_UNION = 0.31
                model.prop.phi_DP = 0.83
                model.prop.phi_F_Q = 0.15
            else
                model.prop.theta_UNION = 0.25
                model.prop.phi_DP = 0.08
                model.prop.phi_F_Q = 0.01
            end
            data_vector = Bit.ensemblerun(model, n_sims; multi_threading = true, abmx = abmx, conditional_forecast = !unconditional_forecasts)

            folder = (empirical_distribution ? "/empirical" : "/calibrated") * (abmx ? (unconditional_forecasts ? "/abmx_uf" : "/abmx") : "/abm") * (long_run ? "/long_run" : "/short_run") * "/simulations/"  
        
            save("data/" * country * folder * string(year) * "Q" * string(quarter) * ".jld2", "data_vector", data_vector)
        end
    end
end

get_simulations(long_run=false, abmx = false, unconditional_forecasts = true)
get_simulations(long_run=false, abmx = true, unconditional_forecasts = true)
#get_simulations(long_run=false, abmx = true, unconditional_forecasts = false)

#get_simulations(long_run=true, abmx = false, unconditional_forecasts = true)
#get_simulations(long_run=true, abmx = true, unconditional_forecasts = true)
#get_simulations(long_run=true, abmx = true, unconditional_forecasts = false)