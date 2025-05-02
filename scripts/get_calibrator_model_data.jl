# Load necessary packages
import BeforeIT as BIT
using Dates, FileIO


"""
    run_model_simulations(
        params::Dict;
        start_year_quarter::Tuple{Int, Int},
        end_year_quarter::Tuple{Int, Int},
        num_simulations::Int = 100,
        model_type::String = "base",
        abmx::Bool = false,
        longrun::Bool = false,
        empirical_distribution::Bool = false,
        unconditional_forecasts::Bool = true,
        apply_transformations::Bool = true,
        data_path::String = "./src/utils/calibration_data/netherlands/data/1996.mat"
    )

Runs the economic model for multiple simulations and returns a structured matrix of results.

# Arguments
- `params::Dict`: Parameters for model initialization.
- `start_year_quarter::Tuple{Int, Int}`: Starting year and quarter, e.g., (2010, 1).
- `end_year_quarter::Tuple{Int, Int}`: Ending year and quarter, e.g., (2019, 4).
- `num_simulations::Int`: Number of simulations to run for each starting quarter.
- `data_path::String`: Path to the MAT file containing data.

# Returns
- `results::Array`: A T×(N×M×S) array where:
  - T is the forecast horizon + 1
  - N is the number of starting quarters
  - M is the number of variables (5)
  - S is the number of simulations
- `quarters::Vector`: List of quarters used
- `variables::Vector`: List of variables included
"""

function run_abm_simulations_with_parameters(
    models,
    calibration_params::Dict,
    start_date::DateTime,
    end_date::DateTime;
    num_simulations::Int = 100,
    multi_threading::Bool = false,
)

    # Extract variables used in the script
    variables = [
        "real_gdp_quarterly",
        "gdp_deflator_growth_quarterly",
        "real_household_consumption_quarterly", 
        "real_fixed_capitalformation_quarterly",
        "euribor"
    ]
    
    data = BIT.NETHERLANDS_CALIBRATION.data


    # Convert start and end dates to numeric format
    start_num = BIT.date2num(start_date)
    end_num = BIT.date2num(end_date)
    
    # Filter quarters within the range
    quarters = data["quarters_num"]
    valid_quarters = quarters[(quarters .>= start_num) .& (quarters .<= end_num)]
    unique_quarters = sort(unique(valid_quarters))
    
    
    # Number of starting quarters
    N = length(unique_quarters)-1
    
    # Number of variables
    M = length(variables)
    
    # Number of simulations
    S = num_simulations
    
    # Initialize output tensor
    # Dimensions: T x (N x M x S)
    # Format is [time_step, quarter_idx, variable_idx, simulation_idx]
    output = zeros(12 + 1, N * M * S)


    for (idx, quarter_num) in enumerate(unique_quarters)
        quarter_date = BIT.num2date(quarter_num)
                
        

        current_idx = idx
        end_idx = length(unique_quarters)

        quarters_diff = end_idx - current_idx
        
        if quarters_diff < 1
            continue
        end

        model = models[idx]

        # Set T to the difference, with a maximum of 12
        T = min(quarters_diff, 12)

        # Run simulations
        sims = BIT.run_n_sims(model, num_simulations; multi_threading = multi_threading)

        
        predictions = BIT.get_predictions_from_sims_directly(data, sims, quarter_num, T, S)

        transformed_predictions = hcat(collect([
            log.(predictions["real_gdp_quarterly"]),
            log.(1 .+ predictions["gdp_deflator_growth_quarterly"]),
            log.(predictions["real_household_consumption_quarterly"]),
            log.(predictions["real_fixed_capitalformation_quarterly"]),
            (1 .+ predictions["euribor"]).^(1/4)
            ])...)

        output[1:(T + 1), (idx - 1) * M * S + 1 : idx * M * S ] = transformed_predictions
    end
    



    
    return output
end


function get_models(
    start_date::DateTime,
    end_date::DateTime;
    model_type::String = "base",
    empirical_distribution::Bool = false,
    conditional_forecasts::Bool = true,
)

    data = BIT.NETHERLANDS_CALIBRATION.data


    # Convert start and end dates to numeric format
    start_num = BIT.date2num(start_date)
    end_num = BIT.date2num(end_date)
    
    T = 12

    # Filter quarters within the range
    quarters = data["quarters_num"]
    valid_quarters = quarters[(quarters .>= start_num) .& (quarters .<= end_num)]
    unique_quarters = sort(unique(valid_quarters))
    
    models = []
    for (idx, quarter_num) in enumerate(unique_quarters)
        quarter_date = BIT.num2date(quarter_num)
        
        # Check if we can get at least one observation (otherwise skip this starting quarter)
        forecast_date_first = lastdayofmonth(quarter_date + Month(0))
        forecast_num_first = BIT.date2num(forecast_date_first)
        
        if forecast_num_first > end_num
            continue  # Skip this starting quarter entirely
        end
        
        current_idx = idx
        end_idx = length(unique_quarters)

        quarters_diff = end_idx - current_idx

        if quarters_diff < 1
            continue
        end
        # Set T to the difference, with a maximum of 12
        T = min(quarters_diff, 12)
        

        # Initialize model with parameters for this quarter
        initial_conditions = load("./src/utils/parameters_initial_conditions_data/netherlands_households_own_firms/initial_conditions/" * 
                                 string(year(quarter_date)) * "Q" * string(Dates.quarterofyear(quarter_date)) * ".jld2")
        
        # Merge user parameters with defaults
        parameters = load("./src/utils/parameters_initial_conditions_data/netherlands_households_own_firms/parameters/" * 
                                           string(year(quarter_date)) * "Q" * string(Dates.quarterofyear(quarter_date)) * ".jld2")
        
        # Initialize model
        if model_type == "optimal_consumption"
            model = BIT.initialise_model(parameters, initial_conditions, T;
                                        optimal_consumption = true,
                                        empirical_distribution = empirical_distribution,
                                        conditional_forecasts = conditional_forecasts)
        else
            model = BIT.initialise_model(parameters, initial_conditions, T;
                                        empirical_distribution = empirical_distribution,
                                        conditional_forecasts = conditional_forecasts)
        end
        
        push!(models,model)
    end
    
    return models
end


start_date = DateTime(2010, 03, 31)
end_date = DateTime(2013, 12, 31)


models = get_models(
    start_date,
    end_date;
    model_type = "base",
    empirical_distribution = false,
    conditional_forecasts = false,
)


calibration_parameters = Dict(
    "mpc_y" => 0.5,
    "mpc_k" => 0.03
)

@time f1 = run_abm_simulations_with_parameters(
    models,
    calibration_parameters,
    start_date,
    end_date;
    num_simulations = 2,
    model_type = "base",
    abmx = false,
    empirical_distribution = false,
    conditional_forecasts = false,
);

calibration_parameters = Dict(
    "mpc_y" => 0.8,
    "mpc_k" => 0.03
)

@time f2 = run_abm_simulations_with_parameters(
    models,
    calibration_parameters,
    start_date,
    end_date;
    num_simulations = 10,
    model_type = "base",
    abmx = false,
    empirical_distribution = false,
    conditional_forecasts = false,
);