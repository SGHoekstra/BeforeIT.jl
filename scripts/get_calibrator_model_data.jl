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
    par_dict::Dict,
    start_date::DateTime,
    end_date::DateTime;
    num_simulations::Int = 10,
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

        model.prop.theta_UNION = par_dict["theta_UNION"]
        
        # Run simulations
        sims = BIT.ensemblerun(model, num_simulations; multi_threading = multi_threading)

        
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
    end_date::DateTime
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
        initial_conditions = load("./src/utils/parameters_initial_conditions_data/netherlands/initial_conditions/" * 
                                 string(year(quarter_date)) * "Q" * string(Dates.quarterofyear(quarter_date)) * ".jld2")
        
        # Merge user parameters with defaults
        parameters = load("./src/utils/parameters_initial_conditions_data/netherlands/parameters/" * 
                                           string(year(quarter_date)) * "Q" * string(Dates.quarterofyear(quarter_date)) * ".jld2")
        
        # Initialize model
        model = Bit.init_model(parameters, initial_conditions, T)
       
        
        push!(models,model)
    end
    
    return models
end

