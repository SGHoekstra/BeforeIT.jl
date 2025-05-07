# Load the necessary packages
import BeforeIT as BIT
using Dates
using CSV
using HDF5
using JLD2
using FileIO
using MAT


function load_timeseries_for_calibrator(
    start_date::DateTime,
    end_date::DateTime;
    apply_transformations::Bool = true,
    data_path::String = "./src/utils/calibration_data/netherlands/data/1996.mat",
    repetitions::Int = 1
)
    # Load the data
    data = matread(data_path)["data"]
    
    # Extract variables used in the script
    variables = [
        "real_gdp_quarterly",
        "gdp_deflator_growth_quarterly",
        "real_household_consumption_quarterly", 
        "real_fixed_capitalformation_quarterly",
        "euribor"
    ]
    
    start_num = BIT.date2num(start_date)
    end_num = BIT.date2num(end_date)
    
    # Filter quarters within the range
    quarters = data["quarters_num"]
    valid_quarters = quarters[(quarters .>= start_num) .& (quarters .< end_num)]
    unique_quarters = sort(unique(valid_quarters))
    
    # Number of starting quarters and max forecast horizon
    N = length(unique_quarters)
    forecast_horizon = 12  # 12 quarters ahead as specified
    
    # Determine the maximum T (number of time steps + 1)
    T = forecast_horizon + 1
    
    # Number of variables
    M = length(variables)
    
    # Initialize output matrix - N*M columns, each repeated 'repetitions' times
    output = zeros(T, N * M * repetitions)
    
    # Process each starting quarter
    for (idx, quarter_num) in enumerate(unique_quarters)
        quarter_date = BIT.num2date(quarter_num)
        
        # For each quarter, get the next 12 quarters
        for h in 0:forecast_horizon
            # Calculate the forecast date
            forecast_date = lastdayofmonth(quarter_date + Month(3 * h))
            forecast_num = BIT.date2num(forecast_date)
            
            # Skip if beyond end date
            if forecast_num > end_num
                # Leave as zeros (padding)
                continue
            end
            
            # Extract and transform data for each variable
            for (var_idx, var_name) in enumerate(variables)
                # Find the data for this quarter
                data_idx = findall(data["quarters_num"] .== forecast_num)
                
                if !isempty(data_idx)
                    value = data[var_name][data_idx][1]
                    
                    # Apply transformations if requested
                    if apply_transformations
                        if var_name == "real_gdp_quarterly" || 
                           var_name == "real_household_consumption_quarterly" || 
                           var_name == "real_fixed_capitalformation_quarterly"
                            value = log(value)
                        elseif var_name == "gdp_deflator_growth_quarterly"
                            value = log(1 + value)
                        elseif var_name == "euribor"
                            value = (1 + value)^(1/4)
                        end
                    end
                    
                    # Calculate base column index: Variables per quarter are adjacent
                    base_col_idx = (idx - 1) * M + var_idx
                    
                    # Repeat the value for each repetition
                    for rep in 1:repetitions
                        # Calculate the actual column index including repetition
                        # For each base column, we'll have 'repetitions' adjacent columns with the same value
                        col_idx = (base_col_idx - 1) * repetitions + rep
                        output[h+1, col_idx] = value
                    end
                end
            end
        end
    end
    
    return output
end

start_date = DateTime(2010, 03, 31)
end_date = DateTime(2013, 12, 31)

# Example usage:
data_matrix = Bit.load_timeseries_for_calibrator(start_date, end_date; repetitions = 2)