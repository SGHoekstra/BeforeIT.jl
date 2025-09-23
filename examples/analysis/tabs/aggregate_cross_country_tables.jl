# Cross-Country Table Aggregation Script
# This script aggregates forecasting performance across all successful countries
# to create 4 summary tables showing average performance by variable and horizon

using Statistics, Printf
using DelimitedFiles, CSV, DataFrames

# =============================================================================
# CONFIGURATION
# =============================================================================

# Table types to aggregate (corresponding to the 4 functions)
TABLE_TYPES = [
    "rmse_abm",
    "rmse_validation_abm",
    "rmse_ar",
    "rmse_validation_var"
]

# Variable names for different table types
BASE_VARIABLE_NAMES = [
    "Real GDP",
    "GDP Deflator Growth",
    "Real Consumption",
    "Real Investment",
    "Euribor"
]

VALIDATION_VARIABLE_NAMES = [
    "Real GDP",
    "GDP Deflator Growth",
    "Real Gov Consumption",
    "Real Exports",
    "Real Imports",
    "Real GDP (EA)",
    "GDP Deflator Growth (EA)",
    "Euribor"
]

# Helper function to get variable names for a table type
function get_variable_names(table_type)
    if occursin("validation", table_type)
        return VALIDATION_VARIABLE_NAMES
    else
        return BASE_VARIABLE_NAMES
    end
end

# Horizons
HORIZONS = [1, 2, 4, 8, 12]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

"""
    discover_successful_countries()

Find all countries that have generated table files.

# Returns
- Vector of country names that have completed table generation
"""
function discover_successful_countries()
    countries = String[]

    # Look for countries with at least one table type generated
    data_dir = "/Users/steven/Github/BeforeIT.jl/data"

    for entry in readdir(data_dir)
        country_analysis_dir = joinpath(data_dir, entry, "analysis")
        if isdir(country_analysis_dir)
            # Check if this country has generated any tables
            table_files = filter(f -> endswith(f, ".csv"), readdir(country_analysis_dir))
            if !isempty(table_files)
                push!(countries, entry)
            end
        end
    end

    @info "Found $(length(countries)) countries with generated tables: $(join(countries, ", "))"
    return sort(countries)
end

"""
    read_csv_table(filepath, table_type)

Read a CSV table file and extract numerical values.

# Arguments
- `filepath`: Path to the .csv file
- `table_type`: Type of table to determine expected variable names

# Returns
- Matrix of extracted values (horizons × variables)
"""
function read_csv_table(filepath, table_type)
    variable_names = get_variable_names(table_type)

    if !isfile(filepath)
        @warn "Table file not found: $filepath"
        return fill(NaN, length(HORIZONS), length(variable_names))
    end

    try
        # Read CSV file
        df = CSV.read(filepath, DataFrame)

        # Initialize result matrix
        values = fill(NaN, length(HORIZONS), length(variable_names))

        # Extract data for each horizon and variable
        for (i, horizon) in enumerate(HORIZONS)
            horizon_str = "$(horizon)q"
            horizon_rows = df[df.Horizon .== horizon_str, :]

            if nrow(horizon_rows) == 1
                for (j, var_name) in enumerate(variable_names)
                    if var_name in names(df)
                        val = horizon_rows[1, var_name]
                        if !ismissing(val) && !isnan(val)
                            values[i, j] = val
                        end
                    end
                end
            end
        end

        return values

    catch e
        @error "Failed to read CSV table file $filepath: $e"
        return fill(NaN, length(HORIZONS), length(variable_names))
    end
end

"""
    aggregate_tables(countries, table_type)

Aggregate tables of a specific type across countries.

# Arguments
- `countries`: Vector of country names
- `table_type`: Type of table to aggregate (e.g., "rmse_abm")

# Returns
- Matrix of average values across countries (horizons × variables)
- Matrix of standard errors (horizons × variables)
"""
function aggregate_tables(countries, table_type)
    n_horizons = length(HORIZONS)
    variable_names = get_variable_names(table_type)
    n_variables = length(variable_names)
    n_countries = length(countries)

    # Collect data from all countries
    all_data = Array{Float64,3}(undef, n_horizons, n_variables, n_countries)

    valid_countries = 0
    for (k, country) in enumerate(countries)
        filepath = "/Users/steven/Github/BeforeIT.jl/data/$(country)/analysis/$(table_type).csv"
        country_data = read_csv_table(filepath, table_type)

        # Check if we got valid data
        if !all(isnan.(country_data))
            all_data[:, :, k] = country_data
            valid_countries += 1
        else
            all_data[:, :, k] .= NaN
        end
    end

    if valid_countries == 0
        @warn "No valid data found for table type: $table_type"
        return fill(NaN, n_horizons, n_variables), fill(NaN, n_horizons, n_variables)
    end

    # Calculate means and standard errors across countries
    mean_values = fill(NaN, n_horizons, n_variables)
    std_errors = fill(NaN, n_horizons, n_variables)

    for i in 1:n_horizons
        for j in 1:n_variables
            country_values = all_data[i, j, :]
            valid_values = filter(!isnan, country_values)

            if !isempty(valid_values)
                mean_values[i, j] = mean(valid_values)
                std_errors[i, j] = std(valid_values) / sqrt(length(valid_values))
            end
        end
    end

    @info "Aggregated $table_type across $valid_countries countries"
    return mean_values, std_errors
end

"""
    write_aggregated_table(table_type, mean_values, std_errors, output_dir)

Write an aggregated table to a LaTeX file.

# Arguments
- `table_type`: Type of table (e.g., "rmse_abm")
- `mean_values`: Matrix of mean values (horizons × variables)
- `std_errors`: Matrix of standard errors (horizons × variables)
- `output_dir`: Output directory
"""
function write_aggregated_table(table_type, mean_values, std_errors, output_dir)
    mkpath(output_dir)

    variable_names = get_variable_names(table_type)
    output_file = joinpath(output_dir, "cross_country_$(table_type).tex")

    open(output_file, "w") do f
        # Write table header
        println(f, "% Cross-country aggregated table for $table_type")
        println(f, "% Averaged across successful countries")
        println(f, "")

        # Write table rows
        for (i, horizon) in enumerate(HORIZONS)
            row_data = String[]
            push!(row_data, "$(horizon)q")

            for j in 1:length(variable_names)
                mean_val = mean_values[i, j]
                std_err = std_errors[i, j]

                if !isnan(mean_val) && !isnan(std_err)
                    # Format: mean(std_error)
                    formatted_val = @sprintf("%.1f(%.1f)", mean_val, std_err)
                else
                    formatted_val = "N/A"
                end

                push!(row_data, formatted_val)
            end

            # Write row
            if i < length(HORIZONS)
                println(f, join(row_data, " & "), " \\\\ ")
            else
                println(f, join(row_data, " & "))
            end
        end
    end

    @info "Saved aggregated table: $output_file"
end

"""
    write_summary_csv(countries, all_results, output_dir)

Write a comprehensive CSV summary of all results.

# Arguments
- `countries`: Vector of country names
- `all_results`: Dictionary containing all aggregated results
- `output_dir`: Output directory
"""
function write_summary_csv(countries, all_results, output_dir)
    # Create a comprehensive summary CSV
    summary_file = joinpath(output_dir, "cross_country_summary.csv")

    # Prepare data for CSV
    rows = []

    for table_type in TABLE_TYPES
        if haskey(all_results, table_type)
            mean_vals, std_errs = all_results[table_type]
            variable_names = get_variable_names(table_type)

            for (i, horizon) in enumerate(HORIZONS)
                for (j, variable) in enumerate(variable_names)
                    push!(rows, [
                        table_type,
                        horizon,
                        variable,
                        length(countries),
                        mean_vals[i, j],
                        std_errs[i, j]
                    ])
                end
            end
        end
    end

    # Write CSV
    open(summary_file, "w") do f
        println(f, "table_type,horizon,variable,n_countries,mean_value,std_error")
        for row in rows
            println(f, join(row, ","))
        end
    end

    @info "Saved summary CSV: $summary_file"
end

# =============================================================================
# MAIN AGGREGATION SCRIPT
# =============================================================================

@info "=== Cross-Country Table Aggregation ==="
@info "Aggregating forecasting performance tables across countries..."

# Discover successful countries
successful_countries = discover_successful_countries()

if isempty(successful_countries)
    @error "No countries with generated tables found!"
    exit(1)
end

# Output directory
output_dir = "/Users/steven/Github/BeforeIT.jl/analysis/tabs"

# Aggregate each table type
all_results = Dict{String, Tuple{Matrix{Float64}, Matrix{Float64}}}()

for table_type in TABLE_TYPES
    @info "Processing table type: $table_type"

    mean_values, std_errors = aggregate_tables(successful_countries, table_type)
    all_results[table_type] = (mean_values, std_errors)

    # Write individual aggregated table
    write_aggregated_table(table_type, mean_values, std_errors, output_dir)
end

# Write comprehensive summary
write_summary_csv(successful_countries, all_results, output_dir)

