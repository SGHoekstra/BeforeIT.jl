# Comprehensive Multi-Country Forecast Performance Analysis
# This script combines all forecast analysis functionality:
# - Heatmaps for visual comparison (Country × Variable matrices)
# - LaTeX tables for publications (Horizon × Variable matrices with standard errors)
# - CSV summaries for programmatic access
# - Processes both RMSE and bias tables

using CSV, DataFrames, Statistics, Plots, Printf
import BeforeIT as Bit

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Control flags
GENERATE_HEATMAPS = true        # Set to true to generate heatmaps (PNG files)
GENERATE_LATEX_TABLES = true   # Set to true to generate LaTeX tables (TEX files)
GENERATE_CSV_SUMMARIES = true  # Set to true to generate CSV summary files

# Table types to process - ALL 8 table types (4 RMSE + 4 BIAS)
TABLE_TYPES = [
    "rmse_abm",
    "bias_abm",
    "rmse_ar",
    "bias_ar",
    "rmse_validation_abm",
    "bias_validation_abm",
    "rmse_validation_var",
    "bias_validation_var"
]

# LaTeX table types - ALL tables for cross-country averaging (both RMSE and BIAS)
LATEX_TABLE_TYPES = [
    "rmse_abm",
    "bias_abm",
    "rmse_ar",
    "bias_ar",
    "rmse_validation_abm",
    "bias_validation_abm",
    "rmse_validation_var",
    "bias_validation_var"
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

# Horizons (for LaTeX tables)
HORIZONS = [1, 2, 4, 8, 12]

# Helper function to get variable names for a table type
function get_variable_names(table_type)
    if occursin("validation", table_type)
        return VALIDATION_VARIABLE_NAMES
    else
        return BASE_VARIABLE_NAMES
    end
end

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

"""
    load_forecast_tables(countries, table_types)

Load forecast RMSE/bias CSV tables from all countries.

# Arguments
- `countries`: Vector of country codes
- `table_types`: Vector of table type names (e.g., ["rmse_abm", "bias_abm"])

# Returns
- Dictionary with structure: data[table_type][country] = DataFrame
"""
function load_forecast_tables(countries, table_types)
    @info "Loading forecast tables for $(length(countries)) countries"

    data = Dict{String, Dict{String, DataFrame}}()

    for table_type in table_types
        data[table_type] = Dict{String, DataFrame}()

        for country in countries
            csv_file = "data/$(country)/analysis/$(table_type).csv"

            if isfile(csv_file)
                try
                    df = CSV.read(csv_file, DataFrame)
                    data[table_type][country] = df
                    @info "Loaded $table_type for $country"
                catch e
                    @warn "Failed to load $csv_file: $e"
                end
            else
                @warn "File not found: $csv_file"
            end
        end

        @info "Loaded $(table_type): $(length(data[table_type])) countries"
    end

    return data
end

# =============================================================================
# COUNTRY × VARIABLE ANALYSIS (for heatmaps)
# =============================================================================

"""
    create_summary_matrices(data, table_types)

Create country × variable matrices by averaging over horizons.

# Arguments
- `data`: Dictionary from load_forecast_tables()
- `table_types`: Vector of table type names

# Returns
- Dictionary with summary matrices: summary[table_type] = (countries, variables, matrix)
"""
function create_summary_matrices(data, table_types)
    @info "Creating summary matrices averaged over horizons"

    summary = Dict{String, Tuple{Vector{String}, Vector{String}, Matrix{Float64}}}()

    for table_type in table_types
        if !haskey(data, table_type) || isempty(data[table_type])
            @warn "No data found for table type: $table_type"
            continue
        end

        # Get variable names for this specific table type
        variable_names = get_variable_names(table_type)

        # Get available countries for this table type
        available_countries = collect(keys(data[table_type]))
        n_countries = length(available_countries)
        n_variables = length(variable_names)

        # Initialize matrix
        summary_matrix = fill(NaN, n_countries, n_variables)

        for (i, country) in enumerate(available_countries)
            df = data[table_type][country]

            # Average over horizons for each variable
            for (j, var_name) in enumerate(variable_names)
                if var_name in names(df)
                    values = df[!, var_name]
                    # Calculate mean, ignoring NaN values
                    valid_values = filter(!isnan, values)
                    if !isempty(valid_values)
                        summary_matrix[i, j] = mean(valid_values)
                    end
                end
            end
        end

        summary[table_type] = (available_countries, variable_names, summary_matrix)
        @info "Created summary matrix for $table_type: $(n_countries) countries × $(n_variables) variables"
    end

    return summary
end

# =============================================================================
# HORIZON × VARIABLE ANALYSIS (for LaTeX tables)
# =============================================================================

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

Aggregate tables of a specific type across countries (average over countries).

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
        filepath = "data/$(country)/analysis/$(table_type).csv"
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

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

"""
    create_forecast_heatmaps(summary, output_dir)

Create heatmaps for forecast performance comparison across countries.

# Arguments
- `summary`: Dictionary with summary matrices from create_summary_matrices()
- `output_dir`: Output directory for saving heatmaps

# Returns
- Nothing (saves PNG files)
"""
function create_forecast_heatmaps(summary, output_dir)
    @info "Creating forecast performance heatmaps"

    mkpath(output_dir)

    for (table_type, (countries, variables, matrix)) in summary
        @info "Creating heatmap for $table_type"

        n_countries, n_variables = size(matrix)

        # Clean variable names for display
        clean_var_names = [replace(v, " " => "\n") for v in variables]

        # Dynamic sizing
        plot_width = max(600, n_variables * 120)
        plot_height = max(400, n_countries * 40)

        # Determine title and colormap based on table type
        if occursin("bias", table_type)
            title_text = "Forecast Bias: $(uppercase(replace(table_type, "_" => " ")))"
            colormap = :RdBu  # Red-blue diverging colormap for bias (red = positive bias, blue = negative bias)
        elseif table_type in ["rmse_abm", "rmse_validation_abm"]
            title_text = "RMSE Improvement (%): $(uppercase(replace(table_type, "_" => " ")))"
            colormap = :RdBu  # Red-blue diverging colormap for percentage improvements (red = deterioration, blue = improvement)
        else
            title_text = "RMSE Values: $(uppercase(replace(table_type, "_" => " ")))"
            colormap = cgrad(:inferno, rev=true)  # Inverted inferno: light for low values, dark for high values
        end

        # Create heatmap with appropriate color limits
        if colormap == :RdBu
            # For diverging colormaps, center white at zero by using symmetric limits
            max_abs_value = max(abs(minimum(matrix[.!isnan.(matrix)])), abs(maximum(matrix[.!isnan.(matrix)])))
            p = heatmap(
                matrix,
                title = title_text,
                xlabel = "Variables",
                ylabel = "Countries",
                color = colormap,
                clims = (-max_abs_value, max_abs_value),  # Force symmetric limits around zero
                size = (plot_width, plot_height),
                margin = 15Plots.mm,
                bottom_margin = 25Plots.mm,
                left_margin = 25Plots.mm,
                right_margin = 15Plots.mm,
                top_margin = 20Plots.mm,
                xticks = (1:n_variables, clean_var_names),
                yticks = (1:n_countries, countries),
                xrotation = 45,
                guidefontsize = 10,
                tickfontsize = 8,
                titlefontsize = 12
            )
        else
            # For sequential colormaps, use default scaling
            p = heatmap(
                matrix,
                title = title_text,
                xlabel = "Variables",
                ylabel = "Countries",
                color = colormap,
                size = (plot_width, plot_height),
                margin = 15Plots.mm,
                bottom_margin = 25Plots.mm,
                left_margin = 25Plots.mm,
                right_margin = 15Plots.mm,
                top_margin = 20Plots.mm,
                xticks = (1:n_variables, clean_var_names),
                yticks = (1:n_countries, countries),
                xrotation = 45,
                guidefontsize = 10,
                tickfontsize = 8,
                titlefontsize = 12
            )
        end

        # Add value annotations
        for i in 1:n_countries
            for j in 1:n_variables
                val = matrix[i, j]
                if !isnan(val)
                    # Format based on table type
                    if occursin("bias", table_type)
                        display_val = string(round(val, digits=4))
                    else
                        display_val = string(round(val, digits=1))
                    end

                    # Choose font size based on density
                    font_size = n_countries > 15 || n_variables > 5 ? 6 : 7
                    annotate!(j, i, text(display_val, font_size, :black, :center))
                end
            end
        end

        # Save heatmap
        filename = "$(table_type)_summary_heatmap.png"
        savefig(p, joinpath(output_dir, filename))
        @info "Saved heatmap: $filename"
    end
end

# =============================================================================
# LATEX TABLE FUNCTIONS
# =============================================================================

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

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

"""
    export_summary_tables(summary, output_dir)

Export summary matrices as CSV files (Country × Variable matrices).

# Arguments
- `summary`: Dictionary with summary matrices
- `output_dir`: Output directory for CSV files

# Returns
- Nothing (saves CSV files)
"""
function export_summary_tables(summary, output_dir)
    @info "Exporting Country×Variable tables as CSV files"

    mkpath(output_dir)

    for (table_type, (countries, variables, matrix)) in summary
        # Create DataFrame
        df = DataFrame(matrix, variables)
        df.Country = countries

        # Move Country column to first position
        select!(df, :Country, Not(:Country))

        # Save CSV with _countries naming convention
        filename = "$(table_type)_countries.csv"
        csv_path = joinpath(output_dir, filename)
        CSV.write(csv_path, df)
        @info "Saved Country×Variable table: $filename"
    end
end

"""
    write_summary_csv(countries, all_results, output_dir)

Write a comprehensive CSV summary of all aggregated results (Horizon × Variable).

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

    for table_type in LATEX_TABLE_TYPES
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
# MAIN ANALYSIS SCRIPT
# =============================================================================

@info "Starting comprehensive multi-country forecast performance analysis"

# Discover countries with forecast tables
countries_with_predictions = Bit.discover_countries_with_predictions()

if isempty(countries_with_predictions)
    @error "No countries with prediction data found!"
    exit(1)
end

@info "Found $(length(countries_with_predictions)) countries with predictions"

try
    # Load all forecast tables
    data = load_forecast_tables(countries_with_predictions, TABLE_TYPES)

    # =======================================================================
    # COUNTRY × VARIABLE ANALYSIS (for heatmaps and country comparison)
    # =======================================================================

    if GENERATE_HEATMAPS || GENERATE_CSV_SUMMARIES
        @info "Creating Country × Variable analysis (averaged over horizons)"

        # Create summary matrices (average over horizons)
        summary = create_summary_matrices(data, TABLE_TYPES)

        if isempty(summary)
            @warn "No Country × Variable summary data could be created"
        else
            # Create output directories
            base_output_dir = "analysis/tabs/multicountry_forecast_results"
            countries_output_dir = joinpath(base_output_dir, "countries")
            heatmaps_output_dir = "analysis/figs/multicountry/forecast_performance/heatmaps"
            mkpath(countries_output_dir)
            mkpath(heatmaps_output_dir)

            # Export Country × Variable summary tables
            if GENERATE_CSV_SUMMARIES
                export_summary_tables(summary, countries_output_dir)
            end

            # Create heatmaps
            if GENERATE_HEATMAPS
                create_forecast_heatmaps(summary, heatmaps_output_dir)
            end
        end
    end

    # =======================================================================
    # HORIZON × VARIABLE ANALYSIS (for LaTeX tables and cross-country averages)
    # =======================================================================

    if GENERATE_LATEX_TABLES
        @info "Creating Horizon × Variable analysis (averaged over countries)"

        # Aggregate each table type (both RMSE and BIAS for LaTeX)
        base_output_dir = "analysis/tabs/multicountry_forecast_results"
        latex_output_dir = joinpath(base_output_dir, "latex")
        mkpath(latex_output_dir)
        all_results = Dict{String, Tuple{Matrix{Float64}, Matrix{Float64}}}()

        for table_type in LATEX_TABLE_TYPES
            @info "Processing table type: $table_type"

            mean_values, std_errors = aggregate_tables(countries_with_predictions, table_type)
            all_results[table_type] = (mean_values, std_errors)

            # Write individual aggregated table
            write_aggregated_table(table_type, mean_values, std_errors, latex_output_dir)
        end

        # Note: Summary CSV generation removed as requested
    end

    @info "Comprehensive forecast performance analysis completed successfully"

catch e
    @error "Analysis failed: $e"
    rethrow(e)
end