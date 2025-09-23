# Multi-Country Forecast Performance Analysis
# This script aggregates forecast RMSE/bias tables from all countries
# and creates cross-country comparison heatmaps and summary tables

using CSV, DataFrames, Statistics, Plots
import BeforeIT as Bit

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Table types to process
TABLE_TYPES = [
    "rmse_abm",
    "bias_abm",
    "rmse_ar",
    "bias_ar",
    "rmse_validation_abm",
    "bias_validation_abm",
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

    heatmap_dir = joinpath(output_dir, "heatmaps")
    mkpath(heatmap_dir)

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
        else
            title_text = "RMSE Improvement (%): $(uppercase(replace(table_type, "_" => " ")))"
            colormap = :viridis  # Sequential colormap for RMSE (better performance in lighter colors)
        end

        # Create heatmap
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
        savefig(p, joinpath(heatmap_dir, filename))
        @info "Saved heatmap: $filename"
    end
end

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

"""
    export_summary_tables(summary, output_dir)

Export summary matrices as CSV files.

# Arguments
- `summary`: Dictionary with summary matrices
- `output_dir`: Output directory for CSV files

# Returns
- Nothing (saves CSV files)
"""
function export_summary_tables(summary, output_dir)
    @info "Exporting summary tables as CSV files"

    tables_dir = joinpath(output_dir, "tables")
    mkpath(tables_dir)

    for (table_type, (countries, variables, matrix)) in summary
        # Create DataFrame
        df = DataFrame(matrix, variables)
        df.Country = countries

        # Move Country column to first position
        select!(df, :Country, Not(:Country))

        # Save CSV
        filename = "$(table_type)_summary.csv"
        csv_path = joinpath(tables_dir, filename)
        CSV.write(csv_path, df)
        @info "Saved summary table: $filename"
    end
end

# =============================================================================
# MAIN ANALYSIS SCRIPT
# =============================================================================

@info "Starting multi-country forecast performance analysis"

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

    # Create summary matrices (average over horizons)
    summary = create_summary_matrices(data, TABLE_TYPES)

    if isempty(summary)
        @error "No summary data could be created"
        exit(1)
    end

    # Create output directories
    tables_output_dir = "analysis/tabs/multicountry_forecast_results"
    figs_output_dir = "analysis/figs/multicountry/forecast_performance"
    mkpath(tables_output_dir)
    mkpath(figs_output_dir)

    # Export summary tables
    export_summary_tables(summary, tables_output_dir)

    # Create heatmaps
    create_forecast_heatmaps(summary, figs_output_dir)

catch e
    @error "Analysis failed: $e"
    rethrow(e)
end