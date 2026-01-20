# Aggregate Forecast Results - Tables (CSV and LaTeX)
# Aggregates per-country forecast results into cross-country summaries

using CSV, DataFrames, Statistics, Printf
import BeforeIT as Bit

# =============================================================================
# CONFIGURATION
# =============================================================================

TABLE_TYPES = [
    "rmse_abm", "bias_abm", "rmse_ar", "bias_ar",
    "rmse_validation_abm", "bias_validation_abm",
    "rmse_validation_var", "bias_validation_var",
    "rmse_abm_canvas", "bias_abm_canvas",
    "rmse_validation_abm_canvas", "bias_validation_abm_canvas"
]

BASE_VARIABLES = ["Real GDP", "GDP Deflator Growth", "Real Consumption", "Real Investment", "Euribor"]
VALIDATION_VARIABLES = ["Real GDP", "GDP Deflator Growth", "Real Gov Consumption", "Real Exports",
                        "Real Imports", "Real GDP (EA)", "GDP Deflator Growth (EA)", "Euribor"]
HORIZONS = [1, 2, 4, 8, 12]

get_variables(t) = occursin("validation", t) ? VALIDATION_VARIABLES : BASE_VARIABLES

# =============================================================================
# MAIN SCRIPT
# =============================================================================

@info "Aggregating forecast results..."

countries = Bit.discover_countries_with_predictions()
@info "Found $(length(countries)) countries"

# Create output directories
mkpath("analysis/tabs/multicountry_forecast_results/countries")
mkpath("analysis/tabs/multicountry_forecast_results/latex")

# Process each table type
for table_type in TABLE_TYPES
    variables = get_variables(table_type)

    # Collect data from all countries
    country_data = Dict{String, DataFrame}()
    for country in countries
        csv_file = "data/$(country)/analysis/$(table_type).csv"
        if isfile(csv_file)
            try
                country_data[country] = CSV.read(csv_file, DataFrame)
            catch e
                @warn "Failed to load $csv_file"
            end
        end
    end

    if isempty(country_data)
        @warn "No data for $table_type"
        continue
    end

    valid_countries = collect(keys(country_data))
    n_countries = length(valid_countries)
    n_variables = length(variables)

    # Create Country × Variable summary (averaged over horizons)
    summary_matrix = fill(NaN, n_countries, n_variables)
    for (i, country) in enumerate(valid_countries)
        df = country_data[country]
        for (j, var) in enumerate(variables)
            if var in names(df)
                vals = filter(!isnan, df[!, var])
                if !isempty(vals)
                    summary_matrix[i, j] = mean(vals)
                end
            end
        end
    end

    # Export Country × Variable CSV
    summary_df = DataFrame(summary_matrix, variables)
    summary_df.Country = valid_countries
    select!(summary_df, :Country, Not(:Country))
    CSV.write("analysis/tabs/multicountry_forecast_results/countries/$(table_type)_countries.csv", summary_df)

    # Create Horizon × Variable LaTeX table (averaged over countries)
    n_horizons = length(HORIZONS)
    mean_vals = fill(NaN, n_horizons, n_variables)
    std_errs = fill(NaN, n_horizons, n_variables)

    for (i, h) in enumerate(HORIZONS)
        for (j, var) in enumerate(variables)
            vals = Float64[]
            for country in valid_countries
                df = country_data[country]
                if var in names(df)
                    rows = df[df.Horizon .== "$(h)q", :]
                    if nrow(rows) == 1 && !ismissing(rows[1, var]) && !isnan(rows[1, var])
                        push!(vals, rows[1, var])
                    end
                end
            end
            if !isempty(vals)
                mean_vals[i, j] = mean(vals)
                std_errs[i, j] = std(vals) / sqrt(length(vals))
            end
        end
    end

    # Write LaTeX table
    open("analysis/tabs/multicountry_forecast_results/latex/cross_country_$(table_type).tex", "w") do f
        println(f, "% Cross-country aggregated table for $table_type")
        for (i, h) in enumerate(HORIZONS)
            row = ["$(h)q"]
            for j in 1:n_variables
                if !isnan(mean_vals[i,j]) && !isnan(std_errs[i,j])
                    push!(row, @sprintf("%.1f(%.1f)", mean_vals[i,j], std_errs[i,j]))
                else
                    push!(row, "N/A")
                end
            end
            println(f, join(row, " & "), i < n_horizons ? " \\\\ " : "")
        end
    end

    @info "✓ $table_type: $(n_countries) countries"
end

@info "Done."
