# Aggregate Forecast Results - Figures (Heatmaps)
# Creates heatmaps for visual comparison of forecast performance across countries

using CSV, DataFrames, Statistics, Plots
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

get_variables(t) = occursin("validation", t) ? VALIDATION_VARIABLES : BASE_VARIABLES

# =============================================================================
# MAIN SCRIPT
# =============================================================================

@info "Creating forecast heatmaps..."

countries = Bit.discover_countries_with_predictions()
@info "Found $(length(countries)) countries"

output_dir = "analysis/figs/multicountry/forecast_performance/heatmaps"
mkpath(output_dir)

for table_type in TABLE_TYPES
    variables = get_variables(table_type)

    # Collect data from all countries
    country_data = Dict{String, DataFrame}()
    for country in countries
        csv_file = "data/$(country)/analysis/$(table_type).csv"
        if isfile(csv_file)
            try
                country_data[country] = CSV.read(csv_file, DataFrame)
            catch; end
        end
    end

    if isempty(country_data)
        continue
    end

    valid_countries = collect(keys(country_data))
    n_countries = length(valid_countries)
    n_variables = length(variables)

    # Create summary matrix (averaged over horizons)
    matrix = fill(NaN, n_countries, n_variables)
    for (i, country) in enumerate(valid_countries)
        df = country_data[country]
        for (j, var) in enumerate(variables)
            if var in names(df)
                vals = filter(!isnan, df[!, var])
                if !isempty(vals)
                    matrix[i, j] = mean(vals)
                end
            end
        end
    end

    # Skip if all NaN
    if all(isnan, matrix)
        continue
    end

    # Create heatmap
    clean_vars = [replace(v, " " => "\n") for v in variables]
    plot_width = max(600, n_variables * 120)
    plot_height = max(400, n_countries * 40)

    # Determine colormap and title
    is_bias = occursin("bias", table_type)
    is_raw_rmse = table_type in ["rmse_ar", "rmse_validation_var"]

    if is_bias
        title = "Bias: $(uppercase(replace(table_type, "_" => " ")))"
    elseif is_raw_rmse
        title = "RMSE: $(uppercase(replace(table_type, "_" => " ")))"
    else
        title = "RMSE Improvement (%): $(uppercase(replace(table_type, "_" => " ")))"
    end

    valid_vals = matrix[.!isnan.(matrix)]
    if is_bias || !is_raw_rmse
        # Diverging colormap centered at zero
        max_abs = maximum(abs.(valid_vals))
        p = heatmap(matrix, title=title, xlabel="Variables", ylabel="Countries",
            color=:RdBu, clims=(-max_abs, max_abs),
            size=(plot_width, plot_height), margin=15Plots.mm,
            xticks=(1:n_variables, clean_vars), yticks=(1:n_countries, valid_countries),
            xrotation=45)
    else
        # Sequential colormap for raw RMSE
        p = heatmap(matrix, title=title, xlabel="Variables", ylabel="Countries",
            color=cgrad(:inferno, rev=true),
            size=(plot_width, plot_height), margin=15Plots.mm,
            xticks=(1:n_variables, clean_vars), yticks=(1:n_countries, valid_countries),
            xrotation=45)
    end

    # Add value annotations
    for i in 1:n_countries, j in 1:n_variables
        val = matrix[i, j]
        if !isnan(val)
            txt = is_bias ? string(round(val, digits=4)) : string(round(val, digits=1))
            annotate!(j, i, text(txt, 6, :black, :center))
        end
    end

    savefig(p, joinpath(output_dir, "$(table_type)_summary_heatmap.png"))
    @info "✓ $table_type"
end

@info "Done."
