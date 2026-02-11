# Aggregate Forecast Results - Figures (Heatmaps)
# Creates heatmaps for visual comparison of forecast performance across countries

using CSV, DataFrames, Statistics, Plots
import BeforeIT as Bit

# =============================================================================
# MODEL VARIANT CONFIGURATION
# =============================================================================
# Change this to create heatmaps for different model variants.
# Reads from: data/{country}/analysis/{MODEL_VARIANT}/
# Writes to:  analysis/figs/multicountry/forecast_performance/{MODEL_VARIANT}/heatmaps/
#
# Options:
#   MODEL_VARIANT = "base"           # Standard BeforeIT model
#   MODEL_VARIANT = "growth_rate"    # GrowthRateAR1 extension
#   MODEL_VARIANT = "canvas"         # CANVAS extension

MODEL_VARIANT = "growth_rate"

# =============================================================================
# CONFIGURATION
# =============================================================================

TABLE_TYPES = [
    # Absolute RMSE + bias
    "rmse_abm", "bias_abm",
    "rmse_validation_abm", "bias_validation_abm",
    # Relative to AR/VAR benchmark
    "rmse_abm_vs_ar", "rmse_ar", "bias_ar",
    "rmse_validation_abm_vs_var", "rmse_validation_var", "bias_validation_var",
    # Variant vs base ABM (may not exist for base variant)
    "rmse_abm_vs_base",
    "rmse_validation_abm_vs_base",
]

BASE_VARIABLES = ["Real GDP", "GDP Deflator Growth", "Real Consumption", "Real Investment", "Euribor"]
VALIDATION_VARIABLES = ["Real GDP", "GDP Deflator Growth", "Real Gov Consumption", "Real Exports",
                        "Real Imports", "Real GDP (EA)", "GDP Deflator Growth (EA)", "Euribor"]

get_variables(t) = occursin("validation", t) ? VALIDATION_VARIABLES : BASE_VARIABLES

# =============================================================================
# MAIN SCRIPT
# =============================================================================

@info "Creating forecast heatmaps for variant: $(MODEL_VARIANT)..."

countries = Bit.discover_countries_with_predictions()
@info "Found $(length(countries)) countries"

# Output directory (variant-specific)
output_dir = "analysis/figs/multicountry/forecast_performance/$(MODEL_VARIANT)/heatmaps"
mkpath(output_dir)

for table_type in TABLE_TYPES
    variables = get_variables(table_type)

    # Collect data from all countries (from variant subfolder)
    country_data = Dict{String, DataFrame}()
    for country in countries
        csv_file = "data/$(country)/analysis/$(MODEL_VARIANT)/$(table_type).csv"
        if isfile(csv_file)
            try
                df = CSV.read(csv_file, DataFrame)
                # Validate: RMSE files must have non-negative values (catch stale relative-format files)
                if startswith(table_type, "rmse_") && !occursin("vs_", table_type)
                    numeric_cols = [c for c in names(df) if c != "Horizon" && eltype(df[!, c]) <: Number]
                    if any(any(x -> !ismissing(x) && !isnan(x) && x < 0, df[!, c]) for c in numeric_cols)
                        @warn "Skipping $csv_file: contains negative values in absolute RMSE (stale file?)"
                        continue
                    end
                end
                country_data[country] = df
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
    is_raw_rmse = table_type in ["rmse_abm", "rmse_ar", "rmse_validation_abm", "rmse_validation_var"]

    valid_vals = matrix[.!isnan.(matrix)]
    if is_bias || !is_raw_rmse
        # Diverging colormap centered at zero
        max_abs = maximum(abs.(valid_vals))
        p = heatmap(matrix,
            color=:RdBu, clims=(-max_abs, max_abs), colorbar=false,
            size=(plot_width, plot_height), margin=15Plots.mm,
            xticks=(1:n_variables, clean_vars), yticks=(1:n_countries, valid_countries),
            xrotation=45)
    else
        # Sequential colormap for raw RMSE (light yellow → orange → red)
        p = heatmap(matrix,
            color=:YlOrRd, colorbar=false,
            size=(plot_width, plot_height), margin=15Plots.mm,
            xticks=(1:n_variables, clean_vars), yticks=(1:n_countries, valid_countries),
            xrotation=45)
    end

    # Add value annotations with adaptive text color
    for i in 1:n_countries, j in 1:n_variables
        val = matrix[i, j]
        if !isnan(val)
            txt = is_bias ? string(round(val, digits=4)) : string(round(val, digits=1))
            if is_bias || !is_raw_rmse
                # Diverging: dark text near center, white at extremes
                txt_color = abs(val) / max_abs > 0.65 ? :white : :black
            else
                # Sequential: dark text on light cells, white on dark cells
                min_val, max_val = extrema(valid_vals)
                range = max_val - min_val
                normalized = range > 0 ? (val - min_val) / range : 0.0
                txt_color = normalized > 0.7 ? :white : :black
            end
            annotate!(j, i, text(txt, 6, txt_color, :center))
        end
    end

    savefig(p, joinpath(output_dir, "$(table_type)_heatmap.png"))
    @info "✓ $table_type"
end

@info "Done."
