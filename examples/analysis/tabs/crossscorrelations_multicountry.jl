using StatsBase, LinearAlgebra, Statistics, Dates
using JLD2, FileIO, CSV, DataFrames
using Plots
import BeforeIT as Bit

include("../../../src/utils/correlation_utils.jl")


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Control flags - set these to control which parts of the script run
RUN_SIMULATION = true   # Set to true to generate simulations using migrated data
RUN_ANALYSIS = true     # Set to true to run cross-country statistical correlation analysis

# Simulation parameters
FIRST_DATE = DateTime(2010, 03, 31)
LAST_DATE = DateTime(2023, 12, 31)
T = 16
N_SIMS = 20
SCALE = 0.0005

# Analysis constants
DEFAULT_HORIZON = 20
MAX_SECTORS = 10
CORRELATION_LAGS = 15
AUTOCORR_LAGS = 20

# Key variables for cross-country comparison
COMPARISON_VARIABLES = [
    "real_gdp_quarterly",
    "real_household_consumption_quarterly",
    "real_capitalformation_quarterly",
    "wages_quarterly",
    "gdp_deflator_quarterly"
]

CYCLE_VARIABLES = [
    "real_gdp_quarterly",
    "real_capitalformation_quarterly",
    "real_household_consumption_quarterly"
]

# Statistical analysis parameters
CONFIDENCE_LEVEL = 0.95

# Default RMSE function
rmse(diff) = sqrt(mean(diff.^2))

# =============================================================================
# STATISTICAL DISTANCE FUNCTIONS
# =============================================================================


"""
    compute_lag_wise_rmse(sim_correlations, real_correlation)

Compute RMSE between simulated and real correlations for each lag separately.

# Arguments
- `sim_correlations`: Matrix of simulated correlations (lags × simulations)
- `real_correlation`: Vector of real correlation (lags)

# Returns
- Vector of RMSE values for each lag
"""
function compute_lag_wise_rmse(sim_correlations, real_correlation)
    # Validate input dimensions
    if isempty(real_correlation)
        error("real_correlation is empty")
    end

    if ndims(sim_correlations) != 2
        error("sim_correlations must be a 2D array, got $(ndims(sim_correlations))D with size $(size(sim_correlations))")
    end

    n_lags = length(real_correlation)
    n_sim_lags, n_sims = size(sim_correlations)

    # Check dimension compatibility
    if n_lags != n_sim_lags
        error("Dimension mismatch: real_correlation has $n_lags lags but sim_correlations has $n_sim_lags lags")
    end

    if n_sims == 0
        error("sim_correlations has no simulation columns")
    end

    @info "Processing RMSE calculation: $n_lags lags × $n_sims simulations"

    lag_rmse = zeros(n_lags)

    # Compute RMSE for each lag across all simulations
    for lag_idx in 1:n_lags
        # Bounds check
        if lag_idx > n_sim_lags
            error("lag_idx $lag_idx exceeds sim_correlations first dimension $n_sim_lags")
        end

        sim_values = sim_correlations[lag_idx, :]  # All simulations at this lag
        real_value = real_correlation[lag_idx]

        # Check for invalid values
        if any(isnan, sim_values) || any(isinf, sim_values)
            @warn "Found NaN or Inf values in simulation data at lag $lag_idx"
            lag_rmse[lag_idx] = NaN
            continue
        end

        if isnan(real_value) || isinf(real_value)
            @warn "Found NaN or Inf value in real data at lag $lag_idx"
            lag_rmse[lag_idx] = NaN
            continue
        end

        diff = sim_values .- real_value
        lag_rmse[lag_idx] = sqrt(mean(diff.^2))
    end

    return lag_rmse
end
# =============================================================================
# SIMULATION SETUP SCRIPT
# =============================================================================

if RUN_SIMULATION
    @info "Starting simulation setup for all analysis countries using migrated data"

    countries = Bit.discover_countries_with_calibration()

    for country in countries
        @info "Setting up simulations for $country"

        try
            # Load calibration data from migrated structure
            calibration = Bit.load_calibration_data(country)

            default_folder = "data/$(lowercase(country))"

            # Check if folder exists (should already exist from migration)
            if !isdir(default_folder)
                error("Migrated data folder not found for $country at $default_folder. Run migrate_calibration_data.jl first.")
            end

            # Check if parameters and initial conditions already exist
            param_dir = joinpath(default_folder, "parameters")
            init_dir = joinpath(default_folder, "initial_conditions")

            param_exists = isdir(param_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(param_dir)))
            init_exists = isdir(init_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(init_dir)))

            if !param_exists || !init_exists
                error("Parameters or initial conditions missing for $country. Run migrate_calibration_data.jl first.")
            end

            # Run simulations using the migrated data
            #Bit.save_all_simulations(default_folder; T = T, n_sims = N_SIMS, simulation_suffix = "simulations_longrun")

            # Generate predictions from simulations
            # Use calibration.data as real_data for predictions
            Bit.save_all_predictions_from_sims(default_folder, calibration.data; simulation_suffix = "simulations_longrun", prediction_suffix = "abm_predictions_longrun")

            @info "Simulation data setup completed successfully for $country"

        catch e
            @error "Failed to setup simulation data for $country: $e"
        end
    end

    @info "Simulation setup completed for all countries"
end

# =============================================================================
# CORRELATION PROCESSING FUNCTIONS
# =============================================================================

"""
    process_country_correlations(real_data, model_folder, variables)

Process correlations for a single country using real data from migrated calibration
and simulation data from BeforeIT.jl.

# Arguments
- `real_data`: Real economic data from migrated calibration
- `model_folder`: Path to simulation prediction files
- `variables`: Variables to analyze

# Returns
- Dictionary with correlation results for the country
"""
function process_country_correlations(real_data, model_folder, variables)
    @info "Processing correlations using migrated real data"

    # Load simulation data and get structure
    model_files = filter(f -> endswith(f, ".jld2"), readdir(model_folder))
    if isempty(model_files)
        error("No simulation files found in $model_folder")
    end

    first_model_path = joinpath(model_folder, first(model_files))
    first_model = load(first_model_path)["predictions_dict"]
    gdp_size = size(first_model["real_gdp_quarterly"])

    # Calculate derived parameters
    n_years = Dates.year(LAST_DATE) - Dates.year(FIRST_DATE) + 1
    n_quarters = 4 * n_years


    
    # Get all quarterly variables from simulation
    all_sim_variables = filter(name -> endswith(name, "_quarterly"), keys(first_model))
    analysis_variables = intersect(variables, all_sim_variables)

    if isempty(analysis_variables)
        @warn "No matching variables found between requested and simulation variables"
        return Dict()
    end

    @info "Processing $(length(analysis_variables)) variables: $(join(analysis_variables, ", "))"

    # Create HP filter cache for real data from CalibrateBeforeIT.jl
    hp_cache = create_hp_filter_cache(real_data, analysis_variables, MAX_SECTORS)

    # Process simulation data using migrated functions
    _, crosscorr, autocorr, _ = process_all_simulation_data(
        model_folder, analysis_variables, gdp_size, N_SIMS, n_quarters,
        CORRELATION_LAGS, DEFAULT_HORIZON, AUTOCORR_LAGS, CYCLE_VARIABLES, MAX_SECTORS
    )

    # Process real data correlations using CalibrateBeforeIT.jl data
    real_crosscorr, real_autocorr, _ = process_real_data_correlations(
        real_data, hp_cache, analysis_variables, CORRELATION_LAGS, AUTOCORR_LAGS
    )

    # Compute statistical comparisons for each variable
    variable_results = Dict{String, Any}()

    for var in analysis_variables
        if !haskey(crosscorr, var) || !haskey(real_crosscorr, var)
            @warn "Skipping $var - missing correlation data"
            continue
        end

        try
            # Get simulation and real correlations
            sim_cross = crosscorr[var]  # (lags × simulations)
            sim_auto = autocorr[var]    # (lags × simulations)
            real_cross = vec(real_crosscorr[var])  # (lags,)
            real_auto = vec(real_autocorr[var])    # (lags,)

            # Log dimensions for debugging
            @info "Processing variable $var:"
            @info "  sim_cross size: $(size(sim_cross))"
            @info "  sim_auto size: $(size(sim_auto))"
            @info "  real_cross length: $(length(real_cross))"
            @info "  real_auto length: $(length(real_auto))"

            # Validate dimensions before processing
            if isempty(real_cross) || isempty(real_auto)
                @warn "Skipping $var - empty real correlation data"
                continue
            end

            if ndims(sim_cross) != 2 || ndims(sim_auto) != 2
                @warn "Skipping $var - simulation correlations are not 2D arrays"
                continue
            end

            # Create lag vectors
            cross_lags = collect(-CORRELATION_LAGS:CORRELATION_LAGS)
            auto_lags = collect(0:AUTOCORR_LAGS)

            # Validate expected dimensions
            expected_cross_lags = length(cross_lags)
            expected_auto_lags = length(auto_lags)

            if length(real_cross) != expected_cross_lags
                @warn "Skipping $var - real cross-correlation has $(length(real_cross)) lags, expected $expected_cross_lags"
                continue
            end

            if length(real_auto) != expected_auto_lags
                @warn "Skipping $var - real auto-correlation has $(length(real_auto)) lags, expected $expected_auto_lags"
                continue
            end

            if size(sim_cross, 1) != expected_cross_lags
                @warn "Skipping $var - sim cross-correlation has $(size(sim_cross, 1)) lags, expected $expected_cross_lags"
                continue
            end

            if size(sim_auto, 1) != expected_auto_lags
                @warn "Skipping $var - sim auto-correlation has $(size(sim_auto, 1)) lags, expected $expected_auto_lags"
                continue
            end

            # Compute lag-wise RMSE for cross-correlations and autocorrelations
            cross_lag_rmse = compute_lag_wise_rmse(sim_cross, real_cross)
            auto_lag_rmse = compute_lag_wise_rmse(sim_auto, real_auto)

            # Check for NaN values in RMSE results
            if any(isnan, cross_lag_rmse) || any(isnan, auto_lag_rmse)
                @warn "Variable $var produced NaN RMSE values, treating as failed"
                continue
            end

            # Calculate overall RMSE as mean of lag-wise RMSE
            cross_overall_rmse = mean(cross_lag_rmse)
            auto_overall_rmse = mean(auto_lag_rmse)

            # Store results for this variable
            variable_results[var] = Dict(
                "cross_correlations" => Dict(
                    "overall_rmse" => cross_overall_rmse,
                    "lag_rmse" => cross_lag_rmse,
                    "lags" => cross_lags,
                    "real_correlation" => real_cross,
                    "sim_correlations" => sim_cross
                ),
                "auto_correlations" => Dict(
                    "overall_rmse" => auto_overall_rmse,
                    "lag_rmse" => auto_lag_rmse,
                    "lags" => auto_lags,
                    "real_correlation" => real_auto,
                    "sim_correlations" => sim_auto
                )
            )

        catch e
            @error "Failed to process variable $var: $e"
            continue
        end
    end

    return Dict(
        "variables" => variable_results,
        "metadata" => Dict(
            "n_simulations" => N_SIMS,
            "correlation_lags" => CORRELATION_LAGS,
            "autocorr_lags" => AUTOCORR_LAGS,
            "confidence_level" => CONFIDENCE_LEVEL,
            "analysis_variables" => analysis_variables,
            "data_source" => "Migrated_BeforeIT_data"
        )
    )
end

"""
    create_lag_tables_per_variable(country_results, countries, variables)

Create tables with countries as rows and lags as columns for each variable.

# Arguments
- `country_results`: Dictionary of results by country
- `countries`: List of countries
- `variables`: List of variables

# Returns
- Dictionary with lag tables for each variable and correlation type
"""
function create_lag_tables_per_variable(country_results, countries, variables)
    @info "Creating lag tables per variable"

    lag_tables = Dict{String, Any}()

    for var in variables
        lag_tables[var] = Dict{String, Any}()

        for corr_type in ["cross_correlations", "auto_correlations"]
            # Get lag information from first available country
            first_country = nothing
            lags = nothing
            for country in countries
                if haskey(country_results, country) && haskey(country_results[country]["variables"], var)
                    first_country = country
                    lags = country_results[country]["variables"][var][corr_type]["lags"]
                    break
                end
            end

            if first_country === nothing
                @warn "No data found for variable $var in correlation type $corr_type"
                continue
            end

            n_countries = length(countries)
            n_lags = length(lags)

            # Initialize matrix: countries × lags
            rmse_matrix = fill(NaN, n_countries, n_lags)

            # Fill matrix with lag-wise RMSE data
            for (i, country) in enumerate(countries)
                if haskey(country_results, country) &&
                   haskey(country_results[country]["variables"], var) &&
                   haskey(country_results[country]["variables"][var], corr_type)

                    lag_rmse = country_results[country]["variables"][var][corr_type]["lag_rmse"]
                    rmse_matrix[i, :] = lag_rmse
                end
            end

            # Create DataFrame with countries as rows and lags as columns
            col_names = [string("Lag_", lag) for lag in lags]
            df = DataFrame(rmse_matrix, col_names)
            df.Country = countries

            # Move Country column to first position
            select!(df, :Country, Not(:Country))

            lag_tables[var][corr_type] = df
        end
    end

    return lag_tables
end

"""
    create_cross_country_matrices(country_results, countries, variables)

Create cross-country comparison matrices from individual country results.

# Arguments
- `country_results`: Dictionary of results by country
- `countries`: List of countries
- `variables`: List of variables

# Returns
- Dictionary with comparison matrices and exports results
"""
function create_cross_country_matrices(country_results, countries, variables)
    @info "Creating cross-country comparison matrices"

    n_countries = length(countries)
    n_variables = length(variables)

    # Initialize result matrices for correlation types
    results = Dict{String, Any}()

    for corr_type in ["cross_correlations", "auto_correlations"]
        # Initialize matrices for this correlation type (only overall RMSE now)
        results[corr_type] = Dict(
            "overall_rmse" => zeros(n_countries, n_variables)
        )
    end

    # Fill matrices with country data
    for (i, country) in enumerate(countries)
        if !haskey(country_results, country)
            @warn "No results for $country, filling with NaN"
            continue
        end

        country_data = country_results[country]

        for (j, var) in enumerate(variables)
            if !haskey(country_data["variables"], var)
                @warn "No results for $country.$var, filling with NaN"
                continue
            end

            var_data = country_data["variables"][var]

            # Process both cross-correlations and autocorrelations
            for corr_type in ["cross_correlations", "auto_correlations"]
                if !haskey(var_data, corr_type)
                    continue
                end

                corr_data = var_data[corr_type]
                overall_rmse = corr_data["overall_rmse"]

                # Fill overall RMSE value
                results[corr_type]["overall_rmse"][i, j] = overall_rmse
            end
        end
    end

    # Create DataFrames for easier analysis and export
    export_results = Dict{String, Any}()

    for corr_type in ["cross_correlations", "auto_correlations"]
        # Only overall RMSE now
        df = DataFrame(results[corr_type]["overall_rmse"], variables)
        df.country = countries
        export_results[corr_type] = df
    end

    return results, export_results
end

"""
    create_overall_rmse_heatmaps(comparison_matrices, countries, variables, output_dir)

Create heatmaps showing overall RMSE values for countries vs variables.

# Arguments
- `comparison_matrices`: Dictionary with comparison matrices
- `countries`: List of country names
- `variables`: List of variable names
- `output_dir`: Output directory for saving heatmaps

# Returns
- Nothing (saves heatmap files)
"""
function create_overall_rmse_heatmaps(comparison_matrices, countries, variables, output_dir)
    @info "Creating overall RMSE heatmaps"

    heatmap_dir = joinpath(output_dir, "heatmaps")
    mkpath(heatmap_dir)

    for corr_type in ["cross_correlations", "auto_correlations"]
        @info "Creating heatmap for $corr_type"

        # Get the RMSE matrix (countries × variables)
        rmse_matrix = comparison_matrices[corr_type]["overall_rmse"]

        # Create readable variable names for plot
        var_labels = [replace(replace(v, "_quarterly" => ""), "_" => " ") |> titlecase for v in variables]

        # Create heatmap with better dimensions
        n_countries = length(countries)
        n_vars = length(var_labels)

        # Dynamic sizing based on number of countries and variables
        plot_width = max(600, n_vars * 120)
        plot_height = max(400, n_countries * 40)

        p = heatmap(
            rmse_matrix,
            title = "$(titlecase(replace(corr_type, "_" => "-"))) RMSE: Countries vs Variables",
            xlabel = "Variables",
            ylabel = "Countries",
            color = :viridis,
            size = (plot_width, plot_height),
            margin = 15Plots.mm,
            bottom_margin = 20Plots.mm,
            left_margin = 20Plots.mm,
            right_margin = 10Plots.mm,
            top_margin = 15Plots.mm,
            xticks = (1:n_vars, var_labels),
            yticks = (1:n_countries, countries),
            xrotation = 45,
            guidefontsize = 10,
            tickfontsize = 8,
            titlefontsize = 12
        )

        # Add values as text annotations with better positioning
        for i in 1:n_countries
            for j in 1:n_vars
                val = rmse_matrix[i, j]
                if !isnan(val)
                    # Use smaller font for many data points
                    font_size = n_countries > 15 || n_vars > 5 ? 6 : 8
                    annotate!(j, i, text(string(round(val, digits=3)), font_size, :white, :center))
                end
            end
        end

        # Save heatmap
        filename = "$(corr_type)_overall_rmse_heatmap.png"
        savefig(p, joinpath(heatmap_dir, filename))
        @info "Saved heatmap: $filename"
    end
end

"""
    create_lag_rmse_heatmaps(lag_tables, countries, variables, output_dir)

Create heatmaps showing lag-wise RMSE values for countries vs lags for each variable.

# Arguments
- `lag_tables`: Dictionary with lag tables per variable
- `countries`: List of country names
- `variables`: List of variable names
- `output_dir`: Output directory for saving heatmaps

# Returns
- Nothing (saves heatmap files)
"""
function create_lag_rmse_heatmaps(lag_tables, countries, variables, output_dir)
    @info "Creating lag-wise RMSE heatmaps"

    heatmap_dir = joinpath(output_dir, "heatmaps", "by_variable")
    mkpath(heatmap_dir)

    for var in variables
        if !haskey(lag_tables, var)
            @warn "No lag table found for variable $var, skipping"
            continue
        end

        var_clean = replace(replace(var, "_quarterly" => ""), "_" => " ") |> titlecase

        for corr_type in ["cross_correlations", "auto_correlations"]
            if !haskey(lag_tables[var], corr_type)
                @warn "No lag table found for $var.$corr_type, skipping"
                continue
            end

            @info "Creating lag heatmap for $var - $corr_type"

            # Get the lag table DataFrame
            lag_df = lag_tables[var][corr_type]

            # Extract country names and lag columns
            country_col = lag_df.Country
            lag_cols = names(lag_df)[names(lag_df) .!= "Country"]

            # Convert to matrix (countries × lags)
            lag_matrix = Matrix(select(lag_df, Not(:Country)))

            # Create lag labels
            lag_labels = [replace(col, "Lag_" => "") for col in lag_cols]

            # Standardized sizing - use autocorrelations formatting approach
            n_countries_lag = length(country_col)
            n_lags = length(lag_labels)

            # Use consistent dimensions regardless of correlation type
            # Base dimensions on autocorrelations format (21 lags standard)
            if corr_type == "auto_correlations"
                # Keep original sizing for autocorrelations
                lag_plot_width = max(800, n_lags * 50)
                lag_plot_height = max(400, n_countries_lag * 40)
            else
                # Use autocorrelations-style sizing for cross-correlations
                # Scale based on autocorrelations reference (21 lags)
                base_width_per_lag = 800 / 21  # ~38 pixels per lag for autocorr baseline
                lag_plot_width = max(800, Int(round(n_lags * base_width_per_lag)))
                lag_plot_height = max(400, n_countries_lag * 40)
            end

            # Create heatmap with standardized formatting
            # Use consistent title formatting for both correlation types
            title_text = if corr_type == "auto_correlations"
                "$var_clean: Autocorrelations Lag-wise RMSE"
            else
                "$var_clean: Cross-correlations Lag-wise RMSE"
            end

            p = heatmap(
                lag_matrix,
                title = title_text,
                xlabel = "Lags",
                ylabel = "Countries",
                color = :viridis,
                size = (lag_plot_width, lag_plot_height),
                margin = 15Plots.mm,
                bottom_margin = 15Plots.mm,
                left_margin = 20Plots.mm,
                right_margin = 10Plots.mm,
                top_margin = 15Plots.mm,
                xticks = (1:n_lags, lag_labels),
                yticks = (1:n_countries_lag, country_col),
                guidefontsize = 10,
                tickfontsize = 8,
                titlefontsize = 11
            )

            # Add values as text annotations with standardized sizing
            # Use consistent annotation thresholds regardless of correlation type
            if n_lags <= 25 && n_countries_lag <= 20
                for i in 1:n_countries_lag
                    for j in 1:n_lags
                        val = lag_matrix[i, j]
                        if !isnan(val)
                            # Use consistent font sizing based on autocorrelations approach
                            # Autocorrelations typically have 21 lags, use this as reference
                            font_size = if corr_type == "auto_correlations"
                                n_lags > 15 || n_countries_lag > 10 ? 5 : 6
                            else
                                # For cross-correlations, use smaller font due to more lags
                                n_lags > 21 || n_countries_lag > 10 ? 5 : 6
                            end
                            annotate!(j, i, text(string(round(val, digits=2)), font_size, :white, :center))
                        end
                    end
                end
            end

            # Save heatmap
            var_filename = replace(var, "_" => "")
            filename = "$(var_filename)_$(corr_type)_lag_rmse_heatmap.png"
            savefig(p, joinpath(heatmap_dir, filename))
            @info "Saved lag heatmap: $filename"
        end
    end
end

# =============================================================================
# CROSS-COUNTRY STATISTICAL ANALYSIS SCRIPT
# =============================================================================

if RUN_ANALYSIS
    @info "Starting cross-country statistical correlation analysis"
    countries = Bit.discover_countries_with_predictions()
    try
        # Initialize result storage
        country_results = Dict{String, Dict{String, Any}}()
        available_countries = String[]

        # Process each country
        for country in countries
            @info "Processing statistical analysis for $country"

            try
                # Load calibration data from migrated structure
                calibration = Bit.load_calibration_data(country)
                real_data = calibration.data

                # Check if simulation data exists
                default_folder = "data/$(lowercase(country))"
                model_folder = joinpath(default_folder, "abm_predictions_longrun")

                if !isdir(model_folder)
                    @warn "No simulation data found for $country at $model_folder, skipping"
                    continue
                end

                # Check if prediction files exist
                prediction_files = filter(f -> endswith(f, ".jld2"), readdir(model_folder))
                if isempty(prediction_files)
                    @warn "No prediction files found for $country in $model_folder, skipping"
                    continue
                end

                @info "Processing correlations for $country using migrated real data"
                @info "Found $(length(prediction_files)) prediction files for $country"

                # Process correlations using the integrated function
                result = process_country_correlations(
                    real_data, model_folder, COMPARISON_VARIABLES
                )

                # Check if any variables were successfully processed
                if isempty(result)
                    @warn "No variables could be processed for $country, skipping"
                    continue
                end

                # Count successfully processed variables
                n_variables = length(result)
                @info "Successfully processed $n_variables variables for $country"

                country_results[country] = result
                push!(available_countries, country)

                @info "Successfully processed $country"

            catch e
                @error "Failed to process $country: $e"
                @error "Error details: $(string(e))"
                if isa(e, BoundsError)
                    @error "BoundsError details: Attempted to access index $(e.i) in array"
                end
            end
        end

        if isempty(available_countries)
            @error "No countries could be processed successfully"
            return
        end

        @info "Successfully processed $(length(available_countries)) countries: $(join(available_countries, ", "))"

        # Create cross-country comparison matrices
        @info "Creating cross-country comparison matrices and export files"
        comparison_matrices, export_dataframes = create_cross_country_matrices(
            country_results, available_countries, COMPARISON_VARIABLES
        )

        # Create lag tables per variable
        @info "Creating lag tables per variable"
        lag_tables = create_lag_tables_per_variable(
            country_results, available_countries, COMPARISON_VARIABLES
        )

        # Create output directories
        output_dir = "analysis/tabs/cross_country_results"
        mkpath(output_dir)

        # Create figure output directory
        fig_output_dir = "analysis/figs/multicountry/crosscorrelations"
        mkpath(fig_output_dir)

        # Create visualization heatmaps
        @info "Creating heatmap visualizations"
        try
            create_overall_rmse_heatmaps(comparison_matrices, available_countries, COMPARISON_VARIABLES, fig_output_dir)
            create_lag_rmse_heatmaps(lag_tables, available_countries, COMPARISON_VARIABLES, fig_output_dir)
        catch e
            @warn "Failed to create heatmaps: $e"
        end

        # Save comprehensive results
        save(joinpath(output_dir, "cross_country_correlation_analysis.jld2"),
             "country_results", country_results,
             "comparison_matrices", comparison_matrices,
             "export_dataframes", export_dataframes,
             "lag_tables", lag_tables,
             "available_countries", available_countries,
             "comparison_variables", COMPARISON_VARIABLES,
             "rmse_function", "RMSE",
             "metadata", Dict(
                 "n_simulations" => N_SIMS,
                 "correlation_lags" => CORRELATION_LAGS,
                 "autocorr_lags" => AUTOCORR_LAGS,
                 "analysis_date" => now(),
                 "script_version" => "2.1"
             ))

        # Export CSV files for easy analysis
        @info "Exporting CSV files for cross-country comparison"

        for corr_type in ["cross_correlations", "auto_correlations"]
            df = export_dataframes[corr_type]
            filename = "$(corr_type)_overall_rmse.csv"
            CSV.write(joinpath(output_dir, filename), df)
        end

        # Export lag tables per variable
        @info "Exporting lag tables per variable"

        for var in COMPARISON_VARIABLES
            for corr_type in ["cross_correlations", "auto_correlations"]
                if haskey(lag_tables, var) && haskey(lag_tables[var], corr_type)
                    df = lag_tables[var][corr_type]
                    filename = "$(var)_$(corr_type)_rmse_by_lag.csv"
                    CSV.write(joinpath(output_dir, filename), df)
                end
            end
        end

        # Create summary report
        @info "Creating summary report"

        # Find best and worst performing countries by average RMSE
        summary_stats = Dict{String, Any}()

        for corr_type in ["cross_correlations", "auto_correlations"]
            overall_rmse_matrix = comparison_matrices[corr_type]["overall_rmse"]

            # Calculate average performance across variables for each country
            country_avg_rmse = [mean(overall_rmse_matrix[i, :]) for i in 1:length(available_countries)]

            # Find best and worst countries
            best_rmse_idx = argmin(country_avg_rmse)
            worst_rmse_idx = argmax(country_avg_rmse)

            summary_stats[corr_type] = Dict(
                "best_rmse_country" => available_countries[best_rmse_idx],
                "best_rmse_value" => country_avg_rmse[best_rmse_idx],
                "worst_rmse_country" => available_countries[worst_rmse_idx],
                "worst_rmse_value" => country_avg_rmse[worst_rmse_idx],
                "average_rmse_across_countries" => mean(country_avg_rmse)
            )
        end

        # Save summary
        save(joinpath(output_dir, "summary_statistics.jld2"), "summary", summary_stats)

        # Print summary to console
        @info "=== CROSS-COUNTRY CORRELATION ANALYSIS SUMMARY ==="
        @info "Countries analyzed: $(join(available_countries, ", "))"
        @info "Variables analyzed: $(join(COMPARISON_VARIABLES, ", "))"
        @info "Number of simulations per country: $N_SIMS"
        @info ""

        for corr_type in ["cross_correlations", "auto_correlations"]
            @info "$(uppercase(replace(corr_type, "_" => "-"))) RESULTS:"
            stats = summary_stats[corr_type]
            @info "  Best model fit (lowest RMSE): $(stats["best_rmse_country"]) ($(round(stats["best_rmse_value"], digits=4)))"
            @info "  Worst model fit (highest RMSE): $(stats["worst_rmse_country"]) ($(round(stats["worst_rmse_value"], digits=4)))"
            @info "  Average RMSE: $(round(stats["average_rmse_across_countries"], digits=4))"
            @info ""
        end

        @info "Results saved to: $output_dir"
        @info "Cross-country statistical correlation analysis completed successfully!"

    catch e
        @error "Cross-country statistical analysis failed: $e"
        rethrow(e)
    end
end