using StatsBase, LinearAlgebra, Statistics, Dates
using Plots, StatsPlots
using MAT, JLD2, FileIO
import BeforeIT as Bit

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Control flags - set these to control which parts of the script run
RUN_SIMULATION = false   # Set to true to run simulation setup
RUN_ANALYSIS = true    # Set to true to run analysis

# Simulation parameters
COUNTRY = "italy"
CALIBRATION = Bit.ITALY_CALIBRATION
FIRST_DATE = DateTime(2010, 03, 31)
LAST_DATE = DateTime(2023, 12, 31)
T = 32
N_SIMS = 10
SCALE = 0.0005

# File and analysis constants
DEFAULT_FOLDER = "data/$(COUNTRY)"
DEFAULT_HORIZON = 20
MAX_SECTORS = 10
CORRELATION_LAGS = 15
AUTOCORR_LAGS = 20
CYCLE_PLOT_YLIM = (-20, 20)

PLOT_VARIABLES = [
    "real_capitalformation_quarterly",
    "wages_quarterly",
    "real_household_consumption_quarterly",
    "gdp_deflator_quarterly",
    "operating_surplus_quarterly"
]

CYCLE_VARIABLES = [
    "real_gdp_quarterly",
    "real_capitalformation_quarterly",
    "real_household_consumption_quarterly",
    "operating_surplus_quarterly"
]

# =============================================================================
# UTILITY FUNCTIONS (PLOTTING ONLY)
# =============================================================================

"""
Create cross-correlation plots.
"""
function create_crosscorr_plots(mean_xcorr, std_xcorr, real_crosscorr, output_folder)
    @info "Creating modernized cross-correlation plots"

    lags = collect(-CORRELATION_LAGS:CORRELATION_LAGS)
    cross_plots = plot(
        layout=(3, 2),
        size=(1400, 1000),
        plot_title="Cross-Correlations with Real GDP",
        plot_titlefontsize=16,
        margin=5Plots.mm,
        dpi=300
    )

    for (k, name) in enumerate(PLOT_VARIABLES)
        if !haskey(mean_xcorr, name) || !haskey(real_crosscorr, name)
            continue
        end

        # Model data with error bands (ribbon plot for better visibility)
        mean_vals = vec(mean_xcorr[name])
        std_vals = vec(std_xcorr[name])

        plot!(cross_plots, subplot=k, lags, mean_vals,
              yerror=std_vals,
              fillalpha=0.3,
              fillcolor=:steelblue,
              label="ABM Model",
              color=:steelblue,
              linewidth=2.5,
              linestyle=:solid)

        # Real data with contrasting color (solid line)
        plot!(cross_plots, subplot=k, lags, vec(real_crosscorr[name]),
              linewidth=3, color=:crimson, linestyle=:solid, label="Real Data")

        # Modern formatting
        formatted_name = format_variable_name(name)
        plot!(cross_plots, subplot=k,
              title=formatted_name,
              titlefontsize=12,
              titlefontweight=:bold,
              xlabel="Lags (Quarters)",
              ylabel="Cross-Correlation",
              labelfontsize=10,
              tickfontsize=9,
              legendfontsize=9,
              grid=true,
              gridwidth=1,
              gridcolor=:lightgray,
              gridalpha=0.5,
              framestyle=:box,
              background_color=:white,
              legend=:topright)

        # Add zero line for reference
        hline!(cross_plots, [0], subplot=k, color=:black, linestyle=:dot, linewidth=1, alpha=0.7, label=false)
        vline!(cross_plots, [0], subplot=k, color=:black, linestyle=:dot, linewidth=1, alpha=0.7, label=false)
    end

    output_path = joinpath(output_folder, "crosscorrelations_abm.png")
    savefig(cross_plots, output_path)
    @info "Modernized cross-correlation plots saved to $output_path"

    return cross_plots
end

"""
Create autocorrelation plots.
"""
function create_autocorr_plots(mean_autocorr, std_autocorr, real_autocorr, output_folder)
    @info "Creating modernized autocorrelation plots"

    lags = collect(0:AUTOCORR_LAGS)
    auto_plots = plot(
        layout=(3, 2),
        size=(1400, 1000),
        plot_title="Autocorrelations",
        plot_titlefontsize=16,
        margin=5Plots.mm,
        dpi=300
    )

    for (k, name) in enumerate(PLOT_VARIABLES)
        if !haskey(mean_autocorr, name)
            continue
        end

        # Model data with error bands (ribbon plot for better visibility)
        mean_vals = vec(mean_autocorr[name])
        std_vals = vec(std_autocorr[name])

        plot!(auto_plots, subplot=k, lags, mean_vals,
              yerror=std_vals,
              fillalpha=0.3,
              fillcolor=:steelblue,
              label="ABM Model",
              color=:steelblue,
              linewidth=2.5,
              linestyle=:solid)

        # Real data (if available) - solid line
        if haskey(real_autocorr, name) && !iszero(real_autocorr[name])
            plot!(auto_plots, subplot=k, lags, vec(real_autocorr[name]),
                  linewidth=3, color=:crimson, linestyle=:solid, label="Real Data")
        end

        # Modern formatting
        formatted_name = format_variable_name(name)
        plot!(auto_plots, subplot=k,
              title=formatted_name,
              titlefontsize=12,
              titlefontweight=:bold,
              xlabel="Lags (Quarters)",
              ylabel="Autocorrelation",
              labelfontsize=10,
              tickfontsize=9,
              legendfontsize=9,
              grid=true,
              gridwidth=1,
              gridcolor=:lightgray,
              gridalpha=0.5,
              framestyle=:box,
              background_color=:white,
              legend=:topright)

        # Add zero line for reference
        hline!(auto_plots, [0], subplot=k, color=:black, linestyle=:dot, linewidth=1, alpha=0.7, label=false)
    end

    output_path = joinpath(output_folder, "autocorrelations_abm.png")
    savefig(auto_plots, output_path)
    @info "Modernized autocorrelation plots saved to $output_path"

    return auto_plots
end

# =============================================================================
# SIMULATION SETUP SCRIPT
# =============================================================================

if RUN_SIMULATION
    @info "Starting simulation setup for $COUNTRY"

    try
        # Check if parameters and initial conditions already exist
        param_dir = joinpath(DEFAULT_FOLDER, "parameters")
        init_dir = joinpath(DEFAULT_FOLDER, "initial_conditions")

        param_exists = isdir(param_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(param_dir)))
        init_exists = isdir(init_dir) && !isempty(filter(f -> endswith(f, ".jld2"), readdir(init_dir)))

        if param_exists && init_exists
            @info "Parameters and initial conditions already exist, skipping generation"
        else
            @info "Generating parameters and initial conditions"
            # Save parameters and initial conditions
            Bit.save_all_params_and_initial_conditions(
                CALIBRATION,
                DEFAULT_FOLDER;
                scale = SCALE,
                first_calibration_date = FIRST_DATE,
                last_calibration_date = LAST_DATE,
            )
        end

        # Save simulations using existing parameters/initial conditions but with longrun suffix
        # Note: We'll need to manually modify the simulation and prediction storage

        # First run standard save_all_simulations (creates simulations/ folder)
        Bit.save_all_simulations(DEFAULT_FOLDER; T = T, n_sims = N_SIMS)

        # Then move/copy the results to longrun folders
        sim_dir = joinpath(DEFAULT_FOLDER, "simulations")
        sim_longrun_dir = joinpath(DEFAULT_FOLDER, "simulations_longrun")

        if isdir(sim_dir) && !isdir(sim_longrun_dir)
            cp(sim_dir, sim_longrun_dir)
            @info "Copied simulations to simulations_longrun folder"
        end

        # Align with real data for predictions (creates abm_predictions/ folder)
        real_data = CALIBRATION.data
        Bit.save_all_predictions_from_sims(DEFAULT_FOLDER, real_data)

        # Move predictions to longrun folder
        pred_dir = joinpath(DEFAULT_FOLDER, "abm_predictions")
        pred_longrun_dir = joinpath(DEFAULT_FOLDER, "abm_predictions_longrun")

        if isdir(pred_dir) && !isdir(pred_longrun_dir)
            cp(pred_dir, pred_longrun_dir)
            @info "Copied predictions to abm_predictions_longrun folder"
        end

        @info "Simulation data setup completed successfully"

    catch e
        error("Failed to setup simulation data: $e")
    end
end

# =============================================================================
# ANALYSIS SCRIPT
# =============================================================================

if RUN_ANALYSIS
    @info "Starting cross-correlation analysis for $COUNTRY"

    try
        # Setup paths and parameters
        output_folder = "./analysis/figs/$COUNTRY"

        # Get real data
        real_data = CALIBRATION.data

        # Calculate derived parameters
        n_years = year(LAST_DATE) - year(FIRST_DATE) + 1
        n_quarters = 4 * n_years

        # Setup paths - use longrun subfolder for analysis
        model_folder = joinpath(DEFAULT_FOLDER, "abm_predictions_longrun")
        mkpath(output_folder)

        # Load initial model to get structure
        model_files = filter(f -> endswith(f, ".jld2"), readdir(model_folder))
        if isempty(model_files)
            error("No model files found in $model_folder")
        end

        first_model_path = joinpath(model_folder, first(model_files))
        first_model = load(first_model_path)["predictions_dict"]
        gdp_size = size(first_model["real_gdp_quarterly"])

        # Get variable names
        variable_names = filter(name -> endswith(name, "_quarterly"), keys(first_model))

        # Create HP filter cache for real data
        hp_cache = create_hp_filter_cache(real_data, collect(variable_names), MAX_SECTORS)

        # Process simulation data (single pass)
        cyclesvar, crosscorr, autocorr, cycles_data = process_all_simulation_data(
            model_folder, collect(variable_names), gdp_size, N_SIMS, n_quarters,
            CORRELATION_LAGS, DEFAULT_HORIZON, AUTOCORR_LAGS, CYCLE_VARIABLES, MAX_SECTORS
        )

        # Calculate statistics
        mean_xcorr, std_xcorr, mean_autocorr, std_autocorr, mean_cyclesvar =
            calculate_statistics(crosscorr, autocorr, cyclesvar, collect(variable_names), MAX_SECTORS, DEFAULT_HORIZON)

        # Process real data correlations
        real_crosscorr, real_autocorr, real_stderr =
            process_real_data_correlations(real_data, hp_cache, collect(variable_names), CORRELATION_LAGS, AUTOCORR_LAGS)

        # Save correlation results for statistical analysis and cross-country comparison
        save_correlation_results(
            mean_xcorr, std_xcorr, mean_autocorr, std_autocorr, mean_cyclesvar,
            real_crosscorr, real_autocorr, real_stderr,
            COUNTRY, CORRELATION_LAGS, AUTOCORR_LAGS, collect(variable_names),
            N_SIMS, FIRST_DATE, LAST_DATE, output_folder
        )

        create_crosscorr_plots(mean_xcorr, std_xcorr, real_crosscorr, output_folder)
        create_autocorr_plots(mean_autocorr, std_autocorr, real_autocorr, output_folder)

        @info "Cross-correlation analysis completed successfully"

    catch e
        @error "Cross-correlation analysis failed: $e"
        rethrow(e)
    end
end
