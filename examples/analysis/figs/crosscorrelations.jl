# Single-Country Cross-Correlation Analysis
# Creates cross-correlation and autocorrelation plots for a single country

using StatsBase, LinearAlgebra, Statistics, Dates
using Plots, JLD2, FileIO
import BeforeIT as Bit

include("../../../src/utils/correlation_utils.jl")

# =============================================================================
# CONFIGURATION
# =============================================================================

COUNTRY = "it"  # Country code
T = 32
N_SIMS = 10
CORRELATION_LAGS = 15
AUTOCORR_LAGS = 20

PLOT_VARIABLES = [
    "real_capitalformation_quarterly",
    "wages_quarterly",
    "real_household_consumption_quarterly",
    "gdp_deflator_quarterly"
]

# =============================================================================
# MAIN SCRIPT
# =============================================================================

@info "Cross-Correlation Analysis for $COUNTRY"

calibration = Bit.load_calibration_data(COUNTRY)
real_data = calibration.data
folder = "data/$(COUNTRY)/abm_predictions"
output_folder = "analysis/figs/$(COUNTRY)"
mkpath(output_folder)

# Load prediction files
files = filter(f -> endswith(f, ".jld2"), readdir(folder))
if isempty(files)
    error("No prediction files found in $folder")
end

first_pred = load(joinpath(folder, first(files)))["predictions_dict"]
variables = filter(v -> endswith(v, "_quarterly"), keys(first_pred))
n_quarters = length(files)
gdp_size = size(first_pred["real_gdp_quarterly"])

# Create HP filter cache and process data
hp_cache = create_hp_filter_cache(real_data, collect(variables), 10)

_, crosscorr, autocorr, _ = process_all_simulation_data(
    folder, collect(variables), gdp_size, N_SIMS, n_quarters,
    CORRELATION_LAGS, 20, AUTOCORR_LAGS, PLOT_VARIABLES[1:min(3,length(PLOT_VARIABLES))], 10
)

mean_xcorr, std_xcorr, mean_autocorr, std_autocorr, _ =
    calculate_statistics(crosscorr, autocorr, Dict(), collect(variables), 10, 20)

real_crosscorr, real_autocorr, _ =
    process_real_data_correlations(real_data, hp_cache, collect(variables), CORRELATION_LAGS, AUTOCORR_LAGS)

# Create cross-correlation plot
cross_lags = collect(-CORRELATION_LAGS:CORRELATION_LAGS)
p1 = plot(layout=(2, 2), size=(1200, 800), plot_title="Cross-Correlations with Real GDP")

for (k, var) in enumerate(PLOT_VARIABLES[1:min(4,length(PLOT_VARIABLES))])
    if !haskey(mean_xcorr, var) || !haskey(real_crosscorr, var)
        continue
    end
    plot!(p1, subplot=k, cross_lags, vec(mean_xcorr[var]), ribbon=vec(std_xcorr[var]),
          label="ABM", color=:steelblue, linewidth=2)
    plot!(p1, subplot=k, cross_lags, vec(real_crosscorr[var]),
          label="Real", color=:crimson, linewidth=2)
    title!(p1, subplot=k, replace(var, "_quarterly" => ""))
    hline!(p1, [0], subplot=k, color=:gray, linestyle=:dot, label=false)
end
savefig(p1, joinpath(output_folder, "crosscorrelations_abm.png"))
@info "✓ Saved cross-correlation plot"

# Create autocorrelation plot
auto_lags = collect(0:AUTOCORR_LAGS)
p2 = plot(layout=(2, 2), size=(1200, 800), plot_title="Autocorrelations")

for (k, var) in enumerate(PLOT_VARIABLES[1:min(4,length(PLOT_VARIABLES))])
    if !haskey(mean_autocorr, var)
        continue
    end
    plot!(p2, subplot=k, auto_lags, vec(mean_autocorr[var]), ribbon=vec(std_autocorr[var]),
          label="ABM", color=:steelblue, linewidth=2)
    if haskey(real_autocorr, var)
        plot!(p2, subplot=k, auto_lags, vec(real_autocorr[var]),
              label="Real", color=:crimson, linewidth=2)
    end
    title!(p2, subplot=k, replace(var, "_quarterly" => ""))
    hline!(p2, [0], subplot=k, color=:gray, linestyle=:dot, label=false)
end
savefig(p2, joinpath(output_folder, "autocorrelations_abm.png"))
@info "✓ Saved autocorrelation plot"

@info "Done."
