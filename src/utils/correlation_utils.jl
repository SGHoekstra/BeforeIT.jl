import BeforeIT as Bit
using LinearAlgebra, Statistics, JLD2, Dates

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

"""
    hpfilter(y; λ = 1600.0)

Hodrick-Prescott filter implementation for business cycle analysis.

# Arguments
- `y`: Time series data
- `λ`: Smoothing parameter (default 1600 for quarterly data)

# Returns
- `trend`: Trend component
- `cycle`: Cyclical component
"""
function hpfilter(y; λ = 1600.0)
    T = length(y)

    # Create the second difference matrix
    D = zeros(T - 2, T)
    for i in 1:(T - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end

    # Calculate the trend
    trend = (I + λ .* (D' * D)) \ y

    # Calculate the cycle
    cycle = y - trend

    return trend, cycle
end

"""
    crosscor(x, y, maxlag = 0)

Cross-correlation function with normalization, similar to MATLAB's xcorr.

# Arguments
- `x`: First time series
- `y`: Second time series
- `maxlag`: Maximum lag to compute

# Returns
- Cross-correlation values at lags -maxlag:maxlag
"""
function crosscor(x, y, maxlag = 0)
    nx = length(x)
    ny = length(y)

    if nx != ny
        error("Inputs must be of the same length")
    end

    # Center the data
    x_centered = x .- mean(x)
    y_centered = y .- mean(y)

    # Calculate population standard deviations (biased estimator for consistency)
    x_std_dev = sqrt(sum(x_centered .^ 2) / nx)
    y_std_dev = sqrt(sum(y_centered .^ 2) / nx)

    lags = (-maxlag):maxlag
    xcorr_result = zeros(length(lags))

    for (i, lag) in enumerate(lags)
        if abs(lag) >= nx
            xcorr_result[i] = 0.0
        else
            if lag == 0
                # No lag case
                covariance = sum(x_centered .* y_centered) / nx
            elseif lag > 0
                # y leads x: correlate x[t] with y[t+lag]
                covariance = sum(x_centered[1:(end-lag)] .* y_centered[(lag+1):end]) / nx
            else  # lag < 0
                # x leads y: correlate x[t+|lag|] with y[t]
                shift = abs(lag)
                covariance = sum(x_centered[(shift+1):end] .* y_centered[1:(end-shift)]) / nx
            end

            # Normalize by standard deviations to get correlation
            xcorr_result[i] = covariance / (x_std_dev * y_std_dev)
        end
    end

    return xcorr_result
end

"""
    autocor(x, lags = 0:20)

Autocorrelation function similar to MATLAB's autocorr.

# Arguments
- `x`: Time series data
- `lags`: Range of lags to compute

# Returns
- Autocorrelation values at specified lags
"""
function autocor(x, lags = 0:20)
    nx = length(x)

    # Center the data (remove mean)
    x_centered = x .- mean(x)

    # Calculate population variance (biased estimator for consistency with MATLAB)
    x_var = sum(x_centered .^ 2) / nx

    acorr_result = zeros(length(lags))

    for (i, lag) in enumerate(lags)
        if lag == 0
            # Autocorrelation at lag 0 is always 1.0
            acorr_result[i] = 1.0
        elseif lag >= nx
            # Not enough data for this lag
            acorr_result[i] = 0.0
        else
            # Calculate autocorrelation for lag > 0 using consistent /nx normalization
            covariance = sum(x_centered[1:(end - lag)] .* x_centered[(lag + 1):end]) / nx
            acorr_result[i] = covariance / x_var
        end
    end

    return acorr_result
end

"""
    format_variable_name(name)

Format variable names for plot titles by removing suffixes and cleaning up.

# Arguments
- `name`: Variable name string

# Returns
- Formatted string suitable for plot titles
"""
function format_variable_name(name)
    # Remove quarterly suffix if present
    if length(name) > 9 && name[(end - 8):end] == "quarterly"
        str = name[1:(end - 10)]
    else
        str = name
    end

    # Replace underscores with spaces and capitalize
    str = replace(str, "_" => " ")
    str = titlecase(str)

    return str
end

"""
    save_correlation_results(mean_xcorr, std_xcorr, mean_autocorr, std_autocorr, mean_cyclesvar,
                            real_crosscorr, real_autocorr, real_stderr, country, correlation_lags,
                            autocorr_lags, variable_names, n_sims, first_date, last_date, output_folder)

Save comprehensive correlation analysis results for statistical testing and cross-country comparison.

# Arguments
- `mean_xcorr`: Mean cross-correlations from model simulations
- `std_xcorr`: Standard deviation of cross-correlations from model simulations
- `mean_autocorr`: Mean autocorrelations from model simulations
- `std_autocorr`: Standard deviation of autocorrelations from model simulations
- `mean_cyclesvar`: Mean cycle variance from model simulations
- `real_crosscorr`: Cross-correlations from real data
- `real_autocorr`: Autocorrelations from real data
- `real_stderr`: Standard errors from real data
- `country`: Country identifier string
- `correlation_lags`: Number of lags used for cross-correlations
- `autocorr_lags`: Number of lags used for autocorrelations
- `variable_names`: Vector of variable names analyzed
- `n_sims`: Number of simulations
- `first_date`: Start date of analysis period
- `last_date`: End date of analysis period
- `output_folder`: Output directory path

# Output
Saves results to `data/{country}/analysis/correlations.jld2`
"""
function save_correlation_results(mean_xcorr, std_xcorr, mean_autocorr, std_autocorr, mean_cyclesvar,
                                 real_crosscorr, real_autocorr, real_stderr, country, correlation_lags,
                                 autocorr_lags, variable_names, n_sims, first_date, last_date, output_folder)

    @info "Saving correlation results for statistical analysis"

    # Create comprehensive results structure
    correlation_results = Dict(
        "model" => Dict(
            "crosscorr" => Dict("mean" => mean_xcorr, "std" => std_xcorr),
            "autocorr" => Dict("mean" => mean_autocorr, "std" => std_autocorr),
            "cycles_variance" => mean_cyclesvar
        ),
        "real" => Dict(
            "crosscorr" => real_crosscorr,
            "autocorr" => real_autocorr,
            "cycles_variance" => real_stderr
        ),
        "metadata" => Dict(
            "country" => country,
            "correlation_lags" => correlation_lags,
            "autocorr_lags" => autocorr_lags,
            "variables" => variable_names,
            "n_simulations" => n_sims,
            "time_period" => (first_date, last_date),
            "timestamp" => now(),
            "lags_range_crosscorr" => (-correlation_lags:correlation_lags),
            "lags_range_autocorr" => (0:autocorr_lags)
        )
    )

    # Ensure analysis directory exists
    analysis_folder = dirname(output_folder)
    if !isdir(analysis_folder)
        mkpath(analysis_folder)
    end

    # Create analysis subdirectory if it doesn't exist
    analysis_dir = joinpath("data", country, "analysis")
    if !isdir(analysis_dir)
        mkpath(analysis_dir)
    end

    # Save results
    output_path = joinpath(analysis_dir, "correlations.jld2")
    save(output_path, "correlation_results", correlation_results)

    @info "Correlation results saved to $output_path"
    @info "Structure: model (crosscorr/autocorr mean±std, cycles_variance), real (crosscorr/autocorr, cycles_variance), metadata"

    return output_path
end

"""
    create_hp_filter_cache(real_data, variable_names, max_sectors = 10)

Create HP filter cache for real data to avoid recomputation.

# Arguments
- `real_data`: Dictionary containing real economic data
- `variable_names`: Collection of variable names to process
- `max_sectors`: Maximum number of sectors to process for multi-dimensional variables

# Returns
- `cache`: Dict with cached HP filter results
"""
function create_hp_filter_cache(real_data, variable_names, max_sectors = 10)
    @info "Creating HP filter cache for real data"

    cache = Dict{String, Any}()

    # Cache GDP data first (used as reference)
    for gdp_var in ["real_gdp_quarterly", "real_gdp"]
        if haskey(real_data, gdp_var)
            trend, cycle = hpfilter(real_data[gdp_var])
            cache[gdp_var] = (trend, cycle)
        end
    end

    # Cache other variables
    for name in variable_names
        if haskey(real_data, name)
            if ndims(real_data[name]) == 1
                trend, cycle = hpfilter(real_data[name])
                cache[name] = (trend, cycle)
            else
                cache[name] = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
                n_sectors = min(max_sectors, size(real_data[name], 2))
                for sector in 1:n_sectors
                    trend, cycle = hpfilter(real_data[name][:, sector])
                    cache[name][sector] = (trend, cycle)
                end
            end
        end
    end

    return cache
end

"""
    determine_gdp_reference(hp_cache)

Determine the appropriate GDP reference for correlations.

# Arguments
- `hp_cache`: HP filter cache dictionary

# Returns
- GDP variable name or nothing if not found
"""
function determine_gdp_reference(hp_cache)
    for gdp_var in ["real_gdp_quarterly", "real_gdp"]
        if haskey(hp_cache, gdp_var)
            return gdp_var
        end
    end
    return nothing
end

"""
    process_real_1d_variable!(crosscorr_data, autocorr_data, stderr_data, real_data, name, gdp_cycle, cached_data, correlation_lags, autocorr_lags)

Process single dimension real variable.

# Arguments
- `crosscorr_data`: Dictionary to store cross-correlation results
- `autocorr_data`: Dictionary to store autocorrelation results
- `stderr_data`: Dictionary to store standard error results
- `real_data`: Real economic data
- `name`: Variable name
- `gdp_cycle`: GDP cycle data for cross-correlation reference
- `cached_data`: Cached HP filter results for this variable
- `correlation_lags`: Number of lags for cross-correlation
- `autocorr_lags`: Number of lags for autocorrelation
"""
function process_real_1d_variable!(crosscorr_data, autocorr_data, stderr_data, real_data, name, gdp_cycle, cached_data, correlation_lags, autocorr_lags)
    _, cycle = cached_data

    # Align cycles by length
    min_length = min(length(cycle), length(gdp_cycle))
    max_length = max(length(cycle), length(gdp_cycle))
    gdp_cycle_adj = gdp_cycle[(1 + max_length - min_length):end]

    crosscorr_data[name] = crosscor(gdp_cycle_adj, cycle, correlation_lags)
    autocorr_data[name] = autocor(cycle, 0:autocorr_lags)
    stderr_data[name] = std(real_data[name])
end

"""
    process_real_3d_variable!(crosscorr_data, autocorr_data, stderr_data, real_data, name, gdp_cycle, cached_data, correlation_lags, autocorr_lags)

Process multi-sector real variable.

# Arguments
- `crosscorr_data`: Dictionary to store cross-correlation results
- `autocorr_data`: Dictionary to store autocorrelation results
- `stderr_data`: Dictionary to store standard error results
- `real_data`: Real economic data
- `name`: Variable name
- `gdp_cycle`: GDP cycle data for cross-correlation reference
- `cached_data`: Cached HP filter results for this variable
- `correlation_lags`: Number of lags for cross-correlation
- `autocorr_lags`: Number of lags for autocorrelation
"""
function process_real_3d_variable!(crosscorr_data, autocorr_data, stderr_data, real_data, name, gdp_cycle, cached_data, correlation_lags, autocorr_lags)
    n_sectors = length(cached_data)
    crosscorr_data[name] = zeros(2 * correlation_lags + 1, n_sectors)
    autocorr_data[name] = zeros(autocorr_lags + 1, n_sectors)
    stderr_data[name] = zeros(n_sectors)

    for (sector, sector_data) in cached_data
        _, cycle = sector_data
        # Align cycles by length
        min_length = min(length(cycle), length(gdp_cycle))
        max_length = max(length(cycle), length(gdp_cycle))
        gdp_cycle_adj = gdp_cycle[(1 + max_length - min_length):end]

        crosscorr_data[name][:, sector] = crosscor(gdp_cycle_adj, cycle, correlation_lags)
        autocorr_data[name][:, sector] = autocor(cycle, 0:autocorr_lags)
        stderr_data[name][sector] = std(real_data[name][:, sector])
    end
end

"""
    process_real_data_correlations(real_data, hp_cache, variable_names, correlation_lags, autocorr_lags)

Process real data correlations using cached HP filters.

# Arguments
- `real_data`: Real economic data dictionary
- `hp_cache`: HP filter cache dictionary
- `variable_names`: Collection of variable names to process
- `correlation_lags`: Number of lags for cross-correlation
- `autocorr_lags`: Number of lags for autocorrelation

# Returns
- `(crosscorr_data, autocorr_data, stderr_data)`: Processed correlation data
"""
function process_real_data_correlations(real_data, hp_cache, variable_names, correlation_lags, autocorr_lags)
    @info "Processing real data correlations"

    crosscorr_data = Dict{String, Any}()
    autocorr_data = Dict{String, Any}()
    stderr_data = Dict{String, Any}()

    for name in variable_names
        if !haskey(hp_cache, name)
            continue
        end

        # Determine GDP reference
        gdp_ref = determine_gdp_reference(hp_cache)
        if gdp_ref === nothing
            @warn "No GDP reference found for correlations"
            continue
        end

        if isa(hp_cache[gdp_ref], Tuple)
            _, gdp_cycle = hp_cache[gdp_ref]
        else
            # Multi-dimensional case - use first sector
            _, gdp_cycle = first(values(hp_cache[gdp_ref]))
        end

        if isa(hp_cache[name], Tuple)
            # Single dimension case
            process_real_1d_variable!(crosscorr_data, autocorr_data, stderr_data,
                                     real_data, name, gdp_cycle, hp_cache[name], correlation_lags, autocorr_lags)
        else
            # Multi-sector case
            process_real_3d_variable!(crosscorr_data, autocorr_data, stderr_data,
                                     real_data, name, gdp_cycle, hp_cache[name], correlation_lags, autocorr_lags)
        end
    end

    return crosscorr_data, autocorr_data, stderr_data
end

# =============================================================================
# SIMULATION PROCESSING FUNCTIONS
# =============================================================================

"""
    initialize_results(first_model, variable_names, gdp_size, n_quarters, correlation_lags, default_horizon, max_sectors)

Initialize result dictionaries based on model structure.

# Arguments
- `first_model`: First model data to determine structure
- `variable_names`: Collection of variable names
- `gdp_size`: Size of GDP data (time, simulations)
- `n_quarters`: Number of quarters
- `correlation_lags`: Number of lags for cross-correlation
- `default_horizon`: Default horizon for autocorrelation
- `max_sectors`: Maximum number of sectors

# Returns
- `(cyclesvar, crosscorr, autocorr, cycles_data)`: Initialized result dictionaries
"""
function initialize_results(first_model, variable_names, gdp_size, n_files, correlation_lags, default_horizon, max_sectors)
    cyclesvar = Dict{String, Any}()
    crosscorr = Dict{String, Any}()
    autocorr = Dict{String, Any}()

    # Calculate total number of simulations across all files
    n_total = n_files * gdp_size[2]

    @info "Initializing arrays for $n_files files × $(gdp_size[2]) simulations = $n_total total simulations"

    for name in variable_names
        if haskey(first_model, name)
            if ndims(first_model[name]) == 2
                cyclesvar[name] = zeros(n_total)
                crosscorr[name] = zeros(2 * correlation_lags + 1, n_total)
                autocorr[name] = zeros(default_horizon + 1, n_total)
            else
                cyclesvar[name] = zeros(n_total, max_sectors)
                crosscorr[name] = zeros(2 * correlation_lags + 1, n_total, max_sectors)
                autocorr[name] = zeros(default_horizon + 1, n_total, max_sectors)
            end
        end
    end

    cycles_data = nothing
    return cyclesvar, crosscorr, autocorr, cycles_data
end

"""
    store_cycle_data!(cycles_data_ref::Ref{Union{Nothing, Dict{String, Any}}}, name, trend, cycle, n, gdp_size, cycle_variables)

Store cycle data for plotting.

# Arguments
- `cycles_data_ref`: Reference to cycles data dictionary
- `name`: Variable name
- `trend`: Trend component
- `cycle`: Cycle component
- `n`: Simulation index
- `gdp_size`: Size of GDP data
- `cycle_variables`: List of cycle variables to track
"""
function store_cycle_data!(cycles_data_ref::Ref{Union{Nothing, Dict{String, Any}}}, name, trend, cycle, n, gdp_size, cycle_variables)
    if cycles_data_ref[] === nothing
        cycles_data_ref[] = Dict(
            "trends" => Dict{String, Matrix{Float64}}(),
            "cycles" => Dict{String, Matrix{Float64}}(),
            "cyclesnames" => cycle_variables
        )

        for cycle_name in cycle_variables
            cycles_data_ref[]["trends"][cycle_name] = zeros(gdp_size)
            cycles_data_ref[]["cycles"][cycle_name] = zeros(gdp_size)
        end
    end

    cycles_data = cycles_data_ref[]
    if haskey(cycles_data["trends"], name)
        cycles_data["trends"][name][:, n] = trend
        cycles_data["cycles"][name][:, n] = cycle
    end
end

"""
    process_2d_variable!(cyclesvar, crosscorr, autocorr, cycles_data_ref::Ref{Union{Nothing, Dict{String, Any}}}, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, cycle_variables)

Process 2D variable (time x simulations).

# Arguments
- `cyclesvar`: Dictionary to store cycle variance results
- `crosscorr`: Dictionary to store cross-correlation results
- `autocorr`: Dictionary to store autocorrelation results
- `cycles_data_ref`: Reference to cycles data for plotting
- `model`: Model data dictionary
- `name`: Variable name
- `file_idx`: File index
- `gdp_size`: Size of GDP data
- `n_seeds`: Number of simulation seeds
- `correlation_lags`: Number of lags for cross-correlation
- `autocorr_lags`: Number of lags for autocorrelation
- `cycle_variables`: List of cycle variables to track
"""
function process_2d_variable!(cyclesvar, crosscorr, autocorr, cycles_data_ref::Ref{Union{Nothing, Dict{String, Any}}}, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, cycle_variables)
    for n in 1:gdp_size[2]
        # Compute HP filters
        _, gdp_cycle = hpfilter(model["real_gdp_quarterly"][:, n])
        _, var_cycle = hpfilter(model[name][:, n])

        # Calculate index
        idx = (file_idx - 1) * n_seeds + n

        # Store results
        cyclesvar[name][idx] = std(model[name][:, n])
        crosscorr[name][:, idx] = crosscor(gdp_cycle, var_cycle, correlation_lags)
        autocorr[name][:, idx] = autocor(var_cycle, 0:autocorr_lags)

        # Store cycle data for first file
        if file_idx == 1 && name in cycle_variables
            var_trend, _ = hpfilter(model[name][:, n])
            store_cycle_data!(cycles_data_ref, name, var_trend, var_cycle, n, gdp_size, cycle_variables)
        end
    end
end

"""
    process_3d_variable!(cyclesvar, crosscorr, autocorr, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, max_sectors)

Process 3D variable (time x simulations x sectors).

# Arguments
- `cyclesvar`: Dictionary to store cycle variance results
- `crosscorr`: Dictionary to store cross-correlation results
- `autocorr`: Dictionary to store autocorrelation results
- `model`: Model data dictionary
- `name`: Variable name
- `file_idx`: File index
- `gdp_size`: Size of GDP data
- `n_seeds`: Number of simulation seeds
- `correlation_lags`: Number of lags for cross-correlation
- `autocorr_lags`: Number of lags for autocorrelation
- `max_sectors`: Maximum number of sectors to process
"""
function process_3d_variable!(cyclesvar, crosscorr, autocorr, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, max_sectors)
    for n in 1:gdp_size[2]
        # GDP reference cycle (computed once per simulation)
        _, gdp_cycle = hpfilter(model["real_gdp_quarterly"][:, n])
        idx = (file_idx - 1) * n_seeds + n

        for sector in 1:max_sectors
            # Variable cycle
            _, var_cycle = hpfilter(model[name][:, n, sector])

            # Store results
            cyclesvar[name][idx, sector] = std(model[name][:, n, sector])
            crosscorr[name][:, idx, sector] = crosscor(gdp_cycle, var_cycle, correlation_lags)
            autocorr[name][:, idx, sector] = autocor(var_cycle, 0:autocorr_lags)
        end
    end
end

"""
    process_variable!(cyclesvar, crosscorr, autocorr, cycles_data_ref::Ref{Union{Nothing, Dict{String, Any}}}, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, cycle_variables, max_sectors)

Process a single variable for correlations and statistics.

# Arguments
- `cyclesvar`: Dictionary to store cycle variance results
- `crosscorr`: Dictionary to store cross-correlation results
- `autocorr`: Dictionary to store autocorrelation results
- `cycles_data_ref`: Reference to cycles data for plotting
- `model`: Model data dictionary
- `name`: Variable name
- `file_idx`: File index
- `gdp_size`: Size of GDP data
- `n_seeds`: Number of simulation seeds
- `correlation_lags`: Number of lags for cross-correlation
- `autocorr_lags`: Number of lags for autocorrelation
- `cycle_variables`: List of cycle variables to track
- `max_sectors`: Maximum number of sectors to process
"""
function process_variable!(cyclesvar, crosscorr, autocorr, cycles_data_ref::Ref{Union{Nothing, Dict{String, Any}}}, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, cycle_variables, max_sectors)
    if !haskey(model, name) || size(model[name], 1) != size(model["real_gdp_quarterly"], 1)
        return
    end

    if ndims(model[name]) == 2
        process_2d_variable!(cyclesvar, crosscorr, autocorr, cycles_data_ref, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, cycle_variables)
    else
        process_3d_variable!(cyclesvar, crosscorr, autocorr, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, max_sectors)
    end
end

"""
    process_all_simulation_data(model_folder, variable_names, gdp_size, n_seeds, n_quarters, correlation_lags, default_horizon, autocorr_lags, cycle_variables, max_sectors)

Single-pass processing of all simulation files.

# Arguments
- `model_folder`: Path to model prediction files
- `variable_names`: Collection of variable names to process
- `gdp_size`: Size of GDP data (time, simulations)
- `n_seeds`: Number of simulation seeds
- `n_quarters`: Number of quarters
- `correlation_lags`: Number of lags for cross-correlation
- `default_horizon`: Default horizon for autocorrelation
- `autocorr_lags`: Number of lags for autocorrelation
- `cycle_variables`: List of cycle variables to track
- `max_sectors`: Maximum number of sectors to process

# Returns
- `(cyclesvar, crosscorr, autocorr, cycles_data)`: Processed correlation data
"""
function process_all_simulation_data(model_folder, variable_names, gdp_size, n_seeds, n_quarters, correlation_lags, default_horizon, autocorr_lags, cycle_variables, max_sectors)
    @info "Processing simulation data with single pass"

    # Get prediction files and extract valid year-quarter combinations
    prediction_files = filter(f -> endswith(f, ".jld2"), readdir(model_folder))

    # Extract year and quarter from filenames like 2010Q1.jld2 (same pattern as save_all_simulations)
    function extract_yq(files)
        Set([match(r"^(\d{4}Q\d)\.jld2$", f)[1] for f in files if occursin(r"Q", f) && match(r"^(\d{4}Q\d)\.jld2$", f) !== nothing])
    end

    prediction_yq = extract_yq(prediction_files)
    valid_filenames = sort(collect(prediction_yq))

    if isempty(valid_filenames)
        error("No valid prediction files found in $model_folder")
    end

    @info "Found $(length(valid_filenames)) valid prediction files with year-quarter pattern"

    # Use first valid file for initialization
    first_file = first(valid_filenames) * ".jld2"
    first_model = load(joinpath(model_folder, first_file))["predictions_dict"]
    cyclesvar, crosscorr, autocorr, cycles_data = initialize_results(first_model, variable_names, gdp_size, length(valid_filenames), correlation_lags, default_horizon, max_sectors)

    # Use reference for cycles_data to allow modification
    cycles_data_ref = Ref{Union{Nothing, Dict{String, Any}}}(cycles_data)

    # Process each valid file exactly once
    for (file_idx, yq_filename) in enumerate(valid_filenames)
        prediction_file = yq_filename * ".jld2"
        @info "Processing file $file_idx/$(length(valid_filenames)): $prediction_file"

        model_path = joinpath(model_folder, prediction_file)
        model = load(model_path)["predictions_dict"]

        # Process all variables for this file
        for name in variable_names
            process_variable!(cyclesvar, crosscorr, autocorr, cycles_data_ref, model, name, file_idx, gdp_size, n_seeds, correlation_lags, autocorr_lags, cycle_variables, max_sectors)
        end
    end

    @info "Simulation data processing completed"
    return cyclesvar, crosscorr, autocorr, cycles_data_ref[]
end

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

"""
    calculate_statistics(crosscorr, autocorr, cyclesvar, variable_names, max_sectors, default_horizon)

Calculate statistics from correlation results.

# Arguments
- `crosscorr`: Cross-correlation results from simulations
- `autocorr`: Autocorrelation results from simulations
- `cyclesvar`: Cycle variance results from simulations
- `variable_names`: Collection of variable names
- `max_sectors`: Maximum number of sectors
- `default_horizon`: Default horizon for autocorrelation

# Returns
- `(mean_xcorr, std_xcorr, mean_autocorr, std_autocorr, mean_cyclesvar)`: Statistical summaries
"""
function calculate_statistics(crosscorr, autocorr, cyclesvar, variable_names, max_sectors, default_horizon)
    @info "Calculating statistics from correlation results"

    mean_xcorr = Dict{String, Any}()
    std_xcorr = Dict{String, Any}()
    mean_autocorr = Dict{String, Any}()
    std_autocorr = Dict{String, Any}()
    mean_cyclesvar = Dict{String, Any}()

    for name in variable_names
        if !haskey(crosscorr, name)
            continue
        end

        if ndims(crosscorr[name]) == 2
            mean_xcorr[name] = mean(crosscorr[name], dims=2)
            std_xcorr[name] = std(crosscorr[name], dims=2)
            mean_autocorr[name] = mean(autocorr[name], dims=2)
            std_autocorr[name] = std(autocorr[name], dims=2)
            mean_cyclesvar[name] = mean(cyclesvar[name])
        else
            # 3D case - aggregate over sectors
            mean_xcorr[name] = zeros(size(crosscorr[name], 1), max_sectors)
            std_xcorr[name] = zeros(size(crosscorr[name], 1), max_sectors)
            mean_autocorr[name] = zeros(default_horizon + 1, max_sectors)
            std_autocorr[name] = zeros(default_horizon + 1, max_sectors)
            mean_cyclesvar[name] = zeros(1, max_sectors)

            for sector in 1:max_sectors
                mean_xcorr[name][:, sector] = mean(crosscorr[name][:, :, sector], dims=2)
                std_xcorr[name][:, sector] = std(crosscorr[name][:, :, sector], dims=2)
                mean_autocorr[name][:, sector] = mean(autocorr[name][:, :, sector], dims=2)
                std_autocorr[name][:, sector] = std(autocorr[name][:, :, sector], dims=2)
                mean_cyclesvar[name][sector] = mean(cyclesvar[name][:, sector])
            end
        end
    end

    return mean_xcorr, std_xcorr, mean_autocorr, std_autocorr, mean_cyclesvar
end

""""
    load_calibration_data(country_code)
Load calibration data for a given country from migrated data.
# Arguments
- `country_code`: Country identifier string
# Returns
- Calibration data object
"""

function load_calibration_data(country_code)
    @info "Loading calibration data for $country_code from migrated data"

    country_dir = "data/$(lowercase(country_code))"
    if !isdir(country_dir)
        error("No migrated data found for $country_code at $country_dir")
    end

    # Try to load calibration_object.jld2 first, then calibration_data.jld2
    calibration_object_file = joinpath(country_dir, "calibration_object.jld2")
    calibration_data_file = joinpath(country_dir, "calibration_data.jld2")

    if isfile(calibration_object_file)
        return load(calibration_object_file)["calibration_object"]
    elseif isfile(calibration_data_file)
        return load(calibration_data_file)["calibration_data"]
    else
        error("No calibration data found for $country_code in $country_dir")
    end
end

"""
    discover_countries_with_calibration()

Discover all countries that have calibration_object.jld2 files.

# Returns
- Vector of country names (directory names in data folder)
"""
function discover_countries_with_calibration()
    data_dir = "data"
    if !isdir(data_dir)
        error("Data directory not found: $data_dir")
    end

    countries = String[]

    for entry in readdir(data_dir)
        country_dir = joinpath(data_dir, entry)
        if isdir(country_dir)
            calibration_file = joinpath(country_dir, "calibration_object.jld2")
            if isfile(calibration_file)
                push!(countries, entry)
            end
        end
    end

    @info "Found $(length(countries)) countries with calibration data"
    @info "Countries: $(join(sort(countries), ", "))"

    return sort(countries)
end

"""
    discover_countries_with_predictions()

Discover all countries that have abm_predictions folder with prediction files.

# Returns
- Vector of country names (directory names in data folder)
"""
function discover_countries_with_predictions()
    data_dir = "data"
    if !isdir(data_dir)
        error("Data directory not found: $data_dir")
    end

    countries = String[]

    for entry in readdir(data_dir)
        country_dir = joinpath(data_dir, entry)
        if isdir(country_dir)
            predictions_dir = joinpath(country_dir, "abm_predictions")
            if isdir(predictions_dir)
                # Check if there are any .jld2 files in the predictions directory
                prediction_files = filter(f -> endswith(f, ".jld2"), readdir(predictions_dir))
                if !isempty(prediction_files)
                    push!(countries, entry)
                end
            end
        end
    end

    @info "Found $(length(countries)) countries with prediction data"
    @info "Countries: $(join(sort(countries), ", "))"

    return sort(countries)
end