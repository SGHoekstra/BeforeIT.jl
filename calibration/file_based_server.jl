using JSON
using Dates
using FileWatching
using BeforeIT

# Configuration
const INPUT_DIR = get(ENV, "BEFOREIT_INPUT_DIR", "./shared/input")
const OUTPUT_DIR = get(ENV, "BEFOREIT_OUTPUT_DIR", "./shared/output")
const POLLING_INTERVAL = parse(Float64, get(ENV, "BEFOREIT_POLLING_INTERVAL", "0.1"))  # seconds

# Global variables to store models and dates
global_models = nothing
start_date = nothing
end_date = nothing

# Ensure directories exist
mkpath(INPUT_DIR)
mkpath(OUTPUT_DIR)

function process_file(input_file)
    @info "Processing file: $input_file"
    
    # Create a unique ID based on the filename
    file_id = basename(input_file)
    id_base = splitext(file_id)[1]
    
    # Read and parse the input file
    try
        open(input_file, "r") do f
            input_data = JSON.parse(f)
            command = input_data["command"]
            
            result = Dict{String, Any}()
            
            if command == "initialize"
                # Parse dates
                global start_date = DateTime(input_data["start_date"])
                global end_date = DateTime(input_data["end_date"])
                model_type = get(input_data, "model_type", "base")
                empirical_distribution = get(input_data, "empirical_distribution", false)
                conditional_forecasts = get(input_data, "conditional_forecasts", false)
                
                # Initialize models
                @info "Initializing models from $start_date to $end_date"
                global global_models = BeforeIT.get_models(
                    start_date, 
                    end_date;
                    model_type=model_type,
                    empirical_distribution=empirical_distribution,
                    conditional_forecasts=conditional_forecasts
                )
                
                result["status"] = "success"
                
            elseif command == "get_data"
                sd = DateTime(input_data["start_date"])
                ed = DateTime(input_data["end_date"])
                repetitions = get(input_data, "repetitions", 1)
                
                @info "Loading data from $sd to $ed"
                real_data = BeforeIT.load_timeseries_for_calibrator(sd, ed; repetitions = repetitions)
                result["data"] = real_data
                
            elseif command == "run_simulation"
                if isnothing(global_models)
                    result["error"] = "Must initialize models first"
                    @warn "Models not initialized"
                else
                    # Get parameters and create dictionary
                    params = input_data["params"]
                    version_c = get(input_data, "version_c", 1)

                    param_dict = Dict()
                    if version_c == 1
                        param_dict = Dict("γ" => params[1])
                    elseif version_c == 2
                        param_dict = Dict("α" => params[1], "β" => params[2])
                    elseif version_c == 3
                        param_dict = Dict("γ" => params[1], "δ" => params[2])
                    elseif version_c == 4
                        param_dict = Dict("α" => params[1], "β" => params[2], "δ" => params[3])
                    elseif version_c == 5
                        param_dict = Dict("α" => params[1], "β" => params[2], "θ" => params[3], "ρ" => params[4])
                    elseif version_c == 6
                        param_dict = Dict("γ" => params[1], "α" => params[2], "β" => params[3])
                    elseif version_c == 7
                        param_dict = Dict("γ" => params[1], "α" => params[2], "β" => params[3], "δ" => params[4])
                    elseif version_c == 8
                        param_dict = Dict("γ" => params[1], "α" => params[2], "β" => params[3], "δ" => params[4], "θ" => params[5], "ρ" => params[6])
                    end
                    
                    # Get simulation parameters
                    sim_start_date = DateTime(input_data["start_date"])
                    sim_end_date = DateTime(input_data["end_date"])
                    
                    model_type = get(input_data, "model_type", "extended_heuristic")
                    num_simulations = get(input_data, "num_simulations", 1)
                    abmx = get(input_data, "abmx", false)
                    conditional_forecasts = get(input_data, "conditional_forecasts", false)
                    multi_threading = get(input_data, "multi_threading", true)
                    
                    @info "Running simulation with $(length(params)) parameters"
                    # Run simulation
                    sim_result = BeforeIT.run_abm_simulations_with_parameters(
                        global_models, 
                        param_dict,
                        sim_start_date,
                        sim_end_date;
                        num_simulations=num_simulations,
                        model_type=model_type,
                        abmx=abmx,
                        conditional_forecasts=conditional_forecasts,
                        version_c=version_c,
                        multi_threading=multi_threading
                    )
                    
                    result["result"] = sim_result
                end
            else
                result["error"] = "Unknown command: $command"
                @warn "Unknown command: $command"
            end
            
            # Write the result to the output file
            output_filename = joinpath(OUTPUT_DIR, "$(id_base)_result.json")
            open(output_filename, "w") do out_f
                JSON.print(out_f, result)
            end
            @info "Wrote result to $output_filename"
            
            # Create a completion flag file to signal the client
            touch(joinpath(OUTPUT_DIR, "$(id_base)_complete"))
        end
        
        # Remove the input file after processing
        rm(input_file)
        
    catch e
        # Log the error
        @error "Error processing file $input_file" exception=(e, catch_backtrace())
        
        # Write error to output file
        output_filename = joinpath(OUTPUT_DIR, "$(id_base)_result.json")
        open(output_filename, "w") do out_f
            JSON.print(out_f, Dict("error" => "Server error: $(string(e))"))
        end
        
        # Create a completion flag file to signal the client
        touch(joinpath(OUTPUT_DIR, "$(id_base)_complete"))
        
        # Remove the input file after processing
        try
            rm(input_file)
        catch
            @warn "Could not remove input file $input_file"
        end
    end
end

function watch_directory()
    @info "Starting file-based server watching directory: $INPUT_DIR"
    @info "Results will be written to: $OUTPUT_DIR"
    
    # Process any existing files
    existing_files = filter(f -> endswith(f, ".json"), readdir(INPUT_DIR))
    if !isempty(existing_files)
        @info "Found $(length(existing_files)) existing files to process"
        for file in existing_files
            process_file(joinpath(INPUT_DIR, file))
        end
    end
    
    # Watch for new files
    while true
        # Check for any JSON files in the directory
        for file in readdir(INPUT_DIR)
            if endswith(file, ".json")
                full_path = joinpath(INPUT_DIR, file)
                if isfile(full_path)  # Make sure it's still there
                    process_file(full_path)
                end
            end
        end
        
        # Sleep to avoid busy waiting
        sleep(POLLING_INTERVAL)
    end
end

# Start watching for files
try
    watch_directory()
catch e
    @error "Server crashed" exception=(e, catch_backtrace())
    rethrow(e)
end