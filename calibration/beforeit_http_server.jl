using JSON
using Dates
using HTTP
using BeforeIT

# Configuration
const PORT = parse(Int, get(ENV, "BEFOREIT_PORT", "8080"))
const HOST = get(ENV, "BEFOREIT_HOST", "0.0.0.0")

# Global variables to store models and dates
global_models = nothing
start_date = nothing
end_date = nothing

# Initialize request counter for logging
request_counter = 0

function log_request(method, endpoint, status)
    global request_counter
    request_counter += 1
    id = request_counter
    @info "[$id] $method $endpoint - $status"
end

function handle_initialize(req)
    try
        # Parse the request body
        body = JSON.parse(String(HTTP.payload(req)))
        
        # Parse dates
        global start_date = DateTime(body["start_date"])
        global end_date = DateTime(body["end_date"])
        
        # Get optional parameters
        model_type = get(body, "model_type", "base")
        empirical_distribution = get(body, "empirical_distribution", false)
        conditional_forecasts = get(body, "conditional_forecasts", false)

        # Initialize models
        @info "Initializing models from $start_date to $end_date"
        global global_models = BeforeIT.get_models(
            start_date, 
            end_date
        )
        
        # Return success response
        log_request("POST", "/initialize", 200)
        return HTTP.Response(200, JSON.json(Dict(
            "status" => "success",
            "cmd_id" => get(body, "cmd_id", ""),
            "message" => "Models initialized successfully"
        )))
    catch e
        # Log the error
        @error "Error in initialize endpoint" exception=(e, catch_backtrace())
        
        # Return error response
        log_request("POST", "/initialize", 500)
        return HTTP.Response(500, JSON.json(Dict(
            "error" => string(e),
            "cmd_id" => get(try JSON.parse(String(HTTP.payload(req))); catch; Dict(); end, "cmd_id", "")
        )))
    end
end

function handle_get_data(req)
    try
        # Parse the request body
        body = JSON.parse(String(HTTP.payload(req)))
        
        # Parse parameters
        sd = DateTime(body["start_date"])
        ed = DateTime(body["end_date"])
        repetitions = get(body, "repetitions", 1)
        
        @info "Loading data from $sd to $ed with $repetitions repetitions"
        real_data = BeforeIT.load_timeseries_for_calibrator(sd, ed; repetitions = repetitions)
        
        # Return data
        log_request("POST", "/get_data", 200)
        return HTTP.Response(200, JSON.json(Dict(
            "data" => real_data,
            "cmd_id" => get(body, "cmd_id", "")
        )))
    catch e
        # Log the error
        @error "Error in get_data endpoint" exception=(e, catch_backtrace())
        
        # Return error response
        log_request("POST", "/get_data", 500)
        return HTTP.Response(500, JSON.json(Dict(
            "error" => string(e),
            "cmd_id" => get(try JSON.parse(String(HTTP.payload(req))); catch; Dict(); end, "cmd_id", "")
        )))
    end
end

function handle_run_simulation(req)
    try
        # Check if models are initialized
        if isnothing(global_models)
            log_request("POST", "/run_simulation", 400)
            return HTTP.Response(400, JSON.json(Dict(
                "error" => "Must initialize models first",
                "cmd_id" => get(try JSON.parse(String(HTTP.payload(req))); catch; Dict(); end, "cmd_id", "")
            )))
        end
        
        # Parse the request body
        body = JSON.parse(String(HTTP.payload(req)))
        
        # Get parameters and create dictionary
        params = body["params"]

        # Get simulation parameters
        sim_start_date = DateTime(body["start_date"])
        sim_end_date = DateTime(body["end_date"])
        
        num_simulations = get(body, "num_simulations", 1)
        multi_threading = get(body, "multi_threading", true)
        abmx = get(body, "abmx", false)
        
        @info "Running simulation with $(length(params)) parameters, $(num_simulations) simulations, abmx: $abmx"
        
        # Run simulation
        sim_result = BeforeIT.run_abm_simulations_with_parameters(
            global_models, 
            params,
            sim_start_date,
            sim_end_date;
            num_simulations=num_simulations,
            multi_threading=multi_threading,
            abmx = abmx,
        )
        
        # Return result
        log_request("POST", "/run_simulation", 200)
        return HTTP.Response(200, JSON.json(Dict(
            "result" => sim_result,
            "cmd_id" => get(body, "cmd_id", "")
        )))
    catch e
        # Log the error
        @error "Error in run_simulation endpoint" exception=(e, catch_backtrace())
        
        # Return error response
        log_request("POST", "/run_simulation", 500)
        return HTTP.Response(500, JSON.json(Dict(
            "error" => string(e),
            "cmd_id" => get(try JSON.parse(String(HTTP.payload(req))); catch; Dict(); end, "cmd_id", "")
        )))
    end
end

function handle_health_check(req)
    # Return a simple health check response
    log_request("GET", "/health", 200)
    return HTTP.Response(200, JSON.json(Dict(
        "status" => "healthy",
        "models_initialized" => !isnothing(global_models)
    )))
end

function start_server()
    router = HTTP.Router()
    
    # Define routes
    HTTP.register!(router, "POST", "/initialize", handle_initialize)
    HTTP.register!(router, "POST", "/get_data", handle_get_data)
    HTTP.register!(router, "POST", "/run_simulation", handle_run_simulation)
    HTTP.register!(router, "GET", "/health", handle_health_check)
    
    # 404 handler
    function handle_404(req)
        log_request(req.method, req.target, 404)
        return HTTP.Response(404, "Not found")
    end
    
    # Start the server
    @info "Starting HTTP server on $HOST:$PORT"
    server = HTTP.serve!(router, HOST, PORT; readtimeout=300)
    
    return server
end

# Start the HTTP server
try
    @info "BeforeIT HTTP Server starting..."
    server = start_server()
    
    # Keep the server running until interrupted
    wait(server)
catch e
    @error "Server crashed" exception=(e, catch_backtrace())
    rethrow(e)
end
