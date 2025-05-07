#!/bin/bash
# run_calibration_http.sh - Runs calibration using BeforeIT HTTP servers in Docker

# Get version to calibrate
VERSION=${1:-2}  # Default to version 2 if not specified
NUM_SERVERS=${2:-4}  # Default to 4 servers if not specified
NUM_CPUS=${3:-2}  # Default to 2 CPUs per server
NUM_CALIBRATION=${4:-10}  # Default to 10 iterations if not specified

# Create logs directory
mkdir -p logs

echo "Setting up environment..."
# Check if HTTP server file exists
if [ ! -f "calibration/beforeit_http_server.jl" ]; then
  echo "Error: calibration/beforeit_http_server.jl not found!"
  exit 1
fi

# Check if BeforeIT.jl exists (main module file)
if [ ! -f "src/BeforeIT.jl" ]; then
  echo "Error: src/BeforeIT.jl not found! Please run this script from the root directory of the BeforeIT project."
  exit 1
fi

# Update Dockerfile.julia with our enhanced version
cat > Dockerfile.julia << EOL
FROM julia:1.10

WORKDIR /app

# Copy the entire project first to maintain structure
COPY . /app/

# Install required HTTP packages explicitly
RUN julia -e 'using Pkg; Pkg.add(["HTTP", "JSON", "Dates"])'

# Activate and instantiate the project environment
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'

# Create directory for logs
RUN mkdir -p /app/logs

# Set environment variables with defaults
ENV BEFOREIT_PORT=8080
ENV BEFOREIT_HOST="0.0.0.0"

# Expose the default port
EXPOSE 8080

# Add a healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=15s --retries=3 \\
  CMD curl -f http://localhost:8080/health || exit 1

# Start the HTTP server
CMD ["julia", "--project=.", "calibration/beforeit_http_server.jl"]
EOL

# Update the HTTP server script with our fixed version
cat > calibration/beforeit_http_server.jl << EOL
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
    @info "[\$id] \$method \$endpoint - \$status"
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
        @info "Initializing models from \$start_date to \$end_date"
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
        
        @info "Loading data from \$sd to \$ed with \$repetitions repetitions"
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
        param_dict = Dict("theta_UNION" => params[1])

        # Get simulation parameters
        sim_start_date = DateTime(body["start_date"])
        sim_end_date = DateTime(body["end_date"])
        
        num_simulations = get(body, "num_simulations", 1)
        multi_threading = get(body, "multi_threading", true)
        
        @info "Running simulation with \$(length(params)) parameters, \$(num_simulations) simulations"
        
        # Run simulation
        sim_result = BeforeIT.run_abm_simulations_with_parameters(
            global_models, 
            param_dict,
            sim_start_date,
            sim_end_date;
            num_simulations=num_simulations,
            multi_threading=multi_threading
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
    @info "Starting HTTP server on \$HOST:\$PORT"
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
EOL

# Build the Docker image with no-cache to ensure fresh installation
echo "Building Docker image for BeforeIT HTTP server..."
docker build -t beforeit-http-server:latest -f Dockerfile.julia .

# Check if build was successful
if [ $? -ne 0 ]; then
  echo "Docker build failed! Check the logs above for errors."
  exit 1
fi

# Start the containers
echo "Starting HTTP servers..."
for i in $(seq 1 $NUM_SERVERS); do
  CONTAINER_NAME="beforeit-http-server-$i"
  PORT=$((8079 + $i))
  
  # Stop and remove container if it already exists
  docker stop $CONTAINER_NAME > /dev/null 2>&1
  docker rm $CONTAINER_NAME > /dev/null 2>&1
  
  echo "Starting $CONTAINER_NAME on port $PORT..."
  docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8080 \
    -e JULIA_NUM_THREADS=$NUM_CPUS \
    -e BEFOREIT_SERVER_ID=$i \
    -e BEFOREIT_PORT=8080 \
    -e BEFOREIT_HOST=0.0.0.0 \
    -v "$(pwd)/logs:/app/logs" \
    beforeit-http-server:latest
  
  # Check if container started successfully
  if [ $? -ne 0 ]; then
    echo "Failed to start container $CONTAINER_NAME! Check Docker logs."
    docker logs $CONTAINER_NAME
    exit 1
  fi
done

# Wait for servers to initialize
echo "Waiting for servers to become healthy..."
for i in $(seq 1 $NUM_SERVERS); do
  CONTAINER_NAME="beforeit-http-server-$i"
  PORT=$((8079 + $i))
  
  echo "Checking health of $CONTAINER_NAME on port $PORT..."
  TIMEOUT=120  # Extended timeout (2 minutes)
  for (( t=1; t<=$TIMEOUT; t++ )); do
    # Check if container is still running
    if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
      echo "Container $CONTAINER_NAME stopped unexpectedly!"
      docker logs $CONTAINER_NAME
      exit 1
    fi
    
    # Try to connect to health endpoint
    if curl -s "http://localhost:$PORT/health" > /dev/null; then
      echo "✓ Server $i is healthy"
      break
    fi
    
    # Show container logs if taking too long
    if [ $t -eq 30 ]; then
      echo "Server $i is taking a long time to start up. Here are the logs:"
      docker logs $CONTAINER_NAME
    fi
    
    if [ $t -eq $TIMEOUT ]; then
      echo "✗ Server $i did not become healthy within $TIMEOUT seconds"
      echo "Container logs:"
      docker logs $CONTAINER_NAME
      exit 1
    fi
    
    echo -n "."
    sleep 1
  done
done

# Test if all servers can handle a basic initialization request
echo "Testing initialization for all servers..."
for i in $(seq 1 $NUM_SERVERS); do
  PORT=$((8079 + $i))
  
  echo "Testing initialization for server on port $PORT..."
  TEST_RESULT=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"start_date": "2010-03-31", "end_date": "2013-12-31"}' \
    "http://localhost:$PORT/initialize")
  
  if [[ $TEST_RESULT == *"success"* ]]; then
    echo "✓ Server on port $PORT initialized successfully"
  else
    echo "✗ Server on port $PORT initialization failed"
    echo "  - Response: $TEST_RESULT"
    echo "  - Check server logs: docker logs beforeit-http-server-$i"
    exit 1
  fi
done

# List server URLs
echo "All servers started and verified. Server URLs:"
for i in $(seq 1 $NUM_SERVERS); do
  PORT=$((8079 + $i))
  echo "Server $i: http://localhost:$PORT"
done

# Create a configuration file for the Python client
echo "Creating configuration file for Python client..."
cat > calibration/beforeit_config.json << EOL
{
  "server_urls": [
$(for i in $(seq 1 $NUM_SERVERS); do
  PORT=$((8079 + $i))
  echo "    \"http://localhost:$PORT\""
  if [ $i -lt $NUM_SERVERS ]; then echo ","; fi
done)
  ],
  "version": $VERSION,
  "num_calibration": $NUM_CALIBRATION
}
EOL

echo "Configuration created in calibration/beforeit_config.json"

# Function to check logs
check_logs() {
  echo "=== Server Logs ==="
  for i in $(seq 1 $NUM_SERVERS); do
    CONTAINER_NAME="beforeit-http-server-$i"
    echo "=== $CONTAINER_NAME logs ==="
    docker logs $CONTAINER_NAME
    echo ""
  done
}

# Function to stop all servers
stop_servers() {
  echo "Stopping all BeforeIT HTTP servers..."
  for i in $(seq 1 $NUM_SERVERS); do
    CONTAINER_NAME="beforeit-http-server-$i"
    docker stop $CONTAINER_NAME
    echo "Stopped $CONTAINER_NAME"
  done
}

# Function to remove all servers
remove_servers() {
  echo "Removing all BeforeIT HTTP servers..."
  for i in $(seq 1 $NUM_SERVERS); do
    CONTAINER_NAME="beforeit-http-server-$i"
    docker rm -f $CONTAINER_NAME 2>/dev/null
    echo "Removed $CONTAINER_NAME"
  done
}

# Handle command line options
case "${5:-}" in
  logs)
    check_logs
    ;;
  stop)
    stop_servers
    ;;
  remove)
    stop_servers
    remove_servers
    ;;
  restart)
    stop_servers
    remove_servers
    # Start the script again without the restart parameter
    $0 $VERSION $NUM_SERVERS $NUM_CPUS $NUM_CALIBRATION
    ;;
esac

echo "Servers are ready for calibration."
echo "To run calibration with HTTP client, use: python calibration/run_http_client_calibration.py"
echo "To check logs: $0 $VERSION $NUM_SERVERS $NUM_CPUS $NUM_CALIBRATION logs"
echo "To stop servers: $0 $VERSION $NUM_SERVERS $NUM_CPUS $NUM_CALIBRATION stop"