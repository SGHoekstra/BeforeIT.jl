#!/bin/bash
# run_calibration_http.sh - Runs calibration using BeforeIT HTTP servers in Docker

# Get version to calibrate
VERSION=${1:-2}  # Default to version 2 if not specified
NUM_SERVERS=${2:-4}  # Default to 4 servers if not specified
NUM_CPUS=${3:-2}  # Default to 2 CPUs per server
NUM_CALIBRATION=${4:-10}  # Default to 10 iterations if not specified

# Create logs directory
mkdir -p logs

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