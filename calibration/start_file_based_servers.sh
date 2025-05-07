#!/bin/bash
# run_calibration.sh - Runs calibration using existing BeforeIT images

# Get version to calibrate
VERSION=${1:-2}  # Default to version 2 if not specified
NUM_SERVERS=${2:-4}  # Default to 4 servers if not specified
NUM_CPUS=${3:-2}  # Default to 2 CPUs if not specified
NUM_CALIBRATION=${4:-10}  # Default to 10 iterations if not specified

# Base directory for all shared data
BASE_DIR="shared_data"
mkdir -p $BASE_DIR

echo "Creating shared directories..."
# Create necessary directories for each server
for i in $(seq 1 $NUM_SERVERS); do
  SERVER_DIR="$BASE_DIR/shared_data_$i"
  mkdir -p "$SERVER_DIR/input" "$SERVER_DIR/output"
  chmod -R 777 "$SERVER_DIR"  # Ensure permissions are set correctly
done

mkdir -p extended_heuristic

# Check if servers are running; start them if they're not
for i in $(seq 1 $NUM_SERVERS); do
  CONTAINER_NAME="beforeit-julia-server-$i"
  SERVER_DIR="$BASE_DIR/shared_data_$i"
  
  if [ -z "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Starting $CONTAINER_NAME..."
    docker run -d \
      --name $CONTAINER_NAME \
      -v "$(pwd)/$SERVER_DIR:/shared_data" \
      -e BEFOREIT_INPUT_DIR=/shared_data/input \
      -e BEFOREIT_OUTPUT_DIR=/shared_data/output \
      -e BEFOREIT_SERVER_ID=$i \
      -e JULIA_NUM_THREADS=$NUM_CPUS \
      beforeit-julia-server:latest
    
    # Verify the container is running and checking the right folders
    sleep 2
    echo "Verifying $CONTAINER_NAME configuration..."
    docker exec $CONTAINER_NAME env | grep BEFOREIT
    docker logs $CONTAINER_NAME | grep "watching directory\|Results will be written to"
  else
    echo "$CONTAINER_NAME is already running"
  fi
done

# Test if servers can access the correct folders
echo "Testing folder access for all servers..."
for i in $(seq 1 $NUM_SERVERS); do
  CONTAINER_NAME="beforeit-julia-server-$i"
  SERVER_DIR="$BASE_DIR/shared_data_$i"
  TEST_FILE="test_$(date +%s).json"
  
  echo "Creating test file for server $i: $TEST_FILE"
  cat > "$SERVER_DIR/input/$TEST_FILE" << EOL
{
  "command": "initialize",
  "start_date": "2025-01-01T00:00:00",
  "end_date": "2025-01-02T00:00:00"
}
EOL
  
  # Wait for the file to be processed
  echo "Waiting for server $i to process the test file..."
  TIMEOUT=10
  for (( t=1; t<=$TIMEOUT; t++ )); do
    if [ ! -f "$SERVER_DIR/input/$TEST_FILE" ]; then
      echo "✓ Server $i processed the test file"
      break
    fi
    if [ $t -eq $TIMEOUT ]; then
      echo "✗ Server $i did not process the test file within $TIMEOUT seconds"
      echo "  - Check server logs: docker logs $CONTAINER_NAME"
    fi
    sleep 1
  done
  
  # Check for result file
  RESULT_BASE="${TEST_FILE%.*}"
  if [ -f "$SERVER_DIR/output/${RESULT_BASE}_result.json" ] || [ -f "$SERVER_DIR/output/${RESULT_BASE}_complete" ]; then
    echo "✓ Server $i created output files correctly"
  else
    echo "✗ Server $i did not create expected output files"
    echo "  - Check server logs: docker logs $CONTAINER_NAME"
  fi
done

# Prepare volume mounts for calibration container
VOLUMES=""
for i in $(seq 1 $NUM_SERVERS); do
  VOLUMES="$VOLUMES -v $(pwd)/$BASE_DIR/shared_data_$i:/app/$BASE_DIR/shared_data_$i"
done
VOLUMES="$VOLUMES -v $(pwd):/app"

echo "All servers are running and verified."
echo "Volume mounts for calibration container:"
echo "$VOLUMES"

# At this point, you can run your calibration container with the prepared volume mounts
# Example (commented out):
# docker run --rm \
#   $VOLUMES \
#   -e NUM_SERVERS=$NUM_SERVERS \
#   -e NUM_CALIBRATION=$NUM_CALIBRATION \
#   -e VERSION=$VERSION \
#   beforeit-calibration:latest