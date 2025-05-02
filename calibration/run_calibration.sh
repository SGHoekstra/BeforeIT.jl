#!/bin/bash
# run_calibration.sh - Runs calibration using existing BeforeIT images

# Get version to calibrate
VERSION=${1:-2}  # Default to version 2 if not specified
NUM_SERVERS=${2:-4}  # Default to 4 servers if not specified
NUM_CPUS=${3:-2}  # Default to 4 CPUs if not specified
NUM_CALIBRATION=${4:-10}  # Default to 100 iterations if not specified


# Create necessary directories
echo "Creating shared directories..."
for i in $(seq 1 $NUM_SERVERS); do
  mkdir -p "shared_data_$i/input" "shared_data_$i/output"
done
mkdir -p extended_heuristic

# Check if servers are running; start them if they're not
for i in $(seq 1 $NUM_SERVERS); do
  CONTAINER_NAME="beforeit-julia-server-$i"
  
  if [ -z "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Starting $CONTAINER_NAME..."
    docker run -d \
      --name $CONTAINER_NAME \
      -v "$(pwd)/shared_data_$i:/data" \
      -e BEFOREIT_INPUT_DIR=/data/input \
      -e BEFOREIT_OUTPUT_DIR=/data/output \
      -e BEFOREIT_SERVER_ID=$i \
      -e JULIA_NUM_THREADS=$NUM_CPUS \
      beforeit-julia-server:latest
  else
    echo "$CONTAINER_NAME is already running"
  fi
done

# Prepare volume mounts for calibration container
VOLUMES=""
for i in $(seq 1 $NUM_SERVERS); do
  VOLUMES="$VOLUMES -v $(pwd)/shared_data_$i:/app/shared_data_$i"
done
VOLUMES="$VOLUMES -v $(pwd)/extended_heuristic:/app/extended_heuristic"

# Run the calibration with correctly formatted arguments
echo "Running calibration for version $VERSION with $NUM_SERVERS servers..."
docker run --rm $VOLUMES beforeit-calibration:latest \
  python /app/file_calibrate.py \
  --version "$VERSION" \
  --num-servers "$NUM_SERVERS" \
  --base-dir "/app/shared_data_" \
  --max-iterations "$NUM_CALIBRATION" \


echo "Calibration complete. Results are in the extended_heuristic directory."