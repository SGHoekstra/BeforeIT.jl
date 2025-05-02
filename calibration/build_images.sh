#!/bin/bash
# build_images.sh - Builds the docker images for BeforeIT

echo "Building BeforeIT Julia server image..."
docker build -t beforeit-julia-server:latest -f Dockerfile.julia .

echo "Building BeforeIT calibration image..."
docker build -t beforeit-calibration:latest -f Dockerfile.python .

echo "Image build complete!"
echo "You can now run the calibration using ./run_calibration.sh"