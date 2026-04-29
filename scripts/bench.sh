#!/bin/bash

set -e

source "$(dirname "$(readlink -f "$0")")/utils.sh"

cd "$ROOT_DIR"

module load cray-hdf5 cray-mpich

color_echo "blue" "Building knn..."
cmake --build "$BUILD_DIR" --target knn --parallel

LOGS_DIR="$ROOT_DIR/data/logs/bench-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOGS_DIR"
color_echo "yellow" "Logs will be saved to $LOGS_DIR"

color_echo "blue" "Running knn implementations..."

color_echo "yellow" "Running knn with serial..."
"$BUILD_DIR/knn" "serial" --cache 2>&1 | tee "$LOGS_DIR/serial.log"

color_echo "yellow" "Running knn with openmp..."
for threads in 2 4 8; do
    color_echo "yellow" "Running knn with openmp threads=$threads"
    OMP_NUM_THREADS=$threads "$BUILD_DIR/knn" openmp 2>&1 | tee "$LOGS_DIR/openmp-t$threads.log"
done

color_echo "yellow" "Running knn with mpi..."
run_configs=(
    "-N 1 -n 2"
    "-N 1 -n 4"
    "-N 1 -n 8"
    "-N 2 -n 2"
    "-N 2 -n 4"
    "-N 2 -n 8"
    # "-N 4 -n 4"
    # "-N 4 -n 8"
)

for run_config in "${run_configs[@]}"; do
    # Extract N and n from the run_config string
    N=$(echo "$run_config" | grep -oP "(?<=-N )\d+")
    n=$(echo "$run_config" | grep -oP "(?<=-n )\d+")
    color_echo "yellow" "Running knn with mpi N=$N n=$n"
    srun $run_config $BUILD_DIR/knn mpi 2>&1 | tee "$LOGS_DIR/mpi-N$N-n$n.log"
done

color_echo "green" "Done! Logs saved to $LOGS_DIR"
