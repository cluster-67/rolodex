#!/bin/bash

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/utils.sh"

cd "$ROOT_DIR"

module load cray-hdf5 cray-mpich

DATASET="${1:-mnist}"
K="${2:-10}"
THREADS="${3:-2}"
MAX_UPDATE_FREQUENCY="${4:-10}"
VALIDATION_COUNT="${5:-100}"
TOP_K="${6:-5}"
NPROBE="${7:-1}"

color_echo "blue" "Building knn..."
cmake --build "$BUILD_DIR" --target knn --parallel

LOGS_DIR="$ROOT_DIR/data/logs/bench-update-frequency-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOGS_DIR"
color_echo "yellow" "Logs will be saved to $LOGS_DIR"

color_echo "blue" \
    "Running OpenMP update-frequency sweep: dataset=$DATASET k=$K threads=$THREADS validation_count=$VALIDATION_COUNT"

for update_frequency in $(seq 1 "$MAX_UPDATE_FREQUENCY"); do
    log_file="$LOGS_DIR/update-frequency-${update_frequency}-openmp-t${THREADS}.log"
    color_echo "yellow" "update_frequency=$update_frequency"
    OMP_NUM_THREADS="$THREADS" "$BUILD_DIR/knn" openmp \
        --dataset "$DATASET" \
        -k "$K" \
        --update-frequency "$update_frequency" \
        --validation-count "$VALIDATION_COUNT" \
        --top-k "$TOP_K" \
        --nprobe "$NPROBE" \
        2>&1 | tee "$log_file"
done

color_echo "green" "Done! Logs saved to $LOGS_DIR"
color_echo "green" "Run: python3 scripts/plot_update_frequency.py $LOGS_DIR"
