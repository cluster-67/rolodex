#!/bin/bash

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/utils.sh"

cd "$ROOT_DIR"

module load cray-hdf5 cray-mpich

color_echo "blue" "Building knn..."
cmake --build "$BUILD_DIR" --target knn --parallel

LOGS_DIR="$ROOT_DIR/data/logs/bench-k-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOGS_DIR"
color_echo "yellow" "Logs will be saved to $LOGS_DIR"

DATASET="${1:-fashion-mnist}"
K_VALUES=(1024)

color_echo "blue" "Running k-scaling benchmark (openmp + mpi only) on dataset=$DATASET"

for k in "${K_VALUES[@]}"; do
    for threads in 2 4 8; do
        color_echo "yellow" "k=$k: openmp threads=$threads"
        OMP_NUM_THREADS=$threads "$BUILD_DIR/knn" openmp --dataset "$DATASET" -k "$k" 2>&1 | tee "$LOGS_DIR/k${k}-openmp-t${threads}.log"
    done

    run_configs=(
        "-N 1 -n 2"
        "-N 1 -n 4"
        "-N 1 -n 8"
        "-N 2 -n 2"
        "-N 2 -n 4"
        "-N 2 -n 8"
    )

    for run_config in "${run_configs[@]}"; do
        N=$(echo "$run_config" | grep -oP "(?<=-N )\d+")
        n=$(echo "$run_config" | grep -oP "(?<=-n )\d+")
        color_echo "yellow" "k=$k: mpi N=$N n=$n"
        srun $run_config "$BUILD_DIR/knn" mpi --dataset "$DATASET" -k "$k" 2>&1 | tee "$LOGS_DIR/k${k}-mpi-N${N}-n${n}.log"
    done
done

color_echo "green" "Done! Logs saved to $LOGS_DIR"
color_echo "green" "Run: python3 scripts/plot_k_scaling.py $LOGS_DIR"
