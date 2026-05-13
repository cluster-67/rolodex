#!/bin/bash

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/utils.sh"

cd "$ROOT_DIR"

module load cray-hdf5 cray-mpich

color_echo "blue" "Building knn..."
cmake --build "$BUILD_DIR" --target knn --parallel

LOGS_DIR="$ROOT_DIR/data/logs/bench-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOGS_DIR"
color_echo "yellow" "Logs will be saved to $LOGS_DIR"

color_echo "blue" "Running knn implementations..."

NUM_VALIDATIONS=100
N_PROBES=1

DATASETS=(
    "fashion-mnist"
    "mnist"
    "sift"
    "gist"
    # "deep1b-1B"
)

for dataset in "${DATASETS[@]}"; do
    mkdir -p "$LOGS_DIR/$dataset"

    CLI_ARGS="--dataset $dataset --validation-count $NUM_VALIDATIONS --nprobe $N_PROBES"
    
    color_echo "yellow" "Running knn with serial for dataset=$dataset"
    "$BUILD_DIR/knn" "serial" $CLI_ARGS 2>&1 | tee "$LOGS_DIR/$dataset/serial.log"

    color_echo "yellow" "Running knn with openmp for dataset=$dataset"
    THREAD_COUNTS=(
        2
        4
        8
        16
        32
    )
    for threads in "${THREAD_COUNTS[@]}"; do
        color_echo "yellow" "Running knn with openmp threads=$threads for dataset=$dataset"
        OMP_NUM_THREADS=$threads "$BUILD_DIR/knn" openmp $CLI_ARGS 2>&1 | tee "$LOGS_DIR/$dataset/openmp-t$threads.log"
    done

    color_echo "yellow" "Running knn with mpi for dataset=$dataset"
    run_configs=(
        "-N 1 -n 2"
        "-N 1 -n 4"
        "-N 1 -n 8"
        "-N 1 -n 16"
        "-N 1 -n 32"
        "-N 2 -n 2"
        "-N 2 -n 4"
        "-N 2 -n 8"
        "-N 2 -n 16"
        "-N 2 -n 32"
        "-N 4 -n 4"
        "-N 4 -n 8"
        "-N 4 -n 16"
        "-N 4 -n 32"
    )

    for run_config in "${run_configs[@]}"; do
        # Extract N and n from the run_config string
        N=$(echo "$run_config" | grep -oP "(?<=-N )\d+")
        n=$(echo "$run_config" | grep -oP "(?<=-n )\d+")
        color_echo "yellow" "Running knn with mpi N=$N n=$n for dataset=$dataset"
        srun $run_config $BUILD_DIR/knn mpi $CLI_ARGS 2>&1 | tee "$LOGS_DIR/$dataset/mpi-N$N-n$n.log"
    done

done

color_echo "green" "Logs saved to $LOGS_DIR"

color_echo "green" "Plotting benchmarks..."
module load python
python3 scripts/plot_benchmarks.py "$LOGS_DIR"

color_echo "green" "Done!"
