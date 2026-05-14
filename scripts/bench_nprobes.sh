#!/bin/bash

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/utils.sh"

cd "$ROOT_DIR"

module load cray-hdf5 cray-mpich

K="${1:-10}"
THREADS="${2:-8}"
VALIDATION_COUNT="${3:-1000}"
TOP_K="${4:-10}"
NPROBES_RAW="${5:-1,2,3,4,5,6,7,8,9,10}"

DATASETS=(mnist fashion-mnist gist sift)

color_echo "blue" "Building knn..."
cmake --build "$BUILD_DIR" --target knn --parallel

LOGS_DIR="$ROOT_DIR/data/logs/bench-nprobes-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOGS_DIR"
color_echo "yellow" "Logs will be saved to $LOGS_DIR"

NPROBES_CLEAN="${NPROBES_RAW//,/ }"
read -r -a NPROBES <<< "$NPROBES_CLEAN"

if [ "${#NPROBES[@]}" -eq 0 ]; then
    color_echo "red" "No nprobes provided. Pass a comma-separated list, e.g. 1,2,4,8,10."
    exit 1
fi

for dataset in "${DATASETS[@]}"; do
    DATASET_LOGS_DIR="$LOGS_DIR/$dataset"
    mkdir -p "$DATASET_LOGS_DIR"

    color_echo "blue" \
        "Running OpenMP nprobe sweep: dataset=$dataset k=$K threads=$THREADS validation_count=$VALIDATION_COUNT top_k=$TOP_K"

    for nprobe in "${NPROBES[@]}"; do
        if ! [[ "$nprobe" =~ ^[0-9]+$ ]] || [ "$nprobe" -le 0 ]; then
            color_echo "red" "Invalid nprobe value: '$nprobe' (must be a positive integer)."
            exit 1
        fi

        log_file="$DATASET_LOGS_DIR/nprobe-${nprobe}-openmp-t${THREADS}.log"
        color_echo "yellow" "dataset=$dataset nprobe=$nprobe"
        OMP_NUM_THREADS="$THREADS" "$BUILD_DIR/knn" openmp \
            --dataset "$dataset" \
            -k "$K" \
            --validation-count "$VALIDATION_COUNT" \
            --top-k "$TOP_K" \
            --nprobe "$nprobe" \
            2>&1 | tee "$log_file"
    done
done

color_echo "green" "Done! Logs saved to $LOGS_DIR"
color_echo "green" "Run: python3 scripts/plot_nprobes.py $LOGS_DIR"
