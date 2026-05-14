#!/bin/bash
#
# Sweep Slurm node count N ∈ {1,2,4} and parallelism P ∈ {2,4,8,16,32,64,128}.
# Each P is paired with a fixed deep1b train size (weak-style problem growth):
#   P=2   -> deep1b-1K
#   P=4   -> deep1b-10K
#   P=8   -> deep1b-100K
#   P=16  -> deep1b-1M
#   P=32  -> deep1b-10M
#   P=64  -> deep1b-100M
#   P=128 -> deep1b-1B
#
# OpenMP: only srun -N 1 -n 1 (multi-node -N>1 with -n 1 is rejected by srun); OMP_NUM_THREADS=P.
# MPI:    srun -N N -n P with mpi --partition only when P >= N (Slurm rejects -n < -N).
#
# Optional env:
#   NUM_VALIDATIONS  (default 10)
#   N_PROBES         (default 1)
#   SKIP_OPENMP=1 / SKIP_MPI=1

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/utils.sh"

cd "$ROOT_DIR"

module load cray-hdf5 cray-mpich

color_echo "blue" "Building knn..."
cmake --build "$BUILD_DIR" --target knn --parallel

LOGS_DIR="$ROOT_DIR/data/logs/bench-weak-$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$LOGS_DIR"
color_echo "yellow" "Logs will be saved to $LOGS_DIR"

NUM_VALIDATIONS="${NUM_VALIDATIONS:-10}"
N_PROBES="${N_PROBES:-1}"

NODE_COUNTS=(1 2 4)
PROC_COUNTS=(2 4 8 16 32)
# Same index as PROC_COUNTS: global train rows scale with P
DEEP1B_DATASETS=(
    deep1b-1K
    deep1b-10K
    deep1b-100K
    deep1b-1M
    deep1b-10M
    # deep1b-100M
    # deep1b-1B
)

if [[ ${#PROC_COUNTS[@]} -ne ${#DEEP1B_DATASETS[@]} ]]; then
    color_echo "red" "Internal error: PROC_COUNTS and DEEP1B_DATASETS length mismatch"
    exit 1
fi

color_echo "blue" "NUM_VALIDATIONS=$NUM_VALIDATIONS  N_PROBES=$N_PROBES"
color_echo "blue" "Grid: N_nodes ∈ {${NODE_COUNTS[*]}} × (P → dataset) pairs:"
for i in "${!PROC_COUNTS[@]}"; do
    color_echo "blue" "  P=${PROC_COUNTS[$i]}  ->  ${DEEP1B_DATASETS[$i]}"
done

for N in "${NODE_COUNTS[@]}"; do
    for i in "${!PROC_COUNTS[@]}"; do
        procs="${PROC_COUNTS[$i]}"
        dataset="${DEEP1B_DATASETS[$i]}"

        mkdir -p "$LOGS_DIR/$dataset"
        CLI_BASE=(--dataset "$dataset" --validation-count "$NUM_VALIDATIONS" --nprobe "$N_PROBES")

        if [[ "${SKIP_OPENMP:-0}" != "1" ]]; then
            if (( N > 1 )); then
                color_echo "yellow" "openmp  skip N=$N P=$procs dataset=$dataset (avoid srun -N $N -n 1)"
            else
                color_echo "yellow" "openmp  dataset=$dataset  OMP_NUM_THREADS=$procs  srun -N $N -n 1"
                OMP_NUM_THREADS="$procs" OMP_PROC_BIND=spread OMP_PLACES=threads \
                    srun -N "$N" -n 1 --export=ALL \
                    "$BUILD_DIR/knn" openmp "${CLI_BASE[@]}" 2>&1 \
                    | tee "$LOGS_DIR/$dataset/openmp-N${N}-t${procs}.log"
            fi
        fi

        if [[ "${SKIP_MPI:-0}" != "1" ]]; then
            if (( procs < N )); then
                color_echo "yellow" "mpi  skip N=$N P=$procs dataset=$dataset (require P >= N for srun -n vs -N)"
            else
                color_echo "yellow" "mpi --partition  dataset=$dataset  srun -N $N -n $procs"
                srun -N "$N" -n "$procs" \
                    "$BUILD_DIR/knn" mpi "${CLI_BASE[@]}" --partition 2>&1 \
                    | tee "$LOGS_DIR/$dataset/mpi-partition-N${N}-n${procs}.log"
            fi
        fi
    done
done

color_echo "green" "Done. Logs under $LOGS_DIR/<dataset>/ (one subdirectory per deep1b-* size)"
