#!/bin/bash

set -e

source "$(dirname "$(readlink -f "$0")")/utils.sh"

cd "$ROOT_DIR"

module load cray-hdf5 cray-mpich

cd "$BUILD_DIR"
color_echo "blue" "Building..."
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --target knn --parallel

DATASET="${1:-fashion-mnist}"
NODES=1
TOTAL_TASKS=1
THREADS_PER_TASK=64

color_echo "blue" "Running knn (openmp) and profiling... [DATASET=$DATASET, NODES=$NODES, TOTAL_TASKS=$TOTAL_TASKS]"
srun -N $NODES -n $TOTAL_TASKS \
  bash -c 'OMP_NUM_THREADS=$THREADS_PER_TASK perf record -F 999 --call-graph dwarf -o $SCRATCH/perf-$SLURM_PROCID.data '"$BUILD_DIR/knn"' openmp --dataset '"$DATASET"

rm -f $SCRATCH/perf-*.data.old

PERF_INPUTS=""
for i in $(seq 0 $((TOTAL_TASKS - 1))); do
  PERF_INPUTS="$PERF_INPUTS -i $SCRATCH/perf-$i.data"
done

color_echo "blue" "Processing perf data..."
perf script $PERF_INPUTS -F +srcline 2>/dev/null \
| ~/code/FlameGraph/stackcollapse-perf.pl  --srcline 2>/dev/null \
| ~/code/FlameGraph/flamegraph.pl > flame-srcline.openmp.svg

perf script $PERF_INPUTS 2>/dev/null \
| ~/code/FlameGraph/stackcollapse-perf.pl 2>/dev/null \
| ~/code/FlameGraph/flamegraph.pl > flame.openmp.svg

perf report $PERF_INPUTS --stdio --sort=dso,symbol -n --percent-limit 0.5 > out-report-1
perf report $PERF_INPUTS --stdio -g graph,0.5,caller --sort=symbol -n --percent-limit 0.5 > out-report-2

color_echo "yellow" "Output files..."
ls -lh $PWD/flame-srcline.openmp.svg $PWD/flame.openmp.svg $PWD/out-report-1 $PWD/out-report-2

color_echo "green" "Success!"
