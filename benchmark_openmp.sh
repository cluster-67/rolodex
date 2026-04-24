#!/bin/bash

module load cray-hdf5 PrgEnv-gnu

cd "$(dirname "$0")/build"
make knn

OUT="../update_iters.out"
> "$OUT"

THREAD_COUNTS=(2 4 8)

echo "=== SERIAL baseline (RunImplementation::Serial in knn_main.cpp) ===" | tee -a "$OUT"
real=$({ time ./knn > /dev/null; } 2>&1 | grep real | awk '{print $2}')
echo "threads=1  real=${real}" | tee -a "$OUT"
echo "" | tee -a "$OUT"

echo "=== OpenMP thread sweep (set RunImplementation::OpenMP in knn_main.cpp) ===" | tee -a "$OUT"
echo "update_frequency is RunConfig.update_frequency in knn_main.cpp (no CLI yet)" | tee -a "$OUT"
for threads in "${THREAD_COUNTS[@]}"; do
    real=$({ time OMP_NUM_THREADS=$threads ./knn > /dev/null; } 2>&1 | grep real | awk '{print $2}')
    echo "threads=${threads}  real=${real}" | tee -a "$OUT"
done
