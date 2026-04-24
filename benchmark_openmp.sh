#!/bin/bash

module load cray-hdf5 PrgEnv-gnu

cd "$(dirname "$0")/build"
make knn knn_openmp

OUT="../update_iters.out"
> "$OUT"

THREAD_COUNTS=(2 4 8)
UPDATE_FREQS=(1 3 5 7 9 11 13 15 17 19)

# Serial baseline
echo "=== SERIAL ===" | tee -a "$OUT"
real=$({ time ./knn > /dev/null; } 2>&1 | grep real | awk '{print $2}')
echo "threads=1  update_freq=N/A  real=${real}" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# OpenMP runs
echo "=== OPENMP ===" | tee -a "$OUT"
for threads in "${THREAD_COUNTS[@]}"; do
    for freq in "${UPDATE_FREQS[@]}"; do
        real=$({ time OMP_NUM_THREADS=$threads ./knn_openmp $freq > /dev/null; } 2>&1 | grep real | awk '{print $2}')
        echo "threads=${threads}  update_freq=${freq}  real=${real}" | tee -a "$OUT"
    done
    echo "" | tee -a "$OUT"
done
