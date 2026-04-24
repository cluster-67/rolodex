#!/bin/bash

module load cray-hdf5 PrgEnv-gnu

cd "$(dirname "$0")/build"
make knn knn_openmp

{
    echo "=== SERIAL ==="
    { time ./knn; }

    echo "=== OPENMP 2 threads ==="
    { time OMP_NUM_THREADS=2 ./knn_openmp; }

    echo "=== OPENMP 4 threads ==="
    { time OMP_NUM_THREADS=4 ./knn_openmp; }

    echo "=== OPENMP 8 threads ==="
    { time OMP_NUM_THREADS=8 ./knn_openmp; }
} 2>&1 | tee ../comparison.out
