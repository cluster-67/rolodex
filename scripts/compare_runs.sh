#!/bin/bash

module load cray-hdf5 PrgEnv-gnu

cd "$(dirname "$0")/build"
make knn

# Implementation (Serial vs OpenMP) is selected in src/knn_main.cpp RunConfig.
{
    echo "=== knn (set RunImplementation in knn_main.cpp) ==="
    { time ./knn; }

    echo "=== same binary, OMP_NUM_THREADS=2 (affects OpenMP path only) ==="
    { time OMP_NUM_THREADS=2 ./knn; }

    echo "=== same binary, OMP_NUM_THREADS=4 ==="
    { time OMP_NUM_THREADS=4 ./knn; }

    echo "=== same binary, OMP_NUM_THREADS=8 ==="
    { time OMP_NUM_THREADS=8 ./knn; }
} 2>&1 | tee ../comparison.out
