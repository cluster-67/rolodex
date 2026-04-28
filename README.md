# Rolodex

## Commands
```bash
module load cray-hdf5 cray-mpich

mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC
make knn

# Run: positional impl + flags
./knn serial
./knn openmp --update-frequency 1
srun -n 8 ./knn mpi

# Build and run from root dir
cmake --build build --target knn --parallel && ./build/knn serial
```
