#include "rolodex/dataset.hpp"
#include "rolodex/kmeans.hpp"
#include "rolodex/utils.hpp"

#include <iostream>
#include <mpi.h>
#include <string>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string dataset_file = utils::path::dataset_path("fashion-mnist-784-euclidean.hdf5");

    Dataset dataset(dataset_file);
    if (rank == 0) {
        dataset.load_dataset();
    }

    MPIKMeans knn_algorithm(&dataset, 10, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    knn_algorithm.create_clusters();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "MPI k-means finished in " << (MPI_Wtime() - t0) << " s\n";
    }

    MPI_Finalize();
    return 0;
}