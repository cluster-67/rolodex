#pragma once

#include "rolodex/types.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace H5 {
class DataSet;
}

struct ValidationPoint {
    TVector query;
    std::vector<TVector> neighbors;
    std::vector<float> distances;

    std::string to_string(std::size_t num_neighbors) const {
        auto format_vector = [](const TVector &vec) {
            std::ostringstream out;
            out << "[";
            for (std::size_t i = 0; i < vec.size(); ++i) {
                if (i > 0) {
                    out << ",";
                }
                out << vec[i];
            }
            out << "]";
            return out.str();
        };

        std::ostringstream out;
        out << "query=" << format_vector(query);

        const std::size_t available = std::min(neighbors.size(), distances.size());
        const std::size_t limit = std::min(num_neighbors, available);
        for (std::size_t i = 0; i < limit; ++i) {
            out << "\n> [" << (i + 1) << "] distance=" << distances[i] << " "
                << format_vector(neighbors[i]);
        }
        return out.str();
    }
};

class Dataset {
  public:
    explicit Dataset(std::string filename);

    void load_dataset();
    /** Load contiguous /train rows for MPI rank `mpi_rank` of `mpi_size` (same layout as
     * MPI_Scatterv). Requires `mpi_size` <= global train row count. Sets `train_file_nrows_` to
     * full file N. */
    void load_train_partition(int mpi_rank, int mpi_size);
    void load_validation_dataset(int count = -1);

    /** Global /train row count from file (equals `n_points()` after full `load_dataset()`). */
    std::size_t train_file_nrows() const {
        return train_file_nrows_;
    }

    /** Read one /train row by global index (HDF5); for centroid init when the full train matrix is
     * not in memory. */
    TVector read_train_row_global(std::size_t global_row) const;

    const std::string &filename() const;
    const std::vector<ValidationPoint> &get_validation_points() const;

    // Flat interface — used by the serial implementation.
    // Primary storage: data_[i * ncols_ + j]
    const float *get_flat() const {
        return data_.data();
    }
    std::size_t n_points() const {
        return nrows_;
    }
    std::size_t dim() const {
        return ncols_;
    }

    // TVector interface — used by OpenMP and MPI implementations.
    // Built lazily from the flat buffer on first call; zero extra cost for
    // the serial binary which never calls this.
    std::vector<TVector> &get_points();

  private:
    std::string filename_;
    TAlignedVector data_; // flat row-major primary storage
    std::size_t nrows_ = 0;
    std::size_t ncols_ = 0;
    /** Global row count of /train in the file (set on train loads). */
    std::size_t train_file_nrows_ = 0;
    std::vector<TVector> points_cache_; // populated on demand by get_points()
    std::vector<ValidationPoint> validation_points_;
    std::vector<TVector> read_train_rows(H5::DataSet *train_dataset, const std::vector<int> &rows);
};
