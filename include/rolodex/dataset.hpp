#pragma once

#include "rolodex/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

class Dataset {
  public:
    explicit Dataset(std::string filename);

    void load_dataset();

    // Flat interface — used by the serial implementation.
    // Primary storage: data_[i * ncols_ + j]
    const float* get_flat()  const { return data_.data(); }
    std::size_t  n_points()  const { return nrows_; }
    std::size_t  dim()       const { return ncols_; }

    // TVector interface — used by OpenMP and MPI implementations.
    // Built lazily from the flat buffer on first call; zero extra cost for
    // the serial binary which never calls this.
    std::vector<TVector>& get_points();

  private:
    std::string        filename_;
    std::vector<float> data_;          // flat row-major primary storage
    std::size_t        nrows_ = 0;
    std::size_t        ncols_ = 0;
    std::vector<TVector> points_cache_; // populated on demand by get_points()
};
