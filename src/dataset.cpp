#include "rolodex/dataset.hpp"

#include <H5Cpp.h>
#include <algorithm>
#include <iostream>

Dataset::Dataset(std::string filename) : filename_(filename) {}

void Dataset::load_dataset() {
    H5::H5File    file(filename_, H5F_ACC_RDONLY);
    H5::DataSet   ds = file.openDataSet("/train");
    H5::DataSpace sp = ds.getSpace();

    if (sp.getSimpleExtentNdims() != 2) {
        std::cerr << "Expected /train to be rank-2\n";
        return;
    }

    hsize_t dims[2];
    sp.getSimpleExtentDims(dims);
    nrows_ = static_cast<std::size_t>(dims[0]);
    ncols_ = static_cast<std::size_t>(dims[1]);

    // Read directly into the flat buffer — no scatter into TVector.
    data_.resize(nrows_ * ncols_);
    ds.read(data_.data(), H5::PredType::NATIVE_FLOAT);

    std::cout << "Loaded /train: " << nrows_ << " x " << ncols_
              << " (flat buffer " << data_.size() * sizeof(float) / (1 << 20) << " MiB)\n";
}

// Built lazily — only called by OpenMP/MPI paths; serial never touches this.
std::vector<TVector>& Dataset::get_points() {
    if (points_cache_.empty() && nrows_ > 0) {
        points_cache_.resize(nrows_, TVector(ncols_));
        for (std::size_t i = 0; i < nrows_; i++) {
            std::copy(data_.data() + i * ncols_,
                      data_.data() + i * ncols_ + ncols_,
                      points_cache_[i].data());
        }
    }
    return points_cache_;
}