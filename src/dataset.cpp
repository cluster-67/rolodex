#include "rolodex/dataset.hpp"

#include <H5Cpp.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

Dataset::Dataset(std::string filename) : filename_(std::move(filename)) {}

void Dataset::load_dataset() {
    H5::H5File file(filename_, H5F_ACC_RDONLY);
    H5::DataSet ds = file.openDataSet("/train");
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

    std::cout << "Loaded /train: " << nrows_ << " x " << ncols_ << " (flat buffer "
              << data_.size() * sizeof(float) / (1 << 20) << " MiB)\n";
}

// Built lazily — only called by OpenMP/MPI paths; serial never touches this.
std::vector<TVector> &Dataset::get_points() {
    if (points_cache_.empty() && nrows_ > 0) {
        points_cache_.resize(nrows_, TVector(ncols_));
        for (std::size_t i = 0; i < nrows_; i++) {
            std::copy(data_.data() + i * ncols_, data_.data() + i * ncols_ + ncols_,
                      points_cache_[i].data());
        }
    }
    return points_cache_;
}

const std::string &Dataset::filename() const {
    return filename_;
}

void Dataset::load_validation_dataset(int count) {
    if (!validation_points_.empty()) {
        return;
    }

    H5::H5File file(filename_, H5F_ACC_RDONLY);

    H5::DataSet dataset = file.openDataSet("/test");
    H5::DataSet train_dataset = file.openDataSet("/train");
    H5::DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    if (rank != 2) {
        throw std::runtime_error("Expected /test to be rank-2");
    }

    hsize_t test_dims[2];
    dataspace.getSimpleExtentDims(test_dims);
    const auto num_queries = static_cast<std::size_t>(test_dims[0]);
    const auto dim = static_cast<std::size_t>(test_dims[1]);
    const std::size_t queries_to_load =
        (count < 0) ? num_queries : std::min(num_queries, static_cast<std::size_t>(count));

    std::vector<float> test_raw(num_queries * dim);
    dataset.read(test_raw.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet neighbors_dataset = file.openDataSet("/neighbors");
    H5::DataSpace neighbors_space = neighbors_dataset.getSpace();
    if (neighbors_space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Expected /neighbors to be rank-2");
    }

    hsize_t neighbors_dims[2];
    neighbors_space.getSimpleExtentDims(neighbors_dims);
    const auto neighbors_rows = static_cast<std::size_t>(neighbors_dims[0]);
    const auto top_k = static_cast<std::size_t>(neighbors_dims[1]);
    if (neighbors_rows != num_queries) {
        throw std::runtime_error("Expected /neighbors rows to match /test rows");
    }

    std::vector<int> neighbors_raw(neighbors_rows * top_k);
    neighbors_dataset.read(neighbors_raw.data(), H5::PredType::NATIVE_INT);

    H5::DataSet distances_dataset = file.openDataSet("/distances");
    H5::DataSpace distances_space = distances_dataset.getSpace();
    if (distances_space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Expected /distances to be rank-2");
    }

    hsize_t distances_dims[2];
    distances_space.getSimpleExtentDims(distances_dims);
    const auto distances_rows = static_cast<std::size_t>(distances_dims[0]);
    const auto distances_k = static_cast<std::size_t>(distances_dims[1]);
    if (distances_rows != num_queries || distances_k != top_k) {
        throw std::runtime_error("Expected /distances shape to match /neighbors shape");
    }

    std::vector<float> distances_raw(distances_rows * distances_k);
    distances_dataset.read(distances_raw.data(), H5::PredType::NATIVE_FLOAT);

    validation_points_.clear();
    validation_points_.reserve(queries_to_load);
    for (std::size_t i = 0; i < queries_to_load; ++i) {
        ValidationPoint validation_point;
        validation_point.query.reserve(dim);
        for (std::size_t j = 0; j < dim; ++j) {
            validation_point.query.push_back(test_raw[i * dim + j]);
        }
        std::vector<int> neighbor_indices;
        neighbor_indices.reserve(top_k);
        validation_point.distances.reserve(top_k);

        for (std::size_t j = 0; j < top_k; ++j) {
            const std::size_t idx = i * top_k + j;
            const int neighbor_idx = neighbors_raw[idx];
            if (neighbor_idx < 0) {
                throw std::runtime_error("Neighbor index out of bounds for /train");
            }
            neighbor_indices.push_back(neighbor_idx);
            validation_point.distances.push_back(distances_raw[idx]);
        }
        validation_point.neighbors = read_train_rows(&train_dataset, neighbor_indices);
        validation_points_.push_back(std::move(validation_point));
    }

    std::cout << "Loaded validation dataset: /test with " << queries_to_load << " queries and top-"
              << top_k << " neighbors per query\n";
}

const std::vector<ValidationPoint> &Dataset::get_validation_points() const {
    return validation_points_;
}

std::vector<TVector> Dataset::read_train_rows(H5::DataSet *train_dataset,
                                              const std::vector<int> &rows) {
    if (train_dataset == nullptr) {
        throw std::runtime_error("Train dataset handle is null");
    }

    H5::DataSpace dataspace = train_dataset->getSpace();
    const int rank = dataspace.getSimpleExtentNdims();
    if (rank != 2) {
        throw std::runtime_error("Expected /train to be rank-2");
    }

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims);
    const std::size_t nrows = static_cast<std::size_t>(dims[0]);
    const std::size_t ncols = static_cast<std::size_t>(dims[1]);

    std::vector<TVector> result;
    result.reserve(rows.size());

    for (int row : rows) {
        if (row < 0 || static_cast<std::size_t>(row) >= nrows) {
            throw std::runtime_error("Neighbor index out of bounds for /train");
        }

        H5::DataSpace row_space = train_dataset->getSpace();
        hsize_t offset[2] = {static_cast<hsize_t>(row), 0};
        hsize_t count[2] = {1, dims[1]};
        row_space.selectHyperslab(H5S_SELECT_SET, count, offset);

        hsize_t mem_dims[2] = {1, dims[1]};
        H5::DataSpace memspace(2, mem_dims);

        TVector buffer(ncols);
        train_dataset->read(buffer.data(), H5::PredType::NATIVE_FLOAT, memspace, row_space);
        result.push_back(std::move(buffer));
    }

    return result;
}
