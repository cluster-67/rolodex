#include "rolodex/dataset.hpp"

#include <H5Cpp.h>
#include <iostream>

Dataset::Dataset(std::string filename) {
    filename_ = filename;
}

void Dataset::load_dataset() {
    H5::H5File file(filename_, H5F_ACC_RDONLY);

    H5::DataSet dataset = file.openDataSet("/train");
    H5::DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    if (rank != 2) {
        std::cerr << "Expected /train to be rank-2, got rank " << rank << std::endl;
        return;
    }

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims);
    const auto nrows = static_cast<std::size_t>(dims[0]);
    const auto ncols = static_cast<std::size_t>(dims[1]);

    std::vector<float> raw(nrows * ncols);
    dataset.read(raw.data(), H5::PredType::NATIVE_FLOAT);

    points_.assign(nrows, TVector(ncols));
    for (std::size_t i = 0; i < nrows; ++i) {
        for (std::size_t j = 0; j < ncols; ++j) {
            points_[i][j] = raw[i * ncols + j];
        }
    }

    std::cout << "Loaded /train into points with shape " << nrows << "x" << ncols << std::endl;
}

std::vector<TVector> &Dataset::get_points() {
    return points_;
}
