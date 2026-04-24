#pragma once

#include "rolodex/types.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

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
    void load_validation_dataset(int count = -1);

    std::vector<TVector> &get_points();
    const std::vector<ValidationPoint> &get_validation_points() const;

  private:
    std::string filename_;
    std::vector<TVector> points_;
    std::vector<ValidationPoint> validation_points_;
    std::vector<TVector> read_train_rows(const std::vector<int> &rows);
};
