#pragma once

#include "rolodex/types.hpp"

#include <string>
#include <vector>

class Dataset {
  public:
    explicit Dataset(std::string filename);

    void load_dataset();

    std::vector<TVector> &get_points();

  private:
    std::string filename_;
    std::vector<TVector> points_;
};
