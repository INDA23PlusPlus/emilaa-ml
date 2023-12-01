#pragma once

#include <vector>

struct DataPoint {
    std::vector<float> input_data{};
    std::vector<float> expected{};
    int label = 0;
};

std::vector<DataPoint> read_mnist_n(const char *images_path, const char *labels_path, size_t n);