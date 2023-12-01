#pragma once

#include "MNIST/mnist.hpp"
#include "layer.hpp"

struct Classified {
    size_t predicted;
    std::vector<float> outputs;
};

class Network {
    public:
        Network(const std::vector<size_t> layer_sizes);

        Classified classify(const DataPoint &input);

        void learn(const std::vector<DataPoint> &data, const float learn_rate);

        std::vector<Layer> layers;
        std::vector<size_t> layer_sizes;
    private:
        std::vector<float> calculate_outputs(const std::vector<float> &inputs);
        void update_gradients(const DataPoint& data_point, NetworkData &instance_data);
        size_t get_index_of_hot(const std::vector<float> &outputs);
};