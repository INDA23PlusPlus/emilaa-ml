#include "network.hpp"

#include <thread>
#include <numeric>

using namespace std;

Network::Network(const vector<size_t> sizes) {
    layer_sizes = sizes;
    layers.reserve(sizes.size());

    for(size_t i = 0; i < sizes.size() - 1; i++) {
        layers.push_back({layer_sizes[i], layer_sizes[i+1]});
    }
}

Classified Network::classify(const DataPoint &input) {
    auto outputs = calculate_outputs(input.input_data);
    size_t predicted = get_index_of_hot(outputs);
    return {predicted, outputs};
}

size_t Network::get_index_of_hot(const vector<float> &outputs) {
    float max = -numeric_limits<float>::max();
    size_t max_index = 0;
    for(size_t i = 0; i < outputs.size(); i++) {
        if(outputs[i] > max) { max_index = i; max = outputs[i]; }
    }
    return max_index;
}

vector<float> Network::calculate_outputs(const std::vector<float> &inputs) {
    vector<float> output;
    output.reserve(1000);
    output = inputs;
    for(auto &layer : layers) {
        const vector<float> tmp = layer.calculate_output(output);
        output.clear();
        for(const auto v : tmp) { output.push_back(v); }
    }
    return vector<float>(output.begin(), output.begin() + 10);
}

void Network::learn(const vector<DataPoint> &data, const float learn_rate) {
    NetworkData network_data{layers};

    for(const auto &point : data) {
        update_gradients(point, network_data);
    }

    for(auto &l : layers) { l.apply_gradients(learn_rate / data.size()); }
}

void Network::update_gradients(const DataPoint &data_point, NetworkData &instance_data) {
    std::vector<float> data_to_next(data_point.input_data.size());
    data_to_next = data_point.input_data;

    for(size_t i = 0; i < layers.size(); i++) {
        data_to_next = layers[i].calculate_output(data_to_next, instance_data.layer_data[i]);
    }

    const size_t output_layer_index = layers.size() - 1;
    auto &output_layer = layers[output_layer_index];
    auto &output_layer_data = instance_data.layer_data[output_layer_index];

    output_layer.calculate_output_node_values(output_layer_data, data_point.expected);
    output_layer.update_gradient_values(output_layer_data);

    for(long i = output_layer_index - 1; i >= 0L; i--) {
        auto &hidden_layer = layers[i];
        auto &hidden_layer_data = instance_data.layer_data[i];

        hidden_layer.calculate_hidden_node_values(hidden_layer_data, layers[i+1], instance_data.layer_data[i+1].node_values);
        hidden_layer.update_gradient_values(hidden_layer_data);
    }
}

LayerData::LayerData(const Layer &layer) {
    weighted_inputs.resize(layer.num_nodes_out, 0.0);
    activations.resize(layer.num_nodes_out, 0.0);
    node_values.resize(layer.num_nodes_out, 0.0);
    inputs.resize(layer.num_nodes_in, 0.0);
}

NetworkData::NetworkData(const vector<Layer> &layers) {
    for(const auto &l : layers) { layer_data.push_back(l); }
}