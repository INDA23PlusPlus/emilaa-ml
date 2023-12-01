#include "layer.hpp"

#include <random>
#include <cassert>
#include <algorithm>

using namespace std;

float absf32(float x) { return x < 0.0 ? -x : x; }

Layer::Layer(const size_t nodes_in, const size_t nodes_out) : num_nodes_in{nodes_in}, num_nodes_out{nodes_out} {
    weights.reserve(num_nodes_in * num_nodes_out);
    grads_w.resize(num_nodes_in * num_nodes_out, 0.0);
    momentum_w.resize(num_nodes_in * num_nodes_out, 0.0);
    biases.resize(num_nodes_out, 0.0);
    grads_b.resize(num_nodes_out, 0.0);
    momentum_b.resize(num_nodes_out, 0.0);

    init_random_weights();
}

void Layer::init_random_weights() {
    random_device rng;
    uniform_real_distribution<float> dist(-1.0, 1.0);
    mt19937 engine(rng());
    const size_t cap = weights.capacity();
    for(size_t i = 0; i < cap; i++) {
        weights.push_back(dist(engine));
    }
}

vector<float> Layer::calculate_output(const vector<float> &input, LayerData &layer_data) {
    layer_data.inputs = input;
    for(size_t i = 0; i < num_nodes_out; i++) {
        auto weighted = biases[i];
        for(size_t j = 0; j < num_nodes_in; j++) {
            weighted += input[j] * weights[i * num_nodes_in + j];
        }
        layer_data.weighted_inputs[i] = weighted;
    }

    for(size_t i = 0; i < layer_data.activations.size(); i++) {
        layer_data.activations[i] = activation(layer_data.weighted_inputs[i]);
    }

    return layer_data.activations;
}

vector<float> Layer::calculate_output(const vector<float> &input) {
    vector<float> outputs;
    for(size_t i = 0; i < num_nodes_out; i++) {
        auto weighted = biases[i];
        for(size_t j = 0; j < num_nodes_in; j++) {
            weighted += input[j] * weights[i * num_nodes_in + j];
        }
        outputs.push_back(activation(weighted));
    }
    return outputs;
}

void Layer::update_gradient_values(LayerData &layer_data) {
    for(size_t i = 0; i < num_nodes_out; i++) {
        auto node_value = layer_data.node_values[i];
        grads_b[i] += node_value;

        for(size_t j = 0; j < num_nodes_in; j++) {
            auto dcdw = layer_data.inputs[j] * node_value;
            grads_w[i * num_nodes_in + j] += dcdw;
        }
    }
}

void Layer::calculate_output_node_values(LayerData &layer_data, const vector<float> &expected) {
    for(size_t i = 0; i < layer_data.node_values.size(); i++) {
        auto c_prime = mean_squared_prime(layer_data.node_values[i], expected[i]);
        auto a_prime = activation_prime(layer_data.weighted_inputs[i]);
        layer_data.node_values[i] = a_prime * c_prime;
    }
}

void Layer::calculate_hidden_node_values(
    LayerData &layer_data,
    const Layer &previous_layer,
    const vector<float> &previous_values) {
    assert(num_nodes_out == previous_layer.num_nodes_in);
    assert(previous_values.size() == previous_layer.num_nodes_out);

    for(size_t i = 0; i < num_nodes_out; i++) {
        float new_node_value = 0.0;
        for(size_t j = 0; j < previous_values.size(); j++) {
            auto weighted_input_derivative = previous_layer.weights[j * previous_layer.num_nodes_in + i];
            new_node_value += weighted_input_derivative * previous_values[j];
        }
        new_node_value *= activation_prime(layer_data.weighted_inputs[i]);
        layer_data.node_values[i] = new_node_value;
    }
}

void Layer::apply_gradients(const float learn_rate) {
    const float decay = 1.0 - 0.4 * learn_rate;

    for(size_t i = 0; i < weights.size(); i++) {
        float w = weights[i];
        float momentum = momentum_w[i] * 0.8 - grads_w[i] * learn_rate;
        momentum_w[i] = momentum;
        weights[i] = w * decay + momentum;
        grads_w[i] = 0.0;
    }

    for(size_t i = 0; i < biases.size(); i++) {
        float momentum = momentum_b[i] * 0.8 - grads_b[i] * learn_rate;
        momentum_b[i] = momentum;
        biases[i] += momentum;
        grads_b[i] = 0.0;
    }
}

float Layer::activation(float val) {
    return 1.0 / (1.0 + expf32(-val));
}

float Layer::activation_prime(float val) {
    float a = activation(val);
    return a * (1.0 - a);
}

float Layer::mean_squared_prime(const float predicted, const float expected) {
    return predicted - expected;
}
