#pragma once

#include <vector>

class Layer;
struct LayerData;
struct NetworkData;

struct LayerData {
    LayerData(const Layer &layer);

    std::vector<float> inputs, weighted_inputs, activations, node_values;
};

struct NetworkData {
    NetworkData(const std::vector<Layer> &layers);

    std::vector<LayerData> layer_data;  
};

class Layer {
    public:
        Layer(const size_t nodes_in, const size_t nodes_out);
        std::vector<float> calculate_output(const std::vector<float> &input,
                                             LayerData &layer_data);
        std::vector<float> calculate_output(const std::vector<float> &input);
        void update_gradient_values(LayerData &layer_data);
        void apply_gradients(const float learn_rate);

        void calculate_output_node_values(LayerData &layer_data,
                                          const std::vector<float> &expected);
        void calculate_hidden_node_values(LayerData &layer_data,
                                          const Layer &previous_layer,
                                          const std::vector<float> &previous_values);

        std::vector<float> weights, biases, grads_w, grads_b, momentum_w, momentum_b;
        const size_t num_nodes_in, num_nodes_out;
        // num_nodes_in  = connections
        // num_nodes_out = neurons
    private:
        float activation(float val);
        float activation_prime(float val);
        float mean_squared_prime(const float predicted, const float expected);

        void init_random_weights();
};