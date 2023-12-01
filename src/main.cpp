#include "MNIST/mnist.hpp"
#include "Network/network.hpp"

#include <random>

int main() {
    const auto images = read_mnist_n("./data/images", "./data/labels", 60000UL);
    std::vector<DataPoint> training_images, test_images;
    for(size_t i = 0; i < images.size(); i++) {
        if(i < 59000UL) { training_images.push_back(images[i]); }
        else { test_images.push_back(images[i]); }
    }

    Network network{{784, 10}};

    size_t b = 0;
    for(;;) {
        const auto batch = std::vector<DataPoint>(training_images.begin() + b * 32, training_images.begin() + b * 32 + 64);

        network.learn(batch, 0.01);
        b = (++b) % 1840;
    }

    return 0;
}