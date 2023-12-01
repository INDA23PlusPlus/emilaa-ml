#include "mnist.hpp"

#include <algorithm>
#include <fstream>
#include <cstdint>

uint32_t swap_endian(uint32_t val) {
    uint8_t c1, c2, c3, c4;
    c1 =  val        & 255;
    c2 = (val >> 8)  & 255;
    c3 = (val >> 16) & 255;
    c4 = (val >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + (uint32_t)c4;
}

std::vector<DataPoint> read_mnist_n(const char *images_path, const char *labels_path, size_t n) {
    std::ifstream image_file(images_path, std::ios::in | std::ios::binary);
    std::ifstream label_file(labels_path, std::ios::in | std::ios::binary);
    uint32_t magic, items, labels, rows, cols;

    image_file.read((char*)&magic, 4);
    magic = swap_endian(magic);
    if(magic != 2051) { return {{{}, {}, -1}}; }

    label_file.read((char*)&magic, 4);
    magic = swap_endian(magic);
    if(magic != 2049) { return {{{}, {}, -1}}; }

    image_file.read((char*)&items, 4);
    label_file.read((char*)&labels, 4);
    items = swap_endian(items);
    labels = swap_endian(labels);
    if(items != labels) { return {{{}, {}, -2}}; }

    image_file.read((char*)&rows, 4);
    image_file.read((char*)&cols, 4);
    rows = swap_endian(rows);
    cols = swap_endian(cols);
    if(rows != cols) { return {{{}, {}, -3}}; }

    std::vector<DataPoint> pictures{};
    std::vector<uint8_t> pixels(784);
    pictures.reserve(n);
    char label = 0;
    
    for(uint32_t i = 0; i < n; i++) {
        image_file.read((char *)pixels.data(), 784);
        label_file.read(&label, 1);

        std::vector<float> pixels_normalized{};
        std::vector<float> expected(10);

        std::fill(expected.begin(), expected.end(), 0.0);
        pixels_normalized.reserve(784);
        expected[label] = 1.0;
        for(const auto &p : pixels) { pixels_normalized.push_back((float)p); }

        pictures.push_back({pixels_normalized, expected, label});
    }

    return pictures;
}