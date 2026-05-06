#pragma once

#include <vector>
#include <cstddef>

// Structure of Arrays (SoA) layout to guarantee memory coalescing
struct CirclesSoA {
    // Host memory (managed by std::vector for safety and ease of use in C++)
    std::vector<float> host_x;
    std::vector<float> host_y;
    std::vector<float> host_radius;

    // Device memory pointers (managed by the CUDA detector classes)
    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_radius = nullptr;

    size_t count = 0;

    void resize(size_t new_count) {
        count = new_count;
        host_x.resize(new_count);
        host_y.resize(new_count);
        host_radius.resize(new_count);
    }
};
