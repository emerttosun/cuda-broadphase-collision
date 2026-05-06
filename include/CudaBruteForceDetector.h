#pragma once

#include <cuda_runtime.h>

#include "ICollisionDetector.h"

class CudaBruteForceDetector : public ICollisionDetector {
public:
    CudaBruteForceDetector();
    ~CudaBruteForceDetector() override;

    void update_data(const CirclesSoA& circles) override;
    CollisionResult run_detection() override;
    std::string get_name() const override { return "CUDA Brute Force (Tiled)"; }

private:
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_radius = nullptr;
    size_t m_capacity = 0;
    size_t m_count = 0;

    unsigned long long* d_collision_count = nullptr;
    unsigned long long* d_candidate_pair_count = nullptr;

    cudaStream_t m_stream;
    cudaEvent_t m_start_event, m_kernel_start_event, m_kernel_end_event, m_end_event;
};
