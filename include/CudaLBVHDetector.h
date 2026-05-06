#pragma once

#include <cuda_runtime.h>

#include "ICollisionDetector.h"

class CudaLBVHDetector : public ICollisionDetector {
public:
    CudaLBVHDetector();
    ~CudaLBVHDetector() override;

    void set_scene_bounds(float scene_width, float scene_height);

    void update_data(const CirclesSoA& circles) override;
    CollisionResult run_detection() override;
    std::string get_name() const override { return "CUDA LBVH (Karras 2012)"; }

private:
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_radius = nullptr;
    size_t m_capacity = 0;
    size_t m_count = 0;

    unsigned int* d_morton_codes = nullptr;
    int* d_object_indices = nullptr;
    
    // LBVH Tree arrays (size: 2N - 1)
    float* d_aabb_min_x = nullptr;
    float* d_aabb_min_y = nullptr;
    float* d_aabb_max_x = nullptr;
    float* d_aabb_max_y = nullptr;
    int* d_parent = nullptr;
    int* d_left_child = nullptr;
    int* d_right_child = nullptr;
    int* d_atomic_flags = nullptr;

    unsigned long long* d_collision_count = nullptr;
    unsigned long long* d_candidate_pair_count = nullptr;

    float m_scene_width = 1000.0f;
    float m_scene_height = 1000.0f;

    cudaStream_t m_stream;
    cudaEvent_t m_start_event, m_kernel_start_event, m_kernel_end_event, m_end_event;
};
