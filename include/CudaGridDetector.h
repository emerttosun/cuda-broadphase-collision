#pragma once

#include "ICollisionDetector.h"

class CudaGridDetector : public ICollisionDetector {
public:
    CudaGridDetector();
    ~CudaGridDetector() override;

    void set_grid_params(float scene_width, float scene_height, float cell_size, int dense_cell_threshold);

    void update_data(const CirclesSoA& circles) override;
    CollisionResult run_detection() override;
    std::string get_name() const override;

private:
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_radius = nullptr;
    size_t m_capacity = 0;
    size_t m_count = 0;

    int* d_cell_keys = nullptr;
    int* d_indices = nullptr;
    int* d_cell_start = nullptr;
    int* d_cell_end = nullptr;

    unsigned long long* d_collision_count = nullptr;
    unsigned long long* d_candidate_pair_count = nullptr;

    float m_scene_width = 1000.0f;
    float m_scene_height = 1000.0f;
    float m_cell_size = 20.0f;
    int m_dense_cell_threshold = 128;
    int m_total_cells = 0;

    cudaStream_t m_stream;
    cudaEvent_t m_start_event, m_kernel_start_event, m_kernel_end_event, m_end_event;
};
