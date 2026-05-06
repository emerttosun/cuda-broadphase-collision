#include "CudaGridDetector.h"
#include "CollisionMath.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

__device__ static int clamp_int_device(int value, int min_value, int max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

__global__ static void compute_cell_keys_kernel(
    const float* x, const float* y,
    int* cell_keys, int* indices,
    size_t count, float cell_size,
    int grid_width, int grid_height) 
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int cell_x = clamp_int_device(static_cast<int>(floorf(x[i] / cell_size)), 0, grid_width - 1);
    const int cell_y = clamp_int_device(static_cast<int>(floorf(y[i] / cell_size)), 0, grid_height - 1);
    cell_keys[i] = cell_y * grid_width + cell_x;
    indices[i] = static_cast<int>(i);
}

__global__ static void build_cell_ranges_kernel(
    const int* sorted_cell_keys,
    int* cell_start, int* cell_end,
    size_t count) 
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int current_cell = sorted_cell_keys[i];
    if (i == 0 || sorted_cell_keys[i - 1] != current_cell) {
        cell_start[current_cell] = static_cast<int>(i);
    }
    if (i == count - 1 || sorted_cell_keys[i + 1] != current_cell) {
        cell_end[current_cell] = static_cast<int>(i) + 1;
    }
}

__global__ static void grid_collision_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ r,
    const int* __restrict__ sorted_cell_keys,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    size_t count,
    int grid_width, int grid_height,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) 
{
    const size_t sorted_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_pos >= count) return;

    const int object_index = sorted_indices[sorted_pos];
    const float my_x = x[object_index];
    const float my_y = y[object_index];
    const float my_r = r[object_index];
    
    const int cell_id = sorted_cell_keys[sorted_pos];
    const int cell_x = cell_id % grid_width;
    const int cell_y = cell_id / grid_width;

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = cell_x + dx;
            const int ny = cell_y + dy;
            if (nx < 0 || ny < 0 || nx >= grid_width || ny >= grid_height) continue;

            const int neighbor_cell = ny * grid_width + nx;
            const int start = cell_start[neighbor_cell];
            const int end = cell_end[neighbor_cell];
            if (start < 0 || end < 0) continue;

            for (int p = start; p < end; ++p) {
                const int other_index = sorted_indices[p];
                if (other_index <= object_index) continue; // Check pairs only once

                local_candidates++;
                if (circles_collide(my_x, my_y, my_r, x[other_index], y[other_index], r[other_index])) {
                    local_collisions++;
                }
            }
        }
    }

    if (local_candidates > 0) atomicAdd(candidate_pair_count, local_candidates);
    if (local_collisions > 0) atomicAdd(collision_count, local_collisions);
}

CudaGridDetector::CudaGridDetector() {
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDA_CHECK(cudaEventCreate(&m_start_event));
    CUDA_CHECK(cudaEventCreate(&m_kernel_start_event));
    CUDA_CHECK(cudaEventCreate(&m_kernel_end_event));
    CUDA_CHECK(cudaEventCreate(&m_end_event));

    CUDA_CHECK(cudaMalloc(&d_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_candidate_pair_count, sizeof(unsigned long long)));
}

CudaGridDetector::~CudaGridDetector() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_radius) cudaFree(d_radius);
    if (d_cell_keys) cudaFree(d_cell_keys);
    if (d_indices) cudaFree(d_indices);
    if (d_cell_start) cudaFree(d_cell_start);
    if (d_cell_end) cudaFree(d_cell_end);

    cudaFree(d_collision_count);
    cudaFree(d_candidate_pair_count);

    cudaStreamDestroy(m_stream);
    cudaEventDestroy(m_start_event);
    cudaEventDestroy(m_kernel_start_event);
    cudaEventDestroy(m_kernel_end_event);
    cudaEventDestroy(m_end_event);
}

void CudaGridDetector::set_grid_params(float scene_width, float scene_height, float cell_size, int dense_cell_threshold) {
    m_scene_width = scene_width;
    m_scene_height = scene_height;
    m_cell_size = cell_size;
    m_dense_cell_threshold = dense_cell_threshold;
    
    int grid_width = static_cast<int>(ceilf(m_scene_width / m_cell_size));
    int grid_height = static_cast<int>(ceilf(m_scene_height / m_cell_size));
    int new_total_cells = grid_width * grid_height;

    if (new_total_cells > m_total_cells) {
        if (d_cell_start) cudaFree(d_cell_start);
        if (d_cell_end) cudaFree(d_cell_end);
        CUDA_CHECK(cudaMalloc(&d_cell_start, new_total_cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cell_end, new_total_cells * sizeof(int)));
        m_total_cells = new_total_cells;
    }
}

void CudaGridDetector::update_data(const CirclesSoA& circles) {
    m_count = circles.count;
    if (m_count > m_capacity) {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_radius) cudaFree(d_radius);
        if (d_cell_keys) cudaFree(d_cell_keys);
        if (d_indices) cudaFree(d_indices);

        CUDA_CHECK(cudaMalloc(&d_x, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_radius, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cell_keys, m_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_indices, m_count * sizeof(int)));
        m_capacity = m_count;
    }

    CUDA_CHECK(cudaEventRecord(m_start_event, m_stream));

    if (m_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_x, circles.host_x.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_y, circles.host_y.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_radius, circles.host_radius.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
    }
}

std::string CudaGridDetector::get_name() const {
    char buf[64];
    snprintf(buf, sizeof(buf), "CUDA Grid (Cell: %.1f)", m_cell_size);
    return std::string(buf);
}

CollisionResult CudaGridDetector::run_detection() {
    CollisionResult result;
    if (m_count == 0 || m_cell_size <= 0.0f || m_total_cells <= 0) return result;

    const int grid_width = static_cast<int>(ceilf(m_scene_width / m_cell_size));
    const int grid_height = static_cast<int>(ceilf(m_scene_height / m_cell_size));
    const int current_total_cells = grid_width * grid_height;

    CUDA_CHECK(cudaMemsetAsync(d_cell_start, 0xFF, current_total_cells * sizeof(int), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_cell_end, 0xFF, current_total_cells * sizeof(int), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_collision_count, 0, sizeof(unsigned long long), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_candidate_pair_count, 0, sizeof(unsigned long long), m_stream));

    CUDA_CHECK(cudaEventRecord(m_kernel_start_event, m_stream));

    const int threads_per_block = 256;
    const int blocks = (m_count + threads_per_block - 1) / threads_per_block;

    // 1. Compute Cell Keys
    compute_cell_keys_kernel<<<blocks, threads_per_block, 0, m_stream>>>(
        d_x, d_y, d_cell_keys, d_indices, m_count, m_cell_size, grid_width, grid_height);

    // 2. Sort by Cell Keys using Thrust
    thrust::device_ptr<int> keys_ptr(d_cell_keys);
    thrust::device_ptr<int> indices_ptr(d_indices);
    thrust::sort_by_key(thrust::cuda::par.on(m_stream), keys_ptr, keys_ptr + m_count, indices_ptr);

    // 3. Build Cell Ranges
    build_cell_ranges_kernel<<<blocks, threads_per_block, 0, m_stream>>>(d_cell_keys, d_cell_start, d_cell_end, m_count);

    // 4. Grid Collision
    grid_collision_kernel<<<blocks, threads_per_block, 0, m_stream>>>(
        d_x, d_y, d_radius, d_cell_keys, d_indices, d_cell_start, d_cell_end, 
        m_count, grid_width, grid_height, d_collision_count, d_candidate_pair_count);
    
    CUDA_CHECK(cudaEventRecord(m_kernel_end_event, m_stream));

    CUDA_CHECK(cudaMemcpyAsync(&result.collision_count, d_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(&result.candidate_pair_count, d_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost, m_stream));

    // Optional: compute grid stats by downloading cell_start and cell_end
    std::vector<int> h_cell_start(current_total_cells);
    std::vector<int> h_cell_end(current_total_cells);
    CUDA_CHECK(cudaMemcpyAsync(h_cell_start.data(), d_cell_start, current_total_cells * sizeof(int), cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_cell_end.data(), d_cell_end, current_total_cells * sizeof(int), cudaMemcpyDeviceToHost, m_stream));

    CUDA_CHECK(cudaEventRecord(m_end_event, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    float transfer_time_ms = 0.0f;
    float kernel_time_ms = 0.0f;
    float total_time_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, m_start_event, m_end_event));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, m_kernel_start_event, m_kernel_end_event));
    
    transfer_time_ms = total_time_ms - kernel_time_ms;

    result.memory_transfer_time_ms = static_cast<double>(transfer_time_ms);
    result.kernel_execution_time_ms = static_cast<double>(kernel_time_ms);
    result.total_time_ms = static_cast<double>(total_time_ms);

    // Compute Stats
    int max_obj = 0;
    int dense = 0;
    int non_empty = 0;
    unsigned long long total_obj = 0;
    for (int i = 0; i < current_total_cells; ++i) {
        if (h_cell_start[i] >= 0 && h_cell_end[i] >= 0) {
            int objs = h_cell_end[i] - h_cell_start[i];
            if (objs > 0) {
                non_empty++;
                total_obj += objs;
                if (objs > max_obj) max_obj = objs;
                if (objs >= m_dense_cell_threshold) dense++;
            }
        }
    }
    result.extra_stats["max_objects_in_cell"] = max_obj;
    result.extra_stats["avg_objects_per_non_empty_cell"] = non_empty > 0 ? static_cast<double>(total_obj) / non_empty : 0.0;
    result.extra_stats["dense_cell_count"] = dense;

    return result;
}
