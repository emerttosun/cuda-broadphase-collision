#include "CudaBruteForceDetector.h"
#include "CollisionMath.cuh"
#include <stdexcept>

// Shared Memory Tiled Brute Force Kernel
__global__ void cuda_brute_force_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ r,
    size_t count,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) 
{
    extern __shared__ float shared_mem[];
    float* sx = shared_mem;
    float* sy = &sx[blockDim.x];
    float* sr = &sy[blockDim.x * 2];

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Register cache for my own elements
    float my_x = 0.0f, my_y = 0.0f, my_r = 0.0f;
    if (idx < count) {
        my_x = x[idx];
        my_y = y[idx];
        my_r = r[idx];
    }

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;

    // Tile over all elements cooperatively
    for (size_t tile = 0; tile < count; tile += blockDim.x) {
        size_t tile_idx = tile + threadIdx.x;
        
        if (tile_idx < count) {
            sx[threadIdx.x] = x[tile_idx];
            sy[threadIdx.x] = y[tile_idx];
            sr[threadIdx.x] = r[tile_idx];
        }
        __syncthreads();

        int num_elements_in_tile = min(static_cast<int>(blockDim.x), static_cast<int>(count - tile));
        
        for (int j = 0; j < num_elements_in_tile; ++j) {
            size_t global_j = tile + j;
            if (idx < global_j && idx < count) { // Ensure pairs are only checked once and valid
                local_candidates++;
                if (circles_collide(my_x, my_y, my_r, sx[j], sy[j], sr[j])) {
                    local_collisions++;
                }
            }
        }
        __syncthreads();
    }

    if (local_candidates > 0) atomicAdd(candidate_pair_count, local_candidates);
    if (local_collisions > 0) atomicAdd(collision_count, local_collisions);
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)


CudaBruteForceDetector::CudaBruteForceDetector() {
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDA_CHECK(cudaEventCreate(&m_start_event));
    CUDA_CHECK(cudaEventCreate(&m_kernel_start_event));
    CUDA_CHECK(cudaEventCreate(&m_kernel_end_event));
    CUDA_CHECK(cudaEventCreate(&m_end_event));

    CUDA_CHECK(cudaMalloc(&d_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_candidate_pair_count, sizeof(unsigned long long)));
}

CudaBruteForceDetector::~CudaBruteForceDetector() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_radius) cudaFree(d_radius);
    
    cudaFree(d_collision_count);
    cudaFree(d_candidate_pair_count);

    cudaStreamDestroy(m_stream);
    cudaEventDestroy(m_start_event);
    cudaEventDestroy(m_kernel_start_event);
    cudaEventDestroy(m_kernel_end_event);
    cudaEventDestroy(m_end_event);
}

void CudaBruteForceDetector::update_data(const CirclesSoA& circles) {
    m_count = circles.count;
    if (m_count > m_capacity) {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_radius) cudaFree(d_radius);

        CUDA_CHECK(cudaMalloc(&d_x, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_radius, m_count * sizeof(float)));
        m_capacity = m_count;
    }

    CUDA_CHECK(cudaEventRecord(m_start_event, m_stream));
    
    if (m_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_x, circles.host_x.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_y, circles.host_y.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_radius, circles.host_radius.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
    }
}

CollisionResult CudaBruteForceDetector::run_detection() {
    CollisionResult result;
    if (m_count == 0) return result;

    CUDA_CHECK(cudaMemsetAsync(d_collision_count, 0, sizeof(unsigned long long), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_candidate_pair_count, 0, sizeof(unsigned long long), m_stream));

    CUDA_CHECK(cudaEventRecord(m_kernel_start_event, m_stream));

    const int threads_per_block = 256;
    const int blocks = (m_count + threads_per_block - 1) / threads_per_block;
    const size_t shared_mem_size = 3 * threads_per_block * sizeof(float);

    cuda_brute_force_tiled_kernel<<<blocks, threads_per_block, shared_mem_size, m_stream>>>(
        d_x, d_y, d_radius, m_count, d_collision_count, d_candidate_pair_count
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(m_kernel_end_event, m_stream));

    CUDA_CHECK(cudaMemcpyAsync(&result.collision_count, d_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(&result.candidate_pair_count, d_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost, m_stream));

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

    return result;
}
