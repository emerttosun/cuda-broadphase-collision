#include "CudaBruteForce.cuh"
#include "CollisionMath.h"
#include "CudaUtils.cuh"
#include "Timer.h"

#include <cuda_runtime.h>

#include <string.h>

__global__ static void cuda_brute_force_kernel(
    const Circle* circles,
    size_t count,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;
    const Circle a = circles[i];

    for (size_t j = i + 1; j < count; ++j) {
        local_candidates++;
        const Circle b = circles[j];
        if (circles_overlap(&a, &b)) {
            local_collisions++;
        }
    }

    if (local_candidates > 0) {
        atomicAdd(candidate_pair_count, local_candidates);
    }
    if (local_collisions > 0) {
        atomicAdd(collision_count, local_collisions);
    }
}

extern "C" CollisionResult run_cuda_brute_force(const Circle* circles, size_t count, const void* params) {
    (void)params;

    CollisionResult result;
    memset(&result, 0, sizeof(result));

    if (circles == NULL || count == 0) {
        return result;
    }

    const double total_start = timer_now_ms();

    Circle* d_circles = NULL;
    unsigned long long* d_collision_count = NULL;
    unsigned long long* d_candidate_pair_count = NULL;
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaMalloc((void**)&d_circles, count * sizeof(Circle)));
    CUDA_CHECK(cudaMalloc((void**)&d_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidate_pair_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_circles, circles, count * sizeof(Circle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threads_per_block = 256;
    const int blocks = (int)((count + threads_per_block - 1) / threads_per_block);

    CUDA_CHECK(cudaEventRecord(start));
    cuda_brute_force_kernel<<<blocks, threads_per_block>>>(
        d_circles,
        count,
        d_collision_count,
        d_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed, start, stop));
    CUDA_CHECK(cudaMemcpy(&result.collision_count, d_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result.candidate_pair_count, d_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    result.kernel_time_ms = (double)kernel_elapsed;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_circles));
    CUDA_CHECK(cudaFree(d_collision_count));
    CUDA_CHECK(cudaFree(d_candidate_pair_count));

    result.total_time_ms = timer_now_ms() - total_start;
    return result;
}
