#include "CudaGrid.cuh"
#include "CollisionMath.h"
#include "CudaUtils.cuh"
#include "Timer.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ static int clamp_int_device(int value, int min_value, int max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

__global__ static void compute_cell_keys_kernel(
    const Circle* circles,
    int* cell_keys,
    int* indices,
    size_t count,
    float cell_size,
    int grid_width,
    int grid_height) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const int cell_x = clamp_int_device((int)floorf(circles[i].x / cell_size), 0, grid_width - 1);
    const int cell_y = clamp_int_device((int)floorf(circles[i].y / cell_size), 0, grid_height - 1);
    cell_keys[i] = cell_y * grid_width + cell_x;
    indices[i] = (int)i;
}

__global__ static void init_cell_ranges_kernel(int* cell_start, int* cell_end, int total_cells) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_cells) {
        return;
    }
    cell_start[i] = -1;
    cell_end[i] = -1;
}

__global__ static void build_cell_ranges_kernel(
    const int* sorted_cell_keys,
    int* cell_start,
    int* cell_end,
    size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const int current_cell = sorted_cell_keys[i];
    if (i == 0 || sorted_cell_keys[i - 1] != current_cell) {
        cell_start[current_cell] = (int)i;
    }
    if (i == count - 1 || sorted_cell_keys[i + 1] != current_cell) {
        cell_end[current_cell] = (int)i + 1;
    }
}

__global__ static void grid_collision_kernel(
    const Circle* circles,
    const int* sorted_cell_keys,
    const int* sorted_indices,
    const int* cell_start,
    const int* cell_end,
    size_t count,
    int grid_width,
    int grid_height,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const size_t sorted_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_pos >= count) {
        return;
    }

    const int object_index = sorted_indices[sorted_pos];
    const Circle object = circles[object_index];
    const int cell_id = sorted_cell_keys[sorted_pos];
    const int cell_x = cell_id % grid_width;
    const int cell_y = cell_id / grid_width;

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = cell_x + dx;
            const int ny = cell_y + dy;
            if (nx < 0 || ny < 0 || nx >= grid_width || ny >= grid_height) {
                continue;
            }

            const int neighbor_cell = ny * grid_width + nx;
            const int start = cell_start[neighbor_cell];
            const int end = cell_end[neighbor_cell];
            if (start < 0 || end < 0) {
                continue;
            }

            for (int p = start; p < end; ++p) {
                const int other_index = sorted_indices[p];
                if (other_index <= object_index) {
                    continue;
                }

                local_candidates++;
                const Circle other = circles[other_index];
                if (circles_overlap(&object, &other)) {
                    local_collisions++;
                }
            }
        }
    }

    if (local_candidates > 0) {
        atomicAdd(candidate_pair_count, local_candidates);
    }
    if (local_collisions > 0) {
        atomicAdd(collision_count, local_collisions);
    }
}

static GridStats compute_grid_stats_on_host(
    const int* cell_start,
    const int* cell_end,
    int total_cells,
    int dense_cell_threshold) {
    GridStats stats;
    stats.max_objects_in_cell = 0;
    stats.avg_objects_per_non_empty_cell = 0.0;
    stats.dense_cell_count = 0;

    int non_empty_cells = 0;
    unsigned long long total_objects_in_non_empty_cells = 0;

    for (int i = 0; i < total_cells; ++i) {
        if (cell_start[i] < 0 || cell_end[i] < 0) {
            continue;
        }

        const int objects_in_cell = cell_end[i] - cell_start[i];
        if (objects_in_cell <= 0) {
            continue;
        }

        non_empty_cells++;
        total_objects_in_non_empty_cells += (unsigned long long)objects_in_cell;
        if (objects_in_cell > stats.max_objects_in_cell) {
            stats.max_objects_in_cell = objects_in_cell;
        }
        if (objects_in_cell >= dense_cell_threshold) {
            stats.dense_cell_count++;
        }
    }

    if (non_empty_cells > 0) {
        stats.avg_objects_per_non_empty_cell =
            (double)total_objects_in_non_empty_cells / (double)non_empty_cells;
    }

    return stats;
}

extern "C" CollisionResult run_cuda_uniform_grid(const Circle* circles, size_t count, const void* params) {
    CollisionResult result;
    memset(&result, 0, sizeof(result));
    result.has_grid_stats = 1;

    if (params == NULL) {
        fprintf(stderr, "run_cuda_uniform_grid: params must point to CudaGridParams.\n");
        return result;
    }

    const CudaGridParams* p = (const CudaGridParams*)params;
    if (circles == NULL || count == 0 || p->cell_size <= 0.0f) {
        return result;
    }

    if (p->max_radius > 0.0f && p->cell_size < 2.0f * p->max_radius) {
        fprintf(stderr,
                "run_cuda_uniform_grid: cell_size %.3f < 2 * max_radius %.3f; "
                "8-neighbor search may miss collisions.\n",
                (double)p->cell_size,
                (double)p->max_radius);
    }

    const double total_start = timer_now_ms();

    const int grid_width = (int)ceilf(p->scene_width / p->cell_size);
    const int grid_height = (int)ceilf(p->scene_height / p->cell_size);
    const int total_cells = grid_width * grid_height;

    Circle* d_circles = NULL;
    int* d_cell_keys = NULL;
    int* d_indices = NULL;
    int* d_cell_start = NULL;
    int* d_cell_end = NULL;
    unsigned long long* d_collision_count = NULL;
    unsigned long long* d_candidate_pair_count = NULL;
    int* h_cell_start = NULL;
    int* h_cell_end = NULL;
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaMalloc((void**)&d_circles, count * sizeof(Circle)));
    CUDA_CHECK(cudaMalloc((void**)&d_cell_keys, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cell_start, total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cell_end, total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidate_pair_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_circles, circles, count * sizeof(Circle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threads_per_block = 256;
    const int blocks = (int)((count + threads_per_block - 1) / threads_per_block);
    const int cell_blocks = (total_cells + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaEventRecord(start));
    init_cell_ranges_kernel<<<cell_blocks, threads_per_block>>>(d_cell_start, d_cell_end, total_cells);
    CUDA_CHECK(cudaGetLastError());

    compute_cell_keys_kernel<<<blocks, threads_per_block>>>(
        d_circles,
        d_cell_keys,
        d_indices,
        count,
        p->cell_size,
        grid_width,
        grid_height);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> keys_ptr(d_cell_keys);
    thrust::device_ptr<int> indices_ptr(d_indices);
    thrust::sort_by_key(keys_ptr, keys_ptr + count, indices_ptr);

    build_cell_ranges_kernel<<<blocks, threads_per_block>>>(d_cell_keys, d_cell_start, d_cell_end, count);
    CUDA_CHECK(cudaGetLastError());

    grid_collision_kernel<<<blocks, threads_per_block>>>(
        d_circles,
        d_cell_keys,
        d_indices,
        d_cell_start,
        d_cell_end,
        count,
        grid_width,
        grid_height,
        d_collision_count,
        d_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed, start, stop));
    result.kernel_time_ms = (double)kernel_elapsed;
    CUDA_CHECK(cudaMemcpy(&result.collision_count, d_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result.candidate_pair_count, d_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    h_cell_start = (int*)malloc(total_cells * sizeof(int));
    h_cell_end = (int*)malloc(total_cells * sizeof(int));
    if (h_cell_start != NULL && h_cell_end != NULL) {
        CUDA_CHECK(cudaMemcpy(h_cell_start, d_cell_start, total_cells * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cell_end, d_cell_end, total_cells * sizeof(int), cudaMemcpyDeviceToHost));
        result.grid_stats = compute_grid_stats_on_host(h_cell_start, h_cell_end, total_cells, p->dense_cell_threshold);
    }

    free(h_cell_start);
    free(h_cell_end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_circles));
    CUDA_CHECK(cudaFree(d_cell_keys));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_cell_start));
    CUDA_CHECK(cudaFree(d_cell_end));
    CUDA_CHECK(cudaFree(d_collision_count));
    CUDA_CHECK(cudaFree(d_candidate_pair_count));

    result.total_time_ms = timer_now_ms() - total_start;
    return result;
}
