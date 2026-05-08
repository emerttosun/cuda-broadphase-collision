#include "CudaLbvh.cuh"
#include "CollisionMath.h"
#include "CudaUtils.cuh"
#include "Timer.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

__device__ static unsigned int lbvh_part_1_by_1(unsigned int v) {
    v &= 0x0000FFFFu;
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

__device__ static unsigned int lbvh_morton2d(float x, float y, float scene_w, float scene_h) {
    float fx = scene_w > 0.0f ? x / scene_w : 0.0f;
    float fy = scene_h > 0.0f ? y / scene_h : 0.0f;
    fx = fmaxf(0.0f, fminf(fx * 65536.0f, 65535.0f));
    fy = fmaxf(0.0f, fminf(fy * 65536.0f, 65535.0f));
    const unsigned int xx = lbvh_part_1_by_1((unsigned int)fx);
    const unsigned int yy = lbvh_part_1_by_1((unsigned int)fy);
    return (xx << 1) | yy;
}

__device__ static int lbvh_common_upper_bits(const unsigned int* morton_codes, int count, int i, int j) {
    if (j < 0 || j >= count) {
        return -1;
    }

    const unsigned int code_i = morton_codes[i];
    const unsigned int code_j = morton_codes[j];
    if (code_i == code_j) {
        return 32 + __clz((unsigned int)(i ^ j));
    }
    return __clz(code_i ^ code_j);
}

__global__ static void lbvh_morton_kernel(
    const Circle* circles,
    unsigned int* morton_codes,
    int* indices,
    int count,
    float scene_w,
    float scene_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    morton_codes[i] = lbvh_morton2d(circles[i].x, circles[i].y, scene_w, scene_h);
    indices[i] = i;
}

__global__ static void lbvh_build_radix_tree_kernel(
    const unsigned int* morton_codes,
    int count,
    int* parent,
    int* left_child,
    int* right_child) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count - 1) {
        return;
    }

    const int d = (lbvh_common_upper_bits(morton_codes, count, i, i + 1)
                   > lbvh_common_upper_bits(morton_codes, count, i, i - 1)) ? 1 : -1;

    const int min_delta = lbvh_common_upper_bits(morton_codes, count, i, i - d);
    int lmax = 2;
    while (lbvh_common_upper_bits(morton_codes, count, i, i + lmax * d) > min_delta) {
        lmax *= 2;
    }

    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2) {
        if (lbvh_common_upper_bits(morton_codes, count, i, i + (l + t) * d) > min_delta) {
            l += t;
        }
    }
    const int j = i + l * d;

    const int delta_node = lbvh_common_upper_bits(morton_codes, count, i, j);
    int s = 0;
    int t_step = l;
    do {
        t_step = (t_step + 1) / 2;
        if (lbvh_common_upper_bits(morton_codes, count, i, i + (s + t_step) * d) > delta_node) {
            s += t_step;
        }
        if (t_step <= 1) {
            break;
        }
    } while (true);

    const int split = i + s * d + min(d, 0);
    const int left = (min(i, j) == split) ? (count - 1 + split) : split;
    const int right = (max(i, j) == split + 1) ? (count - 1 + split + 1) : (split + 1);

    left_child[i] = left;
    right_child[i] = right;
    parent[left] = i;
    parent[right] = i;
}

__global__ static void lbvh_aabbs_kernel(
    const Circle* circles,
    const int* indices,
    int count,
    const int* parent,
    const int* left_child,
    const int* right_child,
    int* flags,
    float* aabb_min_x,
    float* aabb_min_y,
    float* aabb_max_x,
    float* aabb_max_y) {
    const int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= count) {
        return;
    }

    const int node_idx = count - 1 + leaf_idx;
    const int obj_idx = indices[leaf_idx];
    const Circle c = circles[obj_idx];

    aabb_min_x[node_idx] = c.x - c.radius;
    aabb_min_y[node_idx] = c.y - c.radius;
    aabb_max_x[node_idx] = c.x + c.radius;
    aabb_max_y[node_idx] = c.y + c.radius;
    __threadfence();

    int current = parent[node_idx];
    while (current >= 0) {
        const int old = atomicAdd(&flags[current], 1);
        if (old == 0) {
            return;
        }

        const int l = left_child[current];
        const int r = right_child[current];
        aabb_min_x[current] = fminf(aabb_min_x[l], aabb_min_x[r]);
        aabb_min_y[current] = fminf(aabb_min_y[l], aabb_min_y[r]);
        aabb_max_x[current] = fmaxf(aabb_max_x[l], aabb_max_x[r]);
        aabb_max_y[current] = fmaxf(aabb_max_y[l], aabb_max_y[r]);
        __threadfence();
        current = parent[current];
    }
}

__global__ static void lbvh_traverse_kernel(
    const Circle* circles,
    const int* indices,
    int count,
    const int* left_child,
    const int* right_child,
    const float* aabb_min_x,
    const float* aabb_min_y,
    const float* aabb_max_x,
    const float* aabb_max_y,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= count) {
        return;
    }

    const int obj_idx = indices[leaf_idx];
    const Circle self = circles[obj_idx];
    const float my_min_x = self.x - self.radius;
    const float my_min_y = self.y - self.radius;
    const float my_max_x = self.x + self.radius;
    const float my_max_y = self.y + self.radius;

    int stack[128];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;

    while (stack_ptr > 0) {
        const int node = stack[--stack_ptr];
        if (my_max_x < aabb_min_x[node] || my_min_x > aabb_max_x[node]
            || my_max_y < aabb_min_y[node] || my_min_y > aabb_max_y[node]) {
            continue;
        }

        if (node >= count - 1) {
            const int other_leaf = node - (count - 1);
            if (other_leaf > leaf_idx) {
                ++local_candidates;
                const int other_obj = indices[other_leaf];
                if (circles_overlap(&self, &circles[other_obj])) {
                    ++local_collisions;
                }
            }
        } else if (stack_ptr <= 126) {
            stack[stack_ptr++] = left_child[node];
            stack[stack_ptr++] = right_child[node];
        }
    }

    if (local_candidates > 0) {
        atomicAdd(candidate_pair_count, local_candidates);
    }
    if (local_collisions > 0) {
        atomicAdd(collision_count, local_collisions);
    }
}

extern "C" CollisionResult run_cuda_lbvh(const Circle* circles, size_t count, const void* params) {
    CollisionResult result;
    memset(&result, 0, sizeof(result));

    if (circles == NULL || count == 0) {
        return result;
    }
    if (count > (size_t)2147483647) {
        fprintf(stderr, "run_cuda_lbvh: count is too large for int-indexed LBVH.\n");
        return result;
    }

    float scene_width = 1.0f;
    float scene_height = 1.0f;
    if (params != NULL) {
        const CudaLbvhParams* p = (const CudaLbvhParams*)params;
        scene_width = p->scene_width;
        scene_height = p->scene_height;
    }

    const double total_start = timer_now_ms();
    const int object_count = (int)count;
    const int node_count = object_count * 2 - 1;

    Circle* d_circles = NULL;
    unsigned int* d_morton = NULL;
    int* d_indices = NULL;
    int* d_parent = NULL;
    int* d_left = NULL;
    int* d_right = NULL;
    int* d_flags = NULL;
    float* d_aabb_min_x = NULL;
    float* d_aabb_min_y = NULL;
    float* d_aabb_max_x = NULL;
    float* d_aabb_max_y = NULL;
    unsigned long long* d_collision_count = NULL;
    unsigned long long* d_candidate_pair_count = NULL;
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaMalloc((void**)&d_circles, count * sizeof(Circle)));
    CUDA_CHECK(cudaMalloc((void**)&d_morton, count * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_parent, (size_t)node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_left, (size_t)node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_right, (size_t)node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_flags, (size_t)node_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_aabb_min_x, (size_t)node_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_aabb_min_y, (size_t)node_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_aabb_max_x, (size_t)node_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_aabb_max_y, (size_t)node_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidate_pair_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_circles, circles, count * sizeof(Circle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_flags, 0, (size_t)node_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_parent, 0xFF, (size_t)node_count * sizeof(int)));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threads_per_block = 256;
    const int blocks = (object_count + threads_per_block - 1) / threads_per_block;
    const int internal_blocks = ((object_count - 1) + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaEventRecord(start));
    lbvh_morton_kernel<<<blocks, threads_per_block>>>(
        d_circles,
        d_morton,
        d_indices,
        object_count,
        scene_width,
        scene_height);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<unsigned int> morton_ptr(d_morton);
    thrust::device_ptr<int> indices_ptr(d_indices);
    thrust::sort_by_key(morton_ptr, morton_ptr + object_count, indices_ptr);

    if (internal_blocks > 0) {
        lbvh_build_radix_tree_kernel<<<internal_blocks, threads_per_block>>>(
            d_morton,
            object_count,
            d_parent,
            d_left,
            d_right);
        CUDA_CHECK(cudaGetLastError());
    }

    lbvh_aabbs_kernel<<<blocks, threads_per_block>>>(
        d_circles,
        d_indices,
        object_count,
        d_parent,
        d_left,
        d_right,
        d_flags,
        d_aabb_min_x,
        d_aabb_min_y,
        d_aabb_max_x,
        d_aabb_max_y);
    CUDA_CHECK(cudaGetLastError());

    lbvh_traverse_kernel<<<blocks, threads_per_block>>>(
        d_circles,
        d_indices,
        object_count,
        d_left,
        d_right,
        d_aabb_min_x,
        d_aabb_min_y,
        d_aabb_max_x,
        d_aabb_max_y,
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_circles));
    CUDA_CHECK(cudaFree(d_morton));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_left));
    CUDA_CHECK(cudaFree(d_right));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_aabb_min_x));
    CUDA_CHECK(cudaFree(d_aabb_min_y));
    CUDA_CHECK(cudaFree(d_aabb_max_x));
    CUDA_CHECK(cudaFree(d_aabb_max_y));
    CUDA_CHECK(cudaFree(d_collision_count));
    CUDA_CHECK(cudaFree(d_candidate_pair_count));

    result.total_time_ms = timer_now_ms() - total_start;
    return result;
}
