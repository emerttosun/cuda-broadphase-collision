#include "CudaLBVHDetector.h"
#include "CollisionMath.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)

__device__ inline unsigned int expand_bits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ inline unsigned int morton2D(float x, float y, float min_x, float min_y, float max_x, float max_y) {
    x = (x - min_x) / (max_x - min_x);
    y = (y - min_y) / (max_y - min_y);
    x = fmaxf(0.0f, fminf(x * 32768.0f, 32767.0f));
    y = fmaxf(0.0f, fminf(y * 32768.0f, 32767.0f));
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    return xx * 2 + yy;
}

__global__ void compute_morton_codes_kernel(
    const float* x, const float* y,
    unsigned int* morton_codes, int* object_indices,
    size_t count, float scene_width, float scene_height) 
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    
    morton_codes[i] = morton2D(x[i], y[i], 0.0f, 0.0f, scene_width, scene_height);
    object_indices[i] = i;
}

__device__ int common_upper_bits(const unsigned int* morton_codes, int count, int i, int j) {
    if (j < 0 || j >= count) return -1;
    unsigned int code_i = morton_codes[i];
    unsigned int code_j = morton_codes[j];
    if (code_i == code_j) {
        // Tie-breaker using index
        return 32 + __clz(i ^ j);
    }
    return __clz(code_i ^ code_j);
}

__global__ void build_radix_tree_kernel(
    const unsigned int* morton_codes,
    int count,
    int* parent,
    int* left_child,
    int* right_child)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count - 1) return;

    // Determine direction of the range (+1 or -1)
    int d = (common_upper_bits(morton_codes, count, i, i + 1) > common_upper_bits(morton_codes, count, i, i - 1)) ? 1 : -1;
    
    // Compute upper bound for the length of the range
    int min_delta = common_upper_bits(morton_codes, count, i, i - d);
    int lmax = 2;
    while (common_upper_bits(morton_codes, count, i, i + lmax * d) > min_delta) {
        lmax *= 2;
    }
    
    // Find the other end using binary search
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2) {
        if (common_upper_bits(morton_codes, count, i, i + (l + t) * d) > min_delta) {
            l += t;
        }
    }
    int j = i + l * d;
    
    // Find the split position using binary search
    int delta_node = common_upper_bits(morton_codes, count, i, j);
    int s = 0;
    float div = 2.0f;
    int t = l;
    do {
        t = (int)ceilf((float)t / div);
        if (common_upper_bits(morton_codes, count, i, i + (s + t) * d) > delta_node) {
            s += t;
        }
        if (t <= 1) break;
    } while (true);
    
    int split = i + s * d + min(d, 0);

    // Output child pointers
    int left = (min(i, j) == split) ? (count - 1 + split) : split;
    int right = (max(i, j) == split + 1) ? (count - 1 + split + 1) : (split + 1);

    left_child[i] = left;
    right_child[i] = right;
    parent[left] = i;
    parent[right] = i;
}

__global__ void compute_aabbs_bottom_up_kernel(
    const float* x, const float* y, const float* r,
    const int* object_indices,
    int count,
    int* parent,
    int* left_child,
    int* right_child,
    int* atomic_flags,
    float* aabb_min_x, float* aabb_min_y,
    float* aabb_max_x, float* aabb_max_y)
{
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= count) return;

    int node_idx = count - 1 + leaf_idx;
    int obj_idx = object_indices[leaf_idx];

    // Initialize leaf AABB
    float cx = x[obj_idx];
    float cy = y[obj_idx];
    float rad = r[obj_idx];
    
    aabb_min_x[node_idx] = cx - rad;
    aabb_min_y[node_idx] = cy - rad;
    aabb_max_x[node_idx] = cx + rad;
    aabb_max_y[node_idx] = cy + rad;

    int current = parent[node_idx];
    while (current >= 0) {
        int old = atomicAdd(&atomic_flags[current], 1);
        if (old == 0) {
            // First thread to reach this node. Stop here.
            return;
        }

        // Second thread reaching this node. Compute AABB from children.
        int left = left_child[current];
        int right = right_child[current];

        aabb_min_x[current] = fminf(aabb_min_x[left], aabb_min_x[right]);
        aabb_min_y[current] = fminf(aabb_min_y[left], aabb_min_y[right]);
        aabb_max_x[current] = fmaxf(aabb_max_x[left], aabb_max_x[right]);
        aabb_max_y[current] = fmaxf(aabb_max_y[left], aabb_max_y[right]);

        current = parent[current];
    }
}

__global__ void traverse_lbvh_kernel(
    const float* x, const float* y, const float* r,
    int count,
    const int* object_indices,
    const int* left_child, const int* right_child,
    const float* aabb_min_x, const float* aabb_min_y,
    const float* aabb_max_x, const float* aabb_max_y,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count)
{
    int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= count) return;

    int obj_idx = object_indices[leaf_idx];
    float my_x = x[obj_idx];
    float my_y = y[obj_idx];
    float my_r = r[obj_idx];
    
    float my_min_x = my_x - my_r;
    float my_min_y = my_y - my_r;
    float my_max_x = my_x + my_r;
    float my_max_y = my_y + my_r;

    // Stack for BVH traversal
    int stack[64];
    int stack_ptr = 0;
    
    // Push root
    stack[stack_ptr++] = 0;

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;

    while (stack_ptr > 0) {
        int node = stack[--stack_ptr];
        
        // AABB check
        if (my_max_x < aabb_min_x[node] || my_min_x > aabb_max_x[node] ||
            my_max_y < aabb_min_y[node] || my_min_y > aabb_max_y[node]) {
            continue;
        }

        if (node >= count - 1) { // is leaf
            int other_leaf_idx = node - (count - 1);
            if (other_leaf_idx > leaf_idx) { // only count pairs once
                local_candidates++;
                int other_obj_idx = object_indices[other_leaf_idx];
                if (circles_collide(my_x, my_y, my_r, x[other_obj_idx], y[other_obj_idx], r[other_obj_idx])) {
                    local_collisions++;
                }
            }
        } else {
            // Push children
            stack[stack_ptr++] = left_child[node];
            stack[stack_ptr++] = right_child[node];
        }
    }

    if (local_candidates > 0) atomicAdd(candidate_pair_count, local_candidates);
    if (local_collisions > 0) atomicAdd(collision_count, local_collisions);
}

CudaLBVHDetector::CudaLBVHDetector() {
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    CUDA_CHECK(cudaEventCreate(&m_start_event));
    CUDA_CHECK(cudaEventCreate(&m_kernel_start_event));
    CUDA_CHECK(cudaEventCreate(&m_kernel_end_event));
    CUDA_CHECK(cudaEventCreate(&m_end_event));

    CUDA_CHECK(cudaMalloc(&d_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_candidate_pair_count, sizeof(unsigned long long)));
}

CudaLBVHDetector::~CudaLBVHDetector() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_radius) cudaFree(d_radius);
    if (d_morton_codes) cudaFree(d_morton_codes);
    if (d_object_indices) cudaFree(d_object_indices);
    if (d_aabb_min_x) cudaFree(d_aabb_min_x);
    if (d_aabb_min_y) cudaFree(d_aabb_min_y);
    if (d_aabb_max_x) cudaFree(d_aabb_max_x);
    if (d_aabb_max_y) cudaFree(d_aabb_max_y);
    if (d_parent) cudaFree(d_parent);
    if (d_left_child) cudaFree(d_left_child);
    if (d_right_child) cudaFree(d_right_child);
    if (d_atomic_flags) cudaFree(d_atomic_flags);

    cudaFree(d_collision_count);
    cudaFree(d_candidate_pair_count);

    cudaStreamDestroy(m_stream);
    cudaEventDestroy(m_start_event);
    cudaEventDestroy(m_kernel_start_event);
    cudaEventDestroy(m_kernel_end_event);
    cudaEventDestroy(m_end_event);
}

void CudaLBVHDetector::set_scene_bounds(float scene_width, float scene_height) {
    m_scene_width = scene_width;
    m_scene_height = scene_height;
}

void CudaLBVHDetector::update_data(const CirclesSoA& circles) {
    m_count = circles.count;
    if (m_count > m_capacity) {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_radius) cudaFree(d_radius);
        if (d_morton_codes) cudaFree(d_morton_codes);
        if (d_object_indices) cudaFree(d_object_indices);
        if (d_aabb_min_x) cudaFree(d_aabb_min_x);
        if (d_aabb_min_y) cudaFree(d_aabb_min_y);
        if (d_aabb_max_x) cudaFree(d_aabb_max_x);
        if (d_aabb_max_y) cudaFree(d_aabb_max_y);
        if (d_parent) cudaFree(d_parent);
        if (d_left_child) cudaFree(d_left_child);
        if (d_right_child) cudaFree(d_right_child);
        if (d_atomic_flags) cudaFree(d_atomic_flags);

        size_t num_nodes = 2 * m_count - 1;

        CUDA_CHECK(cudaMalloc(&d_x, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_radius, m_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_morton_codes, m_count * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_object_indices, m_count * sizeof(int)));
        
        CUDA_CHECK(cudaMalloc(&d_aabb_min_x, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_aabb_min_y, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_aabb_max_x, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_aabb_max_y, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_parent, num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_left_child, num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_right_child, num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_atomic_flags, num_nodes * sizeof(int)));
        
        m_capacity = m_count;
    }

    CUDA_CHECK(cudaEventRecord(m_start_event, m_stream));

    if (m_count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_x, circles.host_x.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_y, circles.host_y.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_radius, circles.host_radius.data(), m_count * sizeof(float), cudaMemcpyHostToDevice, m_stream));
    }
}

CollisionResult CudaLBVHDetector::run_detection() {
    CollisionResult result;
    if (m_count <= 1) return result;

    CUDA_CHECK(cudaMemsetAsync(d_collision_count, 0, sizeof(unsigned long long), m_stream));
    CUDA_CHECK(cudaMemsetAsync(d_candidate_pair_count, 0, sizeof(unsigned long long), m_stream));
    
    // Reset atomic flags for AABB generation
    size_t num_nodes = 2 * m_count - 1;
    CUDA_CHECK(cudaMemsetAsync(d_atomic_flags, 0, num_nodes * sizeof(int), m_stream));

    CUDA_CHECK(cudaEventRecord(m_kernel_start_event, m_stream));

    const int threads_per_block = 256;
    int blocks = (m_count + threads_per_block - 1) / threads_per_block;

    // 1. Compute Morton Codes
    compute_morton_codes_kernel<<<blocks, threads_per_block, 0, m_stream>>>(
        d_x, d_y, d_morton_codes, d_object_indices, m_count, m_scene_width, m_scene_height);

    // 2. Radix Sort using Thrust
    thrust::device_ptr<unsigned int> morton_ptr(d_morton_codes);
    thrust::device_ptr<int> indices_ptr(d_object_indices);
    thrust::sort_by_key(thrust::cuda::par.on(m_stream), morton_ptr, morton_ptr + m_count, indices_ptr);

    // 3. Build Radix Tree Hierarchy
    int internal_nodes = m_count - 1;
    int internal_blocks = (internal_nodes + threads_per_block - 1) / threads_per_block;
    build_radix_tree_kernel<<<internal_blocks, threads_per_block, 0, m_stream>>>(
        d_morton_codes, m_count, d_parent, d_left_child, d_right_child);

    // 4. Compute Bounding Boxes (Bottom-up)
    compute_aabbs_bottom_up_kernel<<<blocks, threads_per_block, 0, m_stream>>>(
        d_x, d_y, d_radius, d_object_indices, m_count,
        d_parent, d_left_child, d_right_child, d_atomic_flags,
        d_aabb_min_x, d_aabb_min_y, d_aabb_max_x, d_aabb_max_y);

    // 5. Parallel AABB Traversal
    traverse_lbvh_kernel<<<blocks, threads_per_block, 0, m_stream>>>(
        d_x, d_y, d_radius, m_count, d_object_indices,
        d_left_child, d_right_child,
        d_aabb_min_x, d_aabb_min_y, d_aabb_max_x, d_aabb_max_y,
        d_collision_count, d_candidate_pair_count);

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
