#include "RaylibInteropTypes.cuh"
#include "Circle.h"
#include "CollisionMath.h"
#include "CudaUtils.cuh"
#include "Rng.h"
#include "Timer.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <GL/gl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct DeviceBall {
    float x;
    float y;
    float vx;
    float vy;
    float radius;
    int colliding;
} DeviceBall;

#define VISUALIZER_MAX_CLUSTERS 4
#define VISUALIZER_BALL_MIN_RADIUS 3.0f
#define VISUALIZER_BALL_MAX_RADIUS 7.0f
#define VISUALIZER_GRID_CELL_SIZE 16.0f
#define VISUALIZER_INIT_SEED 202405u

static DeviceBall* g_balls = NULL;
static DeviceBall* g_host_balls = NULL;
static unsigned long long* g_collision_count = NULL;
static unsigned long long* g_candidate_pair_count = NULL;
static int* g_cell_keys = NULL;
static int* g_indices = NULL;
static int* g_cell_start = NULL;
static int* g_cell_end = NULL;
static unsigned int* g_lbvh_morton = NULL;
static int* g_lbvh_indices = NULL;
static float* g_lbvh_aabb_min_x = NULL;
static float* g_lbvh_aabb_min_y = NULL;
static float* g_lbvh_aabb_max_x = NULL;
static float* g_lbvh_aabb_max_y = NULL;
static int* g_lbvh_parent = NULL;
static int* g_lbvh_left = NULL;
static int* g_lbvh_right = NULL;
static int* g_lbvh_flags = NULL;
static cudaGraphicsResource* g_vbo_resource = NULL;
static cudaEvent_t g_start_event = NULL;
static cudaEvent_t g_stop_event = NULL;
static int g_object_count = 0;
static int g_width = 1280;
static int g_height = 720;
static int g_grid_width = 0;
static int g_grid_height = 0;
static int g_total_cells = 0;
static int g_mode = VISUALIZER_MODE_CUDA_BRUTE_FORCE;

__global__ static void integrate_kernel(DeviceBall* balls, int count, float dt, int width, int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    DeviceBall b = balls[i];
    b.x += b.vx * dt;
    b.y += b.vy * dt;

    if (b.x < b.radius) {
        b.x = b.radius;
        b.vx = fabsf(b.vx);
    } else if (b.x > width - b.radius) {
        b.x = width - b.radius;
        b.vx = -fabsf(b.vx);
    }

    if (b.y < b.radius) {
        b.y = b.radius;
        b.vy = fabsf(b.vy);
    } else if (b.y > height - b.radius) {
        b.y = height - b.radius;
        b.vy = -fabsf(b.vy);
    }

    b.colliding = 0;
    balls[i] = b;
}

__device__ static Circle ball_to_circle(const DeviceBall* b) {
    Circle c;
    c.x = b->x;
    c.y = b->y;
    c.radius = b->radius;
    return c;
}

__global__ static void brute_force_collision_kernel(
    DeviceBall* balls,
    int count,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    unsigned long long local_candidates = 0;
    unsigned long long local_collisions = 0;
    const DeviceBall a = balls[i];
    const Circle a_circle = ball_to_circle(&a);

    for (int j = i + 1; j < count; ++j) {
        const DeviceBall b = balls[j];
        const Circle b_circle = ball_to_circle(&b);
        local_candidates++;

        if (circles_overlap(&a_circle, &b_circle)) {
            balls[i].colliding = 1;
            balls[j].colliding = 1;
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

__device__ static int clamp_int_device(int value, int lo, int hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

__global__ static void compute_cell_keys_kernel(
    const DeviceBall* balls,
    int* cell_keys,
    int* indices,
    int count,
    float cell_size,
    int grid_width,
    int grid_height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int cell_x = clamp_int_device((int)floorf(balls[i].x / cell_size), 0, grid_width - 1);
    const int cell_y = clamp_int_device((int)floorf(balls[i].y / cell_size), 0, grid_height - 1);
    cell_keys[i] = cell_y * grid_width + cell_x;
    indices[i] = i;
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
    int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int current_cell = sorted_cell_keys[i];
    if (i == 0 || sorted_cell_keys[i - 1] != current_cell) {
        cell_start[current_cell] = i;
    }
    if (i == count - 1 || sorted_cell_keys[i + 1] != current_cell) {
        cell_end[current_cell] = i + 1;
    }
}

__global__ static void grid_collision_kernel(
    DeviceBall* balls,
    const int* sorted_cell_keys,
    const int* sorted_indices,
    const int* cell_start,
    const int* cell_end,
    int count,
    int grid_width,
    int grid_height,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const int sorted_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_pos >= count) {
        return;
    }

    const int object_index = sorted_indices[sorted_pos];
    const DeviceBall self = balls[object_index];
    const Circle self_circle = ball_to_circle(&self);
    const int cell_id = sorted_cell_keys[sorted_pos];
    const int cell_x = cell_id % grid_width;
    const int cell_y = cell_id / grid_width;

    unsigned long long local_candidates = 0;
    unsigned long long local_collisions = 0;

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
                const DeviceBall other = balls[other_index];
                const Circle other_circle = ball_to_circle(&other);
                if (circles_overlap(&self_circle, &other_circle)) {
                    balls[object_index].colliding = 1;
                    balls[other_index].colliding = 1;
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

// ---------------------------------------------------------------------------
// LBVH (Karras 2012) — Morton codes + parallel radix tree + AABB traversal.
// Tree layout: N leaves at indices [N-1 .. 2N-2], N-1 internal nodes at
// [0 .. N-2], root = node 0. Sized arrays therefore have 2N-1 entries.
// ---------------------------------------------------------------------------

__device__ static unsigned int lbvh_expand_bits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ static unsigned int lbvh_morton2d(float x, float y, float scene_w, float scene_h) {
    float fx = x / scene_w;
    float fy = y / scene_h;
    fx = fmaxf(0.0f, fminf(fx * 32768.0f, 32767.0f));
    fy = fmaxf(0.0f, fminf(fy * 32768.0f, 32767.0f));
    const unsigned int xx = lbvh_expand_bits((unsigned int)fx);
    const unsigned int yy = lbvh_expand_bits((unsigned int)fy);
    return xx * 2u + yy;
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
    const DeviceBall* balls,
    unsigned int* morton_codes,
    int* indices,
    int count,
    float scene_w,
    float scene_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    morton_codes[i] = lbvh_morton2d(balls[i].x, balls[i].y, scene_w, scene_h);
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
    const DeviceBall* balls,
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
    const float cx = balls[obj_idx].x;
    const float cy = balls[obj_idx].y;
    const float rad = balls[obj_idx].radius;

    aabb_min_x[node_idx] = cx - rad;
    aabb_min_y[node_idx] = cy - rad;
    aabb_max_x[node_idx] = cx + rad;
    aabb_max_y[node_idx] = cy + rad;
    // Publish leaf AABB to other threads before signalling via the flag.
    // CUDA atomics are relaxed by default; without this fence the second
    // arrival at the parent may read stale child AABBs and produce a
    // garbage tree, devolving traversal into brute force.
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
    DeviceBall* balls,
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
    const DeviceBall self = balls[obj_idx];
    const float my_min_x = self.x - self.radius;
    const float my_min_y = self.y - self.radius;
    const float my_max_x = self.x + self.radius;
    const float my_max_y = self.y + self.radius;
    const Circle self_circle = ball_to_circle(&self);

    int stack[64];
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
                const DeviceBall other = balls[other_obj];
                const Circle other_circle = ball_to_circle(&other);
                if (circles_overlap(&self_circle, &other_circle)) {
                    balls[obj_idx].colliding = 1;
                    balls[other_obj].colliding = 1;
                    ++local_collisions;
                }
            }
        } else {
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

__global__ static void write_vbo_kernel(
    const DeviceBall* balls,
    RenderVertex* vertices,
    int count,
    int width,
    int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const DeviceBall b = balls[i];
    const float center_x = (b.x / (float)width) * 2.0f - 1.0f;
    const float center_y = 1.0f - (b.y / (float)height) * 2.0f;
    const float radius_x = fmaxf(9.0f, b.radius * 2.0f) / (float)width * 2.0f;
    const float radius_y = fmaxf(9.0f, b.radius * 2.0f) / (float)height * 2.0f;

    float r = 0.18f;
    float g = 0.78f;
    float bl = 1.0f;
    if (b.colliding) {
        r = 1.0f;
        g = 0.28f;
        bl = 0.14f;
    }

    const float corners[6][2] = {
        {-1.0f, -1.0f},
        { 1.0f, -1.0f},
        { 1.0f,  1.0f},
        {-1.0f, -1.0f},
        { 1.0f,  1.0f},
        {-1.0f,  1.0f}
    };

    const int base = i * 6;
    for (int vertex_index = 0; vertex_index < 6; ++vertex_index) {
        const float local_x = corners[vertex_index][0];
        const float local_y = corners[vertex_index][1];
        vertices[base + vertex_index].x = center_x + local_x * radius_x;
        vertices[base + vertex_index].y = center_y + local_y * radius_y;
        vertices[base + vertex_index].local_x = local_x;
        vertices[base + vertex_index].local_y = local_y;
        vertices[base + vertex_index].r = r;
        vertices[base + vertex_index].g = g;
        vertices[base + vertex_index].b = bl;
    }
}

static void host_init_balls(DeviceBall* h_balls, int count, int clustered, unsigned int seed) {
    unsigned int rng = seed ^ 0x9E3779B9u ^ (unsigned int)count ^ (clustered ? 0xC0FFEEu : 0xBADD00Du);

    float centers_x[VISUALIZER_MAX_CLUSTERS];
    float centers_y[VISUALIZER_MAX_CLUSTERS];
    if (clustered) {
        for (int i = 0; i < VISUALIZER_MAX_CLUSTERS; ++i) {
            centers_x[i] = rng_range_float(&rng, (float)g_width * 0.15f, (float)g_width * 0.85f);
            centers_y[i] = rng_range_float(&rng, (float)g_height * 0.15f, (float)g_height * 0.85f);
        }
    }

    for (int i = 0; i < count; ++i) {
        const float radius = rng_range_float(&rng, VISUALIZER_BALL_MIN_RADIUS, VISUALIZER_BALL_MAX_RADIUS);
        float x;
        float y;
        if (clustered) {
            const int cluster = (int)(rng_next_u32(&rng) % (unsigned int)VISUALIZER_MAX_CLUSTERS);
            const float angle = rng_range_float(&rng, 0.0f, 6.28318530718f);
            const float distance = rng_range_float(&rng, 0.0f, 90.0f);
            x = centers_x[cluster] + cosf(angle) * distance;
            y = centers_y[cluster] + sinf(angle) * distance;
        } else {
            x = rng_range_float(&rng, radius, (float)g_width - radius);
            y = rng_range_float(&rng, radius, (float)g_height - radius);
        }

        if (x < radius) x = radius;
        if (x > (float)g_width - radius) x = (float)g_width - radius;
        if (y < radius) y = radius;
        if (y > (float)g_height - radius) y = (float)g_height - radius;

        h_balls[i].x = x;
        h_balls[i].y = y;
        h_balls[i].vx = rng_range_float(&rng, -90.0f, 90.0f);
        h_balls[i].vy = rng_range_float(&rng, -90.0f, 90.0f);
        h_balls[i].radius = radius;
        h_balls[i].colliding = 0;
    }
}

static void cpu_integrate(DeviceBall* balls, int count, float dt, int width, int height) {
    for (int i = 0; i < count; ++i) {
        DeviceBall* b = &balls[i];
        b->x += b->vx * dt;
        b->y += b->vy * dt;
        if (b->x < b->radius) {
            b->x = b->radius;
            b->vx = fabsf(b->vx);
        } else if (b->x > (float)width - b->radius) {
            b->x = (float)width - b->radius;
            b->vx = -fabsf(b->vx);
        }
        if (b->y < b->radius) {
            b->y = b->radius;
            b->vy = fabsf(b->vy);
        } else if (b->y > (float)height - b->radius) {
            b->y = (float)height - b->radius;
            b->vy = -fabsf(b->vy);
        }
        b->colliding = 0;
    }
}

static void cpu_brute_force_collide(
    DeviceBall* balls,
    int count,
    unsigned long long* out_collisions,
    unsigned long long* out_candidates) {
    unsigned long long collisions = 0;
    unsigned long long candidates = 0;
    for (int i = 0; i < count; ++i) {
        Circle a;
        a.x = balls[i].x;
        a.y = balls[i].y;
        a.radius = balls[i].radius;
        for (int j = i + 1; j < count; ++j) {
            Circle b;
            b.x = balls[j].x;
            b.y = balls[j].y;
            b.radius = balls[j].radius;
            ++candidates;
            if (circles_overlap(&a, &b)) {
                balls[i].colliding = 1;
                balls[j].colliding = 1;
                ++collisions;
            }
        }
    }
    *out_collisions = collisions;
    *out_candidates = candidates;
}

static void run_brute_force(int blocks, int threads) {
    brute_force_collision_kernel<<<blocks, threads>>>(
        g_balls,
        g_object_count,
        g_collision_count,
        g_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
}

static void run_uniform_grid(int blocks, int threads) {
    const int cell_blocks = (g_total_cells + threads - 1) / threads;
    init_cell_ranges_kernel<<<cell_blocks, threads>>>(g_cell_start, g_cell_end, g_total_cells);
    CUDA_CHECK(cudaGetLastError());

    compute_cell_keys_kernel<<<blocks, threads>>>(
        g_balls,
        g_cell_keys,
        g_indices,
        g_object_count,
        VISUALIZER_GRID_CELL_SIZE,
        g_grid_width,
        g_grid_height);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> keys_ptr(g_cell_keys);
    thrust::device_ptr<int> indices_ptr(g_indices);
    thrust::sort_by_key(keys_ptr, keys_ptr + g_object_count, indices_ptr);

    build_cell_ranges_kernel<<<blocks, threads>>>(g_cell_keys, g_cell_start, g_cell_end, g_object_count);
    CUDA_CHECK(cudaGetLastError());

    grid_collision_kernel<<<blocks, threads>>>(
        g_balls,
        g_cell_keys,
        g_indices,
        g_cell_start,
        g_cell_end,
        g_object_count,
        g_grid_width,
        g_grid_height,
        g_collision_count,
        g_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
}

static void run_lbvh(int blocks, int threads) {
    const int num_nodes = 2 * g_object_count - 1;

    CUDA_CHECK(cudaMemset(g_lbvh_flags, 0, (size_t)num_nodes * sizeof(int)));
    // parent[0] (root) is never written by build_radix_tree; mark every parent
    // slot as -1 so the bottom-up walk terminates cleanly at the root.
    CUDA_CHECK(cudaMemset(g_lbvh_parent, 0xFF, (size_t)num_nodes * sizeof(int)));

    lbvh_morton_kernel<<<blocks, threads>>>(
        g_balls, g_lbvh_morton, g_lbvh_indices,
        g_object_count, (float)g_width, (float)g_height);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<unsigned int> morton_ptr(g_lbvh_morton);
    thrust::device_ptr<int> indices_ptr(g_lbvh_indices);
    thrust::sort_by_key(morton_ptr, morton_ptr + g_object_count, indices_ptr);

    const int internal_blocks = ((g_object_count - 1) + threads - 1) / threads;
    if (internal_blocks > 0) {
        lbvh_build_radix_tree_kernel<<<internal_blocks, threads>>>(
            g_lbvh_morton, g_object_count,
            g_lbvh_parent, g_lbvh_left, g_lbvh_right);
        CUDA_CHECK(cudaGetLastError());
    }

    lbvh_aabbs_kernel<<<blocks, threads>>>(
        g_balls, g_lbvh_indices, g_object_count,
        g_lbvh_parent, g_lbvh_left, g_lbvh_right, g_lbvh_flags,
        g_lbvh_aabb_min_x, g_lbvh_aabb_min_y,
        g_lbvh_aabb_max_x, g_lbvh_aabb_max_y);
    CUDA_CHECK(cudaGetLastError());

    lbvh_traverse_kernel<<<blocks, threads>>>(
        g_balls, g_lbvh_indices, g_object_count,
        g_lbvh_left, g_lbvh_right,
        g_lbvh_aabb_min_x, g_lbvh_aabb_min_y,
        g_lbvh_aabb_max_x, g_lbvh_aabb_max_y,
        g_collision_count, g_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" int cuda_visualizer_create(unsigned int vbo, int object_count, int width, int height) {
    g_object_count = object_count;
    g_width = width;
    g_height = height;
    g_grid_width = (int)ceilf((float)width / VISUALIZER_GRID_CELL_SIZE);
    g_grid_height = (int)ceilf((float)height / VISUALIZER_GRID_CELL_SIZE);
    g_total_cells = g_grid_width * g_grid_height;

    CUDA_CHECK(cudaMalloc((void**)&g_balls, (size_t)object_count * sizeof(DeviceBall)));
    g_host_balls = (DeviceBall*)malloc((size_t)object_count * sizeof(DeviceBall));
    if (g_host_balls == NULL) {
        return 0;
    }
    CUDA_CHECK(cudaMalloc((void**)&g_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&g_candidate_pair_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&g_cell_keys, (size_t)object_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_indices, (size_t)object_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_cell_start, (size_t)g_total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_cell_end, (size_t)g_total_cells * sizeof(int)));

    const size_t lbvh_nodes = (size_t)object_count * 2 - 1;
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_morton, (size_t)object_count * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_indices, (size_t)object_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_aabb_min_x, lbvh_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_aabb_min_y, lbvh_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_aabb_max_x, lbvh_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_aabb_max_y, lbvh_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_parent, lbvh_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_left, lbvh_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_right, lbvh_nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_lbvh_flags, lbvh_nodes * sizeof(int)));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&g_vbo_resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaEventCreate(&g_start_event));
    CUDA_CHECK(cudaEventCreate(&g_stop_event));

    return cuda_visualizer_reset(0);
}

extern "C" void cuda_visualizer_destroy(void) {
    if (g_vbo_resource != NULL) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(g_vbo_resource));
        g_vbo_resource = NULL;
    }
    if (g_balls != NULL) {
        CUDA_CHECK(cudaFree(g_balls));
        g_balls = NULL;
    }
    if (g_host_balls != NULL) {
        free(g_host_balls);
        g_host_balls = NULL;
    }
    if (g_collision_count != NULL) {
        CUDA_CHECK(cudaFree(g_collision_count));
        g_collision_count = NULL;
    }
    if (g_candidate_pair_count != NULL) {
        CUDA_CHECK(cudaFree(g_candidate_pair_count));
        g_candidate_pair_count = NULL;
    }
    if (g_cell_keys != NULL) {
        CUDA_CHECK(cudaFree(g_cell_keys));
        g_cell_keys = NULL;
    }
    if (g_indices != NULL) {
        CUDA_CHECK(cudaFree(g_indices));
        g_indices = NULL;
    }
    if (g_cell_start != NULL) {
        CUDA_CHECK(cudaFree(g_cell_start));
        g_cell_start = NULL;
    }
    if (g_cell_end != NULL) {
        CUDA_CHECK(cudaFree(g_cell_end));
        g_cell_end = NULL;
    }
    if (g_lbvh_morton != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_morton));
        g_lbvh_morton = NULL;
    }
    if (g_lbvh_indices != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_indices));
        g_lbvh_indices = NULL;
    }
    if (g_lbvh_aabb_min_x != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_aabb_min_x));
        g_lbvh_aabb_min_x = NULL;
    }
    if (g_lbvh_aabb_min_y != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_aabb_min_y));
        g_lbvh_aabb_min_y = NULL;
    }
    if (g_lbvh_aabb_max_x != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_aabb_max_x));
        g_lbvh_aabb_max_x = NULL;
    }
    if (g_lbvh_aabb_max_y != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_aabb_max_y));
        g_lbvh_aabb_max_y = NULL;
    }
    if (g_lbvh_parent != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_parent));
        g_lbvh_parent = NULL;
    }
    if (g_lbvh_left != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_left));
        g_lbvh_left = NULL;
    }
    if (g_lbvh_right != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_right));
        g_lbvh_right = NULL;
    }
    if (g_lbvh_flags != NULL) {
        CUDA_CHECK(cudaFree(g_lbvh_flags));
        g_lbvh_flags = NULL;
    }
    if (g_start_event != NULL) {
        CUDA_CHECK(cudaEventDestroy(g_start_event));
        g_start_event = NULL;
    }
    if (g_stop_event != NULL) {
        CUDA_CHECK(cudaEventDestroy(g_stop_event));
        g_stop_event = NULL;
    }
}

extern "C" int cuda_visualizer_reset(int clustered) {
    if (g_balls == NULL) {
        return 0;
    }

    DeviceBall* h_balls = (DeviceBall*)malloc((size_t)g_object_count * sizeof(DeviceBall));
    if (h_balls == NULL) {
        return 0;
    }
    host_init_balls(h_balls, g_object_count, clustered, VISUALIZER_INIT_SEED);
    CUDA_CHECK(cudaMemcpy(g_balls, h_balls, (size_t)g_object_count * sizeof(DeviceBall), cudaMemcpyHostToDevice));
    free(h_balls);
    return 1;
}

extern "C" int cuda_visualizer_set_mode(int mode) {
    if (mode != VISUALIZER_MODE_CUDA_BRUTE_FORCE
        && mode != VISUALIZER_MODE_CUDA_UNIFORM_GRID
        && mode != VISUALIZER_MODE_CPU_BRUTE_FORCE
        && mode != VISUALIZER_MODE_CUDA_LBVH) {
        return 0;
    }
    g_mode = mode;
    return 1;
}

extern "C" int cuda_visualizer_step(float dt, VisualizerMetrics* metrics) {
    if (g_balls == NULL || g_vbo_resource == NULL || metrics == NULL) {
        return 0;
    }

    const int threads = 256;
    const int blocks = (g_object_count + threads - 1) / threads;
    RenderVertex* vertices = NULL;
    size_t mapped_size = 0;

    if (g_mode == VISUALIZER_MODE_CPU_BRUTE_FORCE) {
        CUDA_CHECK(cudaMemcpy(g_host_balls, g_balls,
                              (size_t)g_object_count * sizeof(DeviceBall),
                              cudaMemcpyDeviceToHost));

        const double t0 = timer_now_ms();
        cpu_integrate(g_host_balls, g_object_count, dt, g_width, g_height);
        unsigned long long h_collisions = 0;
        unsigned long long h_candidates = 0;
        cpu_brute_force_collide(g_host_balls, g_object_count, &h_collisions, &h_candidates);
        const double t1 = timer_now_ms();

        CUDA_CHECK(cudaMemcpy(g_balls, g_host_balls,
                              (size_t)g_object_count * sizeof(DeviceBall),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaGraphicsMapResources(1, &g_vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertices, &mapped_size, g_vbo_resource));
        write_vbo_kernel<<<blocks, threads>>>(g_balls, vertices, g_object_count, g_width, g_height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &g_vbo_resource, 0));

        metrics->collision_count = h_collisions;
        metrics->candidate_pair_count = h_candidates;
        metrics->gpu_time_ms = (float)(t1 - t0);
        return 1;
    }

    CUDA_CHECK(cudaMemset(g_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(g_start_event));

    integrate_kernel<<<blocks, threads>>>(g_balls, g_object_count, dt, g_width, g_height);
    CUDA_CHECK(cudaGetLastError());

    if (g_mode == VISUALIZER_MODE_CUDA_UNIFORM_GRID) {
        run_uniform_grid(blocks, threads);
    } else if (g_mode == VISUALIZER_MODE_CUDA_LBVH) {
        run_lbvh(blocks, threads);
    } else {
        run_brute_force(blocks, threads);
    }

    CUDA_CHECK(cudaGraphicsMapResources(1, &g_vbo_resource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertices, &mapped_size, g_vbo_resource));
    write_vbo_kernel<<<blocks, threads>>>(g_balls, vertices, g_object_count, g_width, g_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &g_vbo_resource, 0));

    CUDA_CHECK(cudaEventRecord(g_stop_event));
    CUDA_CHECK(cudaEventSynchronize(g_stop_event));
    CUDA_CHECK(cudaMemcpy(&metrics->collision_count, g_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&metrics->candidate_pair_count, g_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&metrics->gpu_time_ms, g_start_event, g_stop_event));

    return 1;
}
