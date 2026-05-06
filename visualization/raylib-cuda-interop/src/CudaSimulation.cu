#include "RaylibInteropTypes.cuh"
#include "Circle.h"
#include "CollisionMath.h"
#include "CudaUtils.cuh"
#include "Rng.h"

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
static unsigned long long* g_collision_count = NULL;
static unsigned long long* g_candidate_pair_count = NULL;
static int* g_cell_keys = NULL;
static int* g_indices = NULL;
static int* g_cell_start = NULL;
static int* g_cell_end = NULL;
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

extern "C" int cuda_visualizer_create(unsigned int vbo, int object_count, int width, int height) {
    g_object_count = object_count;
    g_width = width;
    g_height = height;
    g_grid_width = (int)ceilf((float)width / VISUALIZER_GRID_CELL_SIZE);
    g_grid_height = (int)ceilf((float)height / VISUALIZER_GRID_CELL_SIZE);
    g_total_cells = g_grid_width * g_grid_height;

    CUDA_CHECK(cudaMalloc((void**)&g_balls, (size_t)object_count * sizeof(DeviceBall)));
    CUDA_CHECK(cudaMalloc((void**)&g_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&g_candidate_pair_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&g_cell_keys, (size_t)object_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_indices, (size_t)object_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_cell_start, (size_t)g_total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&g_cell_end, (size_t)g_total_cells * sizeof(int)));
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
    if (mode != VISUALIZER_MODE_CUDA_BRUTE_FORCE && mode != VISUALIZER_MODE_CUDA_UNIFORM_GRID) {
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

    CUDA_CHECK(cudaMemset(g_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(g_start_event));

    integrate_kernel<<<blocks, threads>>>(g_balls, g_object_count, dt, g_width, g_height);
    CUDA_CHECK(cudaGetLastError());

    if (g_mode == VISUALIZER_MODE_CUDA_UNIFORM_GRID) {
        run_uniform_grid(blocks, threads);
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
