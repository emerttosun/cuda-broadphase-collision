#include "RaylibInteropTypes.cuh"
#include "Circle.h"
#include "CollisionMath.h"
#include "CudaUtils.cuh"
#include "RadiusProfile.h"
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
    float z;
    float vx;
    float vy;
    float vz;
    float radius;
    int colliding;
} DeviceBall;

#define VISUALIZER_MAX_CLUSTERS 4
#define VISUALIZER_GRID_CELL_SIZE 16.0f
#define VISUALIZER_INIT_SEED 202405u
#define VISUALIZER_MOUSE_RADIUS 54.0f
#define VISUALIZER_BALL_MIN_SPEED 20.0f
#define VISUALIZER_BALL_MAX_SPEED 200.0f
#define VISUALIZER_COLLISION_SPEED_BOOST 1.04f

static DeviceBall* g_balls = NULL;
static DeviceBall* g_host_balls = NULL;
static float* g_collision_delta_vx = NULL;
static float* g_collision_delta_vy = NULL;
static float* g_collision_delta_vz = NULL;
static float* g_collision_delta_x = NULL;
static float* g_collision_delta_y = NULL;
static float* g_collision_delta_z = NULL;
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
/* Hierarchical Spatial Hashing (HSH) persistent device buffers. */
static int* g_hash_composite_keys = NULL;
static int* g_hash_indices = NULL;
static int* g_hash_cell_start = NULL;
static int* g_hash_cell_end = NULL;
static int g_hash_table_M = 0;
static int g_hash_total_buckets = 0;
static int g_hash_num_levels = 0;
static float g_hash_base_cell_size = 0.0f;
static cudaGraphicsResource* g_vbo_resource = NULL;
static cudaEvent_t g_start_event = NULL;
static cudaEvent_t g_stop_event = NULL;
static int g_object_count = 0;
static int g_width = 1280;
static int g_height = 720;
static float g_depth = 540.0f;
static int g_grid_width = 0;
static int g_grid_height = 0;
static int g_grid_depth = 0;
static int g_total_cells = 0;
static int g_mode = VISUALIZER_MODE_CUDA_BRUTE_FORCE;
static RadiusProfile g_radius_profile = RADIUS_PROFILE_NARROW;

__device__ __host__ static int balls_overlap_3d(const DeviceBall* a, const DeviceBall* b) {
    const float dx = a->x - b->x;
    const float dy = a->y - b->y;
    const float dz = a->z - b->z;
    const float radius_sum = a->radius + b->radius;
    const float distance_squared = dx * dx + dy * dy + dz * dz;
    return distance_squared <= radius_sum * radius_sum;
}

__global__ static void integrate_kernel(
    DeviceBall* balls,
    int count,
    float dt,
    int width,
    int height,
    float depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    DeviceBall b = balls[i];
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    b.z += b.vz * dt;

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

    if (b.z < b.radius) {
        b.z = b.radius;
        b.vz = fabsf(b.vz);
    } else if (b.z > depth - b.radius) {
        b.z = depth - b.radius;
        b.vz = -fabsf(b.vz);
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

__device__ __host__ static void project_ball_to_screen(
    const DeviceBall* b,
    int width,
    int height,
    float depth,
    const VisualizerCamera* camera,
    float* screen_x,
    float* screen_y,
    float* sprite_radius) {
    const float cy = cosf(camera->yaw);
    const float sy = sinf(camera->yaw);
    const float cp = cosf(camera->pitch);
    const float sp = sinf(camera->pitch);

    const float forward_x = sy * cp;
    const float forward_y = sp;
    const float forward_z = cy * cp;
    const float right_x = cy;
    const float right_z = -sy;
    const float up_x = -sy * sp;
    const float up_y = cp;
    const float up_z = -cy * sp;

    const float world_x = b->x - width * 0.5f;
    const float world_y = height * 0.5f - b->y;
    const float world_z = b->z - depth * 0.5f;
    const float rel_x = world_x + forward_x * camera->distance;
    const float rel_y = world_y + forward_y * camera->distance;
    const float rel_z = world_z + forward_z * camera->distance;

    const float camera_x = rel_x * right_x + rel_z * right_z;
    const float camera_y = rel_x * up_x + rel_y * up_y + rel_z * up_z;
    float camera_z = rel_x * forward_x + rel_y * forward_y + rel_z * forward_z;
    if (camera_z < 20.0f) {
        camera_z = 20.0f;
    }

    const float focal = 760.0f;
    *screen_x = width * 0.5f + focal * camera_x / camera_z;
    *screen_y = height * 0.5f - focal * camera_y / camera_z;
    *sprite_radius = fmaxf(6.0f, b->radius * focal / camera_z);
}

__device__ __host__ static void apply_bounds(DeviceBall* b, int width, int height, float depth) {
    if (b->x < b->radius) {
        b->x = b->radius;
        b->vx = fabsf(b->vx);
    } else if (b->x > width - b->radius) {
        b->x = width - b->radius;
        b->vx = -fabsf(b->vx);
    }

    if (b->y < b->radius) {
        b->y = b->radius;
        b->vy = fabsf(b->vy);
    } else if (b->y > height - b->radius) {
        b->y = height - b->radius;
        b->vy = -fabsf(b->vy);
    }

    if (b->z < b->radius) {
        b->z = b->radius;
        b->vz = fabsf(b->vz);
    } else if (b->z > depth - b->radius) {
        b->z = depth - b->radius;
        b->vz = -fabsf(b->vz);
    }
}

__device__ __host__ static void clamp_speed_after_collision(DeviceBall* b, float original_speed) {
    const float speed_sq = b->vx * b->vx + b->vy * b->vy + b->vz * b->vz;
    if (speed_sq <= 1.0e-6f || original_speed <= 1.0e-6f) {
        return;
    }

    float target_speed = original_speed * VISUALIZER_COLLISION_SPEED_BOOST;
    if (target_speed > VISUALIZER_BALL_MAX_SPEED) {
        target_speed = VISUALIZER_BALL_MAX_SPEED;
    }

    const float scale = target_speed / sqrtf(speed_sq);
    b->vx *= scale;
    b->vy *= scale;
    b->vz *= scale;
}

__device__ static void accumulate_collision_response(
    const DeviceBall* balls,
    int a_index,
    int b_index,
    float* delta_vx,
    float* delta_vy,
    float* delta_vz,
    float* delta_x,
    float* delta_y,
    float* delta_z) {
    const DeviceBall a = balls[a_index];
    const DeviceBall b = balls[b_index];
    float nx = b.x - a.x;
    float ny = b.y - a.y;
    float nz = b.z - a.z;
    const float radius_sum = a.radius + b.radius;
    float distance_sq = nx * nx + ny * ny + nz * nz;

    if (distance_sq < 1.0e-6f) {
        const float fallback = (a_index < b_index) ? 1.0f : -1.0f;
        nx = fallback;
        ny = 0.0f;
        nz = 0.0f;
        distance_sq = 1.0f;
    }

    const float distance = sqrtf(distance_sq);
    nx /= distance;
    ny /= distance;
    nz /= distance;

    const float penetration = radius_sum - distance;
    if (penetration > 0.0f) {
        const float correction = penetration * 0.5f + 0.01f;
        atomicAdd(&delta_x[a_index], -nx * correction);
        atomicAdd(&delta_y[a_index], -ny * correction);
        atomicAdd(&delta_z[a_index], -nz * correction);
        atomicAdd(&delta_x[b_index], nx * correction);
        atomicAdd(&delta_y[b_index], ny * correction);
        atomicAdd(&delta_z[b_index], nz * correction);
    }

    const float rvx = a.vx - b.vx;
    const float rvy = a.vy - b.vy;
    const float rvz = a.vz - b.vz;
    const float relative_speed = rvx * nx + rvy * ny + rvz * nz;
    if (relative_speed <= 0.0f) {
        return;
    }

    atomicAdd(&delta_vx[a_index], -relative_speed * nx);
    atomicAdd(&delta_vy[a_index], -relative_speed * ny);
    atomicAdd(&delta_vz[a_index], -relative_speed * nz);
    atomicAdd(&delta_vx[b_index], relative_speed * nx);
    atomicAdd(&delta_vy[b_index], relative_speed * ny);
    atomicAdd(&delta_vz[b_index], relative_speed * nz);
}

__global__ static void apply_collision_response_kernel(
    DeviceBall* balls,
    int count,
    const float* delta_vx,
    const float* delta_vy,
    const float* delta_vz,
    const float* delta_x,
    const float* delta_y,
    const float* delta_z,
    int width,
    int height,
    float depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    DeviceBall b = balls[i];
    const float original_speed = sqrtf(b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
    b.vx += delta_vx[i];
    b.vy += delta_vy[i];
    b.vz += delta_vz[i];
    b.x += delta_x[i];
    b.y += delta_y[i];
    b.z += delta_z[i];

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

    if (b.z < b.radius) {
        b.z = b.radius;
        b.vz = fabsf(b.vz);
    } else if (b.z > depth - b.radius) {
        b.z = depth - b.radius;
        b.vz = -fabsf(b.vz);
    }

    clamp_speed_after_collision(&b, original_speed);
    balls[i] = b;
}

__device__ __host__ static void apply_mouse_collision(
    DeviceBall* b,
    float mouse_x,
    float mouse_y,
    int width,
    int height,
    float depth,
    const VisualizerCamera* camera) {
    float screen_x = 0.0f;
    float screen_y = 0.0f;
    float sprite_radius = 0.0f;
    project_ball_to_screen(b, width, height, depth, camera, &screen_x, &screen_y, &sprite_radius);

    float dx = screen_x - mouse_x;
    float dy = screen_y - mouse_y;
    float distance_sq = dx * dx + dy * dy;
    const float collider_radius = VISUALIZER_MOUSE_RADIUS + sprite_radius;
    if (distance_sq > collider_radius * collider_radius) {
        return;
    }

    if (distance_sq < 1.0e-5f) {
        dx = 1.0f;
        dy = 0.0f;
        distance_sq = 1.0f;
    }

    const float distance = sqrtf(distance_sq);
    const float nx = dx / distance;
    const float ny = dy / distance;
    const float penetration = collider_radius - distance;
    const float correction = penetration * 0.72f + 0.5f;
    const float original_speed = sqrtf(b->vx * b->vx + b->vy * b->vy + b->vz * b->vz);
    const float tangential_vx = b->vx - (b->vx * nx + b->vy * ny) * nx;
    const float tangential_vy = b->vy - (b->vx * nx + b->vy * ny) * ny;

    b->x += nx * correction;
    b->y += ny * correction;
    b->vx = tangential_vx + nx * original_speed;
    b->vy = tangential_vy + ny * original_speed;
    clamp_speed_after_collision(b, original_speed);
    b->colliding = 1;
    apply_bounds(b, width, height, depth);
}

__global__ static void mouse_collision_kernel(
    DeviceBall* balls,
    int count,
    float mouse_x,
    float mouse_y,
    int mouse_active,
    int width,
    int height,
    float depth,
    VisualizerCamera camera) {
    if (!mouse_active) {
        return;
    }

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    DeviceBall b = balls[i];
    apply_mouse_collision(&b, mouse_x, mouse_y, width, height, depth, &camera);
    balls[i] = b;
}

__global__ static void brute_force_collision_kernel(
    DeviceBall* balls,
    int count,
    float* delta_vx,
    float* delta_vy,
    float* delta_vz,
    float* delta_x,
    float* delta_y,
    float* delta_z,
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

        if (balls_overlap_3d(&a, &b)) {
            balls[i].colliding = 1;
            balls[j].colliding = 1;
            accumulate_collision_response(
                balls, i, j,
                delta_vx, delta_vy, delta_vz,
                delta_x, delta_y, delta_z);
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

__global__ static void compute_cell_keys_3d_kernel(
    const DeviceBall* balls,
    int* cell_keys,
    int* indices,
    int count,
    float cell_size,
    int grid_width,
    int grid_height,
    int grid_depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int cell_x = clamp_int_device((int)floorf(balls[i].x / cell_size), 0, grid_width - 1);
    const int cell_y = clamp_int_device((int)floorf(balls[i].y / cell_size), 0, grid_height - 1);
    const int cell_z = clamp_int_device((int)floorf(balls[i].z / cell_size), 0, grid_depth - 1);
    cell_keys[i] = (cell_z * grid_height + cell_y) * grid_width + cell_x;
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
    float cell_size,
    float max_radius,
    float* delta_vx,
    float* delta_vy,
    float* delta_vz,
    float* delta_x,
    float* delta_y,
    float* delta_z,
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
    int neighbor_range = (int)ceilf((self.radius + max_radius) / cell_size);
    if (neighbor_range < 1) {
        neighbor_range = 1;
    }

    for (int dy = -neighbor_range; dy <= neighbor_range; ++dy) {
        for (int dx = -neighbor_range; dx <= neighbor_range; ++dx) {
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
                if (balls_overlap_3d(&self, &other)) {
                    balls[object_index].colliding = 1;
                    balls[other_index].colliding = 1;
                    accumulate_collision_response(
                        balls, object_index, other_index,
                        delta_vx, delta_vy, delta_vz,
                        delta_x, delta_y, delta_z);
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

__global__ static void grid_collision_3d_kernel(
    DeviceBall* balls,
    const int* sorted_cell_keys,
    const int* sorted_indices,
    const int* cell_start,
    const int* cell_end,
    int count,
    int grid_width,
    int grid_height,
    int grid_depth,
    float cell_size,
    float max_radius,
    float* delta_vx,
    float* delta_vy,
    float* delta_vz,
    float* delta_x,
    float* delta_y,
    float* delta_z,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const int sorted_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_pos >= count) {
        return;
    }

    const int object_index = sorted_indices[sorted_pos];
    const DeviceBall self = balls[object_index];
    const int cell_id = sorted_cell_keys[sorted_pos];
    const int xy_cells = grid_width * grid_height;
    const int cell_z = cell_id / xy_cells;
    const int cell_xy = cell_id - cell_z * xy_cells;
    const int cell_y = cell_xy / grid_width;
    const int cell_x = cell_xy - cell_y * grid_width;

    unsigned long long local_candidates = 0;
    unsigned long long local_collisions = 0;
    int neighbor_range = (int)ceilf((self.radius + max_radius) / cell_size);
    if (neighbor_range < 1) {
        neighbor_range = 1;
    }

    for (int dz = -neighbor_range; dz <= neighbor_range; ++dz) {
        for (int dy = -neighbor_range; dy <= neighbor_range; ++dy) {
            for (int dx = -neighbor_range; dx <= neighbor_range; ++dx) {
                const int nx = cell_x + dx;
                const int ny = cell_y + dy;
                const int nz = cell_z + dz;
                if (nx < 0 || ny < 0 || nz < 0
                    || nx >= grid_width || ny >= grid_height || nz >= grid_depth) {
                    continue;
                }

                const int neighbor_cell = (nz * grid_height + ny) * grid_width + nx;
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
                    if (balls_overlap_3d(&self, &other)) {
                        balls[object_index].colliding = 1;
                        balls[other_index].colliding = 1;
                        accumulate_collision_response(
                            balls, object_index, other_index,
                            delta_vx, delta_vy, delta_vz,
                            delta_x, delta_y, delta_z);
                        local_collisions++;
                    }
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

__device__ static unsigned int lbvh_part_1_by_1(unsigned int v) {
    v &= 0x0000FFFFu;
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

__device__ static unsigned int lbvh_morton2d(float x, float y, float scene_w, float scene_h) {
    float fx = x / scene_w;
    float fy = y / scene_h;
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
    float* delta_vx,
    float* delta_vy,
    float* delta_vz,
    float* delta_x,
    float* delta_y,
    float* delta_z,
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
                if (balls_overlap_3d(&self, &other)) {
                    balls[obj_idx].colliding = 1;
                    balls[other_obj].colliding = 1;
                    accumulate_collision_response(
                        balls, obj_idx, other_obj,
                        delta_vx, delta_vy, delta_vz,
                        delta_x, delta_y, delta_z);
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

/* ------------------------------------------------------------------ */
/*  Hierarchical Spatial Hashing (HSH) — 3D variant                    */
/*                                                                     */
/*  Algorithm matches src/CudaHash.cu (benchmark module) extended to   */
/*  three dimensions with collision-response writes that match the     */
/*  visualizer's existing per-method dispatch contract.                */
/*  References: Teschner et al. 2003 (hash); Eitz/Lixu 2007 (levels).  */
/* ------------------------------------------------------------------ */

#define VIS_HASH_PRIME_X 73856093u
#define VIS_HASH_PRIME_Y 19349663u
#define VIS_HASH_PRIME_Z 83492791u
#define VIS_HASH_MAX_LEVELS 8

__device__ static inline unsigned int hash_bucket_3d(int cx, int cy, int cz, unsigned int table_M) {
    const unsigned int h = ((unsigned int)cx) * VIS_HASH_PRIME_X
                         ^ ((unsigned int)cy) * VIS_HASH_PRIME_Y
                         ^ ((unsigned int)cz) * VIS_HASH_PRIME_Z;
    return h % table_M;
}

__device__ static inline int hash_select_level_d(
    float radius,
    float base_cell_size,
    int max_level_inclusive) {
    const float diameter = 2.0f * radius;
    if (diameter <= base_cell_size) {
        return 0;
    }
    int L = (int)ceilf(log2f(diameter / base_cell_size));
    if (L < 0) L = 0;
    if (L > max_level_inclusive) L = max_level_inclusive;
    return L;
}

__global__ static void hash_assign_3d_kernel(
    const DeviceBall* balls,
    int count,
    float base_cell_size,
    int num_levels,
    unsigned int table_M,
    int* composite_keys,
    int* indices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const DeviceBall b = balls[i];
    const int L = hash_select_level_d(b.radius, base_cell_size, num_levels - 1);
    const float cs_L = base_cell_size * (float)(1 << L);
    const float inv = 1.0f / cs_L;
    const int cx = (int)floorf(b.x * inv);
    const int cy = (int)floorf(b.y * inv);
    const int cz = (int)floorf(b.z * inv);
    const unsigned int bucket = hash_bucket_3d(cx, cy, cz, table_M);
    composite_keys[i] = (int)((unsigned int)L * table_M + bucket);
    indices[i] = i;
}

/**
 * @brief 3D HSH query + collision-response kernel.
 *
 * One thread per ball. Walks the ball's storage level upward, scanning
 * a 3x3x3 cell neighbourhood per level around the ball's own cell
 * (justified by the level invariant @c 2*r <= cs(L); see CudaHash.cu).
 * Counts each unordered pair exactly once via the same self-level
 * @c j > i filter as the benchmark module; cross-level pairs are
 * naturally counted once because the larger object's level is queried
 * upward only.
 */
__global__ static void hash_collide_3d_kernel(
    DeviceBall* balls,
    const int* sorted_indices,
    const int* cell_start,
    const int* cell_end,
    int count,
    float base_cell_size,
    int num_levels,
    unsigned int table_M,
    float* delta_vx,
    float* delta_vy,
    float* delta_vz,
    float* delta_x,
    float* delta_y,
    float* delta_z,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const DeviceBall self = balls[i];
    const int L_self = hash_select_level_d(self.radius, base_cell_size, num_levels - 1);

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;

    for (int L_q = L_self; L_q < num_levels; ++L_q) {
        const float cs_q = base_cell_size * (float)(1 << L_q);
        const float inv = 1.0f / cs_q;
        const int self_cx = (int)floorf(self.x * inv);
        const int self_cy = (int)floorf(self.y * inv);
        const int self_cz = (int)floorf(self.z * inv);

        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int cx = self_cx + dx;
                    const int cy = self_cy + dy;
                    const int cz = self_cz + dz;
                    const unsigned int bucket = hash_bucket_3d(cx, cy, cz, table_M);
                    const int composite = (int)((unsigned int)L_q * table_M + bucket);
                    const int start = cell_start[composite];
                    const int end = cell_end[composite];
                    if (start < 0 || end < 0) {
                        continue;
                    }

                    for (int p = start; p < end; ++p) {
                        const int j = sorted_indices[p];
                        const DeviceBall other = balls[j];
                        const int other_cx = (int)floorf(other.x * inv);
                        const int other_cy = (int)floorf(other.y * inv);
                        const int other_cz = (int)floorf(other.z * inv);
                        if (other_cx != cx || other_cy != cy || other_cz != cz) {
                            continue;
                        }
                        if (L_q == L_self && j <= i) {
                            continue;
                        }

                        ++local_candidates;
                        if (balls_overlap_3d(&self, &other)) {
                            balls[i].colliding = 1;
                            balls[j].colliding = 1;
                            accumulate_collision_response(
                                balls, i, j,
                                delta_vx, delta_vy, delta_vz,
                                delta_x, delta_y, delta_z);
                            ++local_collisions;
                        }
                    }
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
    int height,
    float depth,
    VisualizerCamera camera) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const DeviceBall b = balls[i];
    float projected_x = 0.0f;
    float projected_y = 0.0f;
    float sprite_radius = 0.0f;
    project_ball_to_screen(&b, width, height, depth, &camera, &projected_x, &projected_y, &sprite_radius);
    const float center_x = (projected_x / (float)width) * 2.0f - 1.0f;
    const float center_y = 1.0f - (projected_y / (float)height) * 2.0f;
    const float radius_x = sprite_radius / (float)width * 2.0f;
    const float radius_y = sprite_radius / (float)height * 2.0f;
    const float vertex_depth = fminf(0.95f, fmaxf(-0.95f, b.z / depth * 1.9f - 0.95f));

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
        vertices[base + vertex_index].depth = vertex_depth;
        vertices[base + vertex_index].local_x = local_x;
        vertices[base + vertex_index].local_y = local_y;
        vertices[base + vertex_index].r = r;
        vertices[base + vertex_index].g = g;
        vertices[base + vertex_index].b = bl;
    }
}

static void host_init_balls(DeviceBall* h_balls, int count, int clustered, RadiusProfile radius_profile, unsigned int seed) {
    unsigned int rng = seed
        ^ 0x9E3779B9u
        ^ (unsigned int)count
        ^ (unsigned int)radius_profile * 0x85EBCA6Bu
        ^ (clustered ? 0xC0FFEEu : 0xBADD00Du);

    float centers_x[VISUALIZER_MAX_CLUSTERS];
    float centers_y[VISUALIZER_MAX_CLUSTERS];
    float centers_z[VISUALIZER_MAX_CLUSTERS];
    if (clustered) {
        for (int i = 0; i < VISUALIZER_MAX_CLUSTERS; ++i) {
            centers_x[i] = rng_range_float(&rng, (float)g_width * 0.15f, (float)g_width * 0.85f);
            centers_y[i] = rng_range_float(&rng, (float)g_height * 0.15f, (float)g_height * 0.85f);
            centers_z[i] = rng_range_float(&rng, g_depth * 0.15f, g_depth * 0.85f);
        }
    }

    for (int i = 0; i < count; ++i) {
        const float radius = sample_radius_for_profile(&rng, radius_profile);
        float x;
        float y;
        float z;
        if (clustered) {
            const int cluster = (int)(rng_next_u32(&rng) % (unsigned int)VISUALIZER_MAX_CLUSTERS);
            const float angle = rng_range_float(&rng, 0.0f, 6.28318530718f);
            const float z_angle = rng_range_float(&rng, -1.57079632679f, 1.57079632679f);
            const float distance = rng_range_float(&rng, 0.0f, 90.0f);
            x = centers_x[cluster] + cosf(angle) * distance;
            y = centers_y[cluster] + sinf(angle) * distance;
            z = centers_z[cluster] + sinf(z_angle) * distance;
        } else {
            x = rng_range_float(&rng, radius, (float)g_width - radius);
            y = rng_range_float(&rng, radius, (float)g_height - radius);
            z = rng_range_float(&rng, radius, g_depth - radius);
        }

        if (x < radius) x = radius;
        if (x > (float)g_width - radius) x = (float)g_width - radius;
        if (y < radius) y = radius;
        if (y > (float)g_height - radius) y = (float)g_height - radius;
        if (z < radius) z = radius;
        if (z > g_depth - radius) z = g_depth - radius;

        h_balls[i].x = x;
        h_balls[i].y = y;
        h_balls[i].z = z;
        float dir_x = rng_range_float(&rng, -1.0f, 1.0f);
        float dir_y = rng_range_float(&rng, -1.0f, 1.0f);
        float dir_z = rng_range_float(&rng, -1.0f, 1.0f);
        float dir_len_sq = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z;
        if (dir_len_sq < 1.0e-5f) {
            dir_x = 1.0f;
            dir_y = 0.0f;
            dir_z = 0.0f;
            dir_len_sq = 1.0f;
        }
        const float inv_dir_len = 1.0f / sqrtf(dir_len_sq);
        const float speed = rng_range_float(&rng, VISUALIZER_BALL_MIN_SPEED, VISUALIZER_BALL_MAX_SPEED);
        h_balls[i].vx = dir_x * inv_dir_len * speed;
        h_balls[i].vy = dir_y * inv_dir_len * speed;
        h_balls[i].vz = dir_z * inv_dir_len * speed;
        h_balls[i].radius = radius;
        h_balls[i].colliding = 0;
    }
}

static void cpu_integrate(DeviceBall* balls, int count, float dt, int width, int height, float depth) {
    for (int i = 0; i < count; ++i) {
        DeviceBall* b = &balls[i];
        b->x += b->vx * dt;
        b->y += b->vy * dt;
        b->z += b->vz * dt;
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
        if (b->z < b->radius) {
            b->z = b->radius;
            b->vz = fabsf(b->vz);
        } else if (b->z > depth - b->radius) {
            b->z = depth - b->radius;
            b->vz = -fabsf(b->vz);
        }
        b->colliding = 0;
    }
}

static void cpu_apply_collision_response(DeviceBall* balls, int a_index, int b_index) {
    DeviceBall* a = &balls[a_index];
    DeviceBall* b = &balls[b_index];
    const float a_original_speed = sqrtf(a->vx * a->vx + a->vy * a->vy + a->vz * a->vz);
    const float b_original_speed = sqrtf(b->vx * b->vx + b->vy * b->vy + b->vz * b->vz);
    float nx = b->x - a->x;
    float ny = b->y - a->y;
    float nz = b->z - a->z;
    const float radius_sum = a->radius + b->radius;
    float distance_sq = nx * nx + ny * ny + nz * nz;

    if (distance_sq < 1.0e-6f) {
        nx = (a_index < b_index) ? 1.0f : -1.0f;
        ny = 0.0f;
        nz = 0.0f;
        distance_sq = 1.0f;
    }

    const float distance = sqrtf(distance_sq);
    nx /= distance;
    ny /= distance;
    nz /= distance;

    const float penetration = radius_sum - distance;
    if (penetration > 0.0f) {
        const float correction = penetration * 0.5f + 0.01f;
        a->x -= nx * correction;
        a->y -= ny * correction;
        a->z -= nz * correction;
        b->x += nx * correction;
        b->y += ny * correction;
        b->z += nz * correction;
    }

    const float rvx = a->vx - b->vx;
    const float rvy = a->vy - b->vy;
    const float rvz = a->vz - b->vz;
    const float relative_speed = rvx * nx + rvy * ny + rvz * nz;
    if (relative_speed <= 0.0f) {
        return;
    }

    a->vx -= relative_speed * nx;
    a->vy -= relative_speed * ny;
    a->vz -= relative_speed * nz;
    b->vx += relative_speed * nx;
    b->vy += relative_speed * ny;
    b->vz += relative_speed * nz;
    clamp_speed_after_collision(a, a_original_speed);
    clamp_speed_after_collision(b, b_original_speed);
}

static void cpu_clamp_to_bounds(DeviceBall* balls, int count, int width, int height, float depth) {
    for (int i = 0; i < count; ++i) {
        DeviceBall* b = &balls[i];
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
        if (b->z < b->radius) {
            b->z = b->radius;
            b->vz = fabsf(b->vz);
        } else if (b->z > depth - b->radius) {
            b->z = depth - b->radius;
            b->vz = -fabsf(b->vz);
        }
    }
}

static void cpu_apply_mouse_collision(
    DeviceBall* balls,
    int count,
    float mouse_x,
    float mouse_y,
    int mouse_active,
    int width,
    int height,
    float depth,
    const VisualizerCamera* camera) {
    if (!mouse_active) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        apply_mouse_collision(&balls[i], mouse_x, mouse_y, width, height, depth, camera);
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
        for (int j = i + 1; j < count; ++j) {
            ++candidates;
            if (balls_overlap_3d(&balls[i], &balls[j])) {
                balls[i].colliding = 1;
                balls[j].colliding = 1;
                cpu_apply_collision_response(balls, i, j);
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
        g_collision_delta_vx,
        g_collision_delta_vy,
        g_collision_delta_vz,
        g_collision_delta_x,
        g_collision_delta_y,
        g_collision_delta_z,
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
        VISUALIZER_GRID_CELL_SIZE,
        radius_profile_max_radius(g_radius_profile),
        g_collision_delta_vx,
        g_collision_delta_vy,
        g_collision_delta_vz,
        g_collision_delta_x,
        g_collision_delta_y,
        g_collision_delta_z,
        g_collision_count,
        g_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
}

static void run_uniform_grid_3d(int blocks, int threads) {
    const int cell_blocks = (g_total_cells + threads - 1) / threads;
    init_cell_ranges_kernel<<<cell_blocks, threads>>>(g_cell_start, g_cell_end, g_total_cells);
    CUDA_CHECK(cudaGetLastError());

    compute_cell_keys_3d_kernel<<<blocks, threads>>>(
        g_balls,
        g_cell_keys,
        g_indices,
        g_object_count,
        VISUALIZER_GRID_CELL_SIZE,
        g_grid_width,
        g_grid_height,
        g_grid_depth);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> keys_ptr(g_cell_keys);
    thrust::device_ptr<int> indices_ptr(g_indices);
    thrust::sort_by_key(keys_ptr, keys_ptr + g_object_count, indices_ptr);

    build_cell_ranges_kernel<<<blocks, threads>>>(g_cell_keys, g_cell_start, g_cell_end, g_object_count);
    CUDA_CHECK(cudaGetLastError());

    grid_collision_3d_kernel<<<blocks, threads>>>(
        g_balls,
        g_cell_keys,
        g_indices,
        g_cell_start,
        g_cell_end,
        g_object_count,
        g_grid_width,
        g_grid_height,
        g_grid_depth,
        VISUALIZER_GRID_CELL_SIZE,
        radius_profile_max_radius(g_radius_profile),
        g_collision_delta_vx,
        g_collision_delta_vy,
        g_collision_delta_vz,
        g_collision_delta_x,
        g_collision_delta_y,
        g_collision_delta_z,
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
        g_collision_delta_vx, g_collision_delta_vy, g_collision_delta_vz,
        g_collision_delta_x, g_collision_delta_y, g_collision_delta_z,
        g_collision_count, g_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
}

static void run_hash(int blocks, int threads) {
    const int bucket_blocks = (g_hash_total_buckets + threads - 1) / threads;

    init_cell_ranges_kernel<<<bucket_blocks, threads>>>(
        g_hash_cell_start, g_hash_cell_end, g_hash_total_buckets);
    CUDA_CHECK(cudaGetLastError());

    hash_assign_3d_kernel<<<blocks, threads>>>(
        g_balls,
        g_object_count,
        g_hash_base_cell_size,
        g_hash_num_levels,
        (unsigned int)g_hash_table_M,
        g_hash_composite_keys,
        g_hash_indices);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> keys_ptr(g_hash_composite_keys);
    thrust::device_ptr<int> indices_ptr(g_hash_indices);
    thrust::sort_by_key(keys_ptr, keys_ptr + g_object_count, indices_ptr);

    build_cell_ranges_kernel<<<blocks, threads>>>(
        g_hash_composite_keys, g_hash_cell_start, g_hash_cell_end, g_object_count);
    CUDA_CHECK(cudaGetLastError());

    hash_collide_3d_kernel<<<blocks, threads>>>(
        g_balls,
        g_hash_indices,
        g_hash_cell_start,
        g_hash_cell_end,
        g_object_count,
        g_hash_base_cell_size,
        g_hash_num_levels,
        (unsigned int)g_hash_table_M,
        g_collision_delta_vx, g_collision_delta_vy, g_collision_delta_vz,
        g_collision_delta_x, g_collision_delta_y, g_collision_delta_z,
        g_collision_count, g_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Pick HSH base cell size and level count from the active radius
 *        profile. Called from create() and from reset() so toggling the
 *        @c V key (radius profile) keeps the level partitioning in sync
 *        with the data without reallocating buffers.
 */
static void hash_configure_from_profile(RadiusProfile profile) {
    const float min_r = radius_profile_min_radius(profile);
    const float max_r = radius_profile_max_radius(profile);
    g_hash_base_cell_size = 2.0f * min_r;
    int L = (int)ceilf(log2f(max_r / min_r)) + 1;
    if (L < 1) L = 1;
    if (L > VIS_HASH_MAX_LEVELS) L = VIS_HASH_MAX_LEVELS;
    g_hash_num_levels = L;
}

/**
 * @brief Smallest prime >= @c lower_bound. Trial division (host).
 */
static unsigned int hash_next_prime_host(unsigned int lower_bound) {
    if (lower_bound <= 2u) return 2u;
    unsigned int n = (lower_bound % 2u == 0u) ? (lower_bound + 1u) : lower_bound;
    while (1) {
        int composite = 0;
        for (unsigned int d = 3u; d * d <= n; d += 2u) {
            if ((n % d) == 0u) { composite = 1; break; }
        }
        if (!composite) return n;
        n += 2u;
    }
}

extern "C" int cuda_visualizer_create(unsigned int vbo, int object_count, int width, int height) {
    g_object_count = object_count;
    g_width = width;
    g_height = height;
    g_depth = fminf((float)width, (float)height) * 0.75f;
    g_grid_width = (int)ceilf((float)width / VISUALIZER_GRID_CELL_SIZE);
    g_grid_height = (int)ceilf((float)height / VISUALIZER_GRID_CELL_SIZE);
    g_grid_depth = (int)ceilf(g_depth / VISUALIZER_GRID_CELL_SIZE);
    g_total_cells = g_grid_width * g_grid_height * g_grid_depth;

    CUDA_CHECK(cudaMalloc((void**)&g_balls, (size_t)object_count * sizeof(DeviceBall)));
    g_host_balls = (DeviceBall*)malloc((size_t)object_count * sizeof(DeviceBall));
    if (g_host_balls == NULL) {
        return 0;
    }
    CUDA_CHECK(cudaMalloc((void**)&g_collision_delta_vx, (size_t)object_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_collision_delta_vy, (size_t)object_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_collision_delta_vz, (size_t)object_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_collision_delta_x, (size_t)object_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_collision_delta_y, (size_t)object_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&g_collision_delta_z, (size_t)object_count * sizeof(float)));
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

    /* HSH: hash table sized for the visualizer's worst-case (max levels +
     * next_prime(2N+1)) so toggling radius profiles never requires
     * reallocation. Configure base/levels from the initial profile. */
    {
        const unsigned int target = (unsigned int)(2u * (unsigned int)object_count + 1u);
        g_hash_table_M = (int)hash_next_prime_host(target < 31u ? 31u : target);
        g_hash_total_buckets = VIS_HASH_MAX_LEVELS * g_hash_table_M;
        CUDA_CHECK(cudaMalloc((void**)&g_hash_composite_keys, (size_t)object_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&g_hash_indices, (size_t)object_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&g_hash_cell_start, (size_t)g_hash_total_buckets * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&g_hash_cell_end, (size_t)g_hash_total_buckets * sizeof(int)));
        hash_configure_from_profile(RADIUS_PROFILE_NARROW);
    }

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&g_vbo_resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaEventCreate(&g_start_event));
    CUDA_CHECK(cudaEventCreate(&g_stop_event));

    return cuda_visualizer_reset(0, RADIUS_PROFILE_NARROW);
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
    if (g_collision_delta_vx != NULL) {
        CUDA_CHECK(cudaFree(g_collision_delta_vx));
        g_collision_delta_vx = NULL;
    }
    if (g_collision_delta_vy != NULL) {
        CUDA_CHECK(cudaFree(g_collision_delta_vy));
        g_collision_delta_vy = NULL;
    }
    if (g_collision_delta_vz != NULL) {
        CUDA_CHECK(cudaFree(g_collision_delta_vz));
        g_collision_delta_vz = NULL;
    }
    if (g_collision_delta_x != NULL) {
        CUDA_CHECK(cudaFree(g_collision_delta_x));
        g_collision_delta_x = NULL;
    }
    if (g_collision_delta_y != NULL) {
        CUDA_CHECK(cudaFree(g_collision_delta_y));
        g_collision_delta_y = NULL;
    }
    if (g_collision_delta_z != NULL) {
        CUDA_CHECK(cudaFree(g_collision_delta_z));
        g_collision_delta_z = NULL;
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
    if (g_hash_composite_keys != NULL) {
        CUDA_CHECK(cudaFree(g_hash_composite_keys));
        g_hash_composite_keys = NULL;
    }
    if (g_hash_indices != NULL) {
        CUDA_CHECK(cudaFree(g_hash_indices));
        g_hash_indices = NULL;
    }
    if (g_hash_cell_start != NULL) {
        CUDA_CHECK(cudaFree(g_hash_cell_start));
        g_hash_cell_start = NULL;
    }
    if (g_hash_cell_end != NULL) {
        CUDA_CHECK(cudaFree(g_hash_cell_end));
        g_hash_cell_end = NULL;
    }
    g_hash_table_M = 0;
    g_hash_total_buckets = 0;
    g_hash_num_levels = 0;
    g_hash_base_cell_size = 0.0f;
    if (g_start_event != NULL) {
        CUDA_CHECK(cudaEventDestroy(g_start_event));
        g_start_event = NULL;
    }
    if (g_stop_event != NULL) {
        CUDA_CHECK(cudaEventDestroy(g_stop_event));
        g_stop_event = NULL;
    }
}

extern "C" int cuda_visualizer_reset(int clustered, RadiusProfile radius_profile) {
    if (g_balls == NULL) {
        return 0;
    }
    g_radius_profile = radius_profile;
    hash_configure_from_profile(radius_profile);

    DeviceBall* h_balls = (DeviceBall*)malloc((size_t)g_object_count * sizeof(DeviceBall));
    if (h_balls == NULL) {
        return 0;
    }
    host_init_balls(h_balls, g_object_count, clustered, radius_profile, VISUALIZER_INIT_SEED);
    CUDA_CHECK(cudaMemcpy(g_balls, h_balls, (size_t)g_object_count * sizeof(DeviceBall), cudaMemcpyHostToDevice));
    free(h_balls);
    return 1;
}

extern "C" int cuda_visualizer_set_mode(int mode) {
    if (mode != VISUALIZER_MODE_CUDA_BRUTE_FORCE
        && mode != VISUALIZER_MODE_CUDA_UNIFORM_GRID
        && mode != VISUALIZER_MODE_CPU_BRUTE_FORCE
        && mode != VISUALIZER_MODE_CUDA_LBVH
        && mode != VISUALIZER_MODE_CUDA_UNIFORM_GRID_3D
        && mode != VISUALIZER_MODE_CUDA_HASH) {
        return 0;
    }
    g_mode = mode;
    return 1;
}

extern "C" int cuda_visualizer_step(
    float dt,
    float mouse_x,
    float mouse_y,
    int mouse_active,
    const VisualizerCamera* camera,
    VisualizerMetrics* metrics) {
    if (g_balls == NULL || g_vbo_resource == NULL || metrics == NULL || camera == NULL) {
        return 0;
    }
    const VisualizerCamera camera_value = *camera;

    const int threads = 256;
    const int blocks = (g_object_count + threads - 1) / threads;
    RenderVertex* vertices = NULL;
    size_t mapped_size = 0;

    if (g_mode == VISUALIZER_MODE_CPU_BRUTE_FORCE) {
        CUDA_CHECK(cudaMemcpy(g_host_balls, g_balls,
                              (size_t)g_object_count * sizeof(DeviceBall),
                              cudaMemcpyDeviceToHost));

        const double t0 = timer_now_ms();
        cpu_integrate(g_host_balls, g_object_count, dt, g_width, g_height, g_depth);
        unsigned long long h_collisions = 0;
        unsigned long long h_candidates = 0;
        cpu_brute_force_collide(g_host_balls, g_object_count, &h_collisions, &h_candidates);
        cpu_clamp_to_bounds(g_host_balls, g_object_count, g_width, g_height, g_depth);
        cpu_apply_mouse_collision(
            g_host_balls,
            g_object_count,
            mouse_x,
            mouse_y,
            mouse_active,
            g_width,
            g_height,
            g_depth,
            &camera_value);
        const double t1 = timer_now_ms();

        CUDA_CHECK(cudaMemcpy(g_balls, g_host_balls,
                              (size_t)g_object_count * sizeof(DeviceBall),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaGraphicsMapResources(1, &g_vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertices, &mapped_size, g_vbo_resource));
        write_vbo_kernel<<<blocks, threads>>>(
            g_balls, vertices, g_object_count, g_width, g_height, g_depth, camera_value);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &g_vbo_resource, 0));

        metrics->collision_count = h_collisions;
        metrics->candidate_pair_count = h_candidates;
        metrics->gpu_time_ms = (float)(t1 - t0);
        return 1;
    }

    CUDA_CHECK(cudaMemset(g_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_collision_delta_vx, 0, (size_t)g_object_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_collision_delta_vy, 0, (size_t)g_object_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_collision_delta_vz, 0, (size_t)g_object_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_collision_delta_x, 0, (size_t)g_object_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_collision_delta_y, 0, (size_t)g_object_count * sizeof(float)));
    CUDA_CHECK(cudaMemset(g_collision_delta_z, 0, (size_t)g_object_count * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(g_start_event));

    integrate_kernel<<<blocks, threads>>>(g_balls, g_object_count, dt, g_width, g_height, g_depth);
    CUDA_CHECK(cudaGetLastError());

    if (g_mode == VISUALIZER_MODE_CUDA_UNIFORM_GRID) {
        run_uniform_grid(blocks, threads);
    } else if (g_mode == VISUALIZER_MODE_CUDA_UNIFORM_GRID_3D) {
        run_uniform_grid_3d(blocks, threads);
    } else if (g_mode == VISUALIZER_MODE_CUDA_LBVH) {
        run_lbvh(blocks, threads);
    } else if (g_mode == VISUALIZER_MODE_CUDA_HASH) {
        run_hash(blocks, threads);
    } else {
        run_brute_force(blocks, threads);
    }

    apply_collision_response_kernel<<<blocks, threads>>>(
        g_balls,
        g_object_count,
        g_collision_delta_vx,
        g_collision_delta_vy,
        g_collision_delta_vz,
        g_collision_delta_x,
        g_collision_delta_y,
        g_collision_delta_z,
        g_width,
        g_height,
        g_depth);
    CUDA_CHECK(cudaGetLastError());

    mouse_collision_kernel<<<blocks, threads>>>(
        g_balls,
        g_object_count,
        mouse_x,
        mouse_y,
        mouse_active,
        g_width,
        g_height,
        g_depth,
        camera_value);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaGraphicsMapResources(1, &g_vbo_resource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertices, &mapped_size, g_vbo_resource));
    write_vbo_kernel<<<blocks, threads>>>(
        g_balls, vertices, g_object_count, g_width, g_height, g_depth, camera_value);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &g_vbo_resource, 0));

    CUDA_CHECK(cudaEventRecord(g_stop_event));
    CUDA_CHECK(cudaEventSynchronize(g_stop_event));
    CUDA_CHECK(cudaMemcpy(&metrics->collision_count, g_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&metrics->candidate_pair_count, g_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&metrics->gpu_time_ms, g_start_event, g_stop_event));

    return 1;
}
