#pragma once

#include "RadiusProfile.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VisualizerMode {
    VISUALIZER_MODE_CUDA_BRUTE_FORCE = 0,
    VISUALIZER_MODE_CUDA_UNIFORM_GRID = 1,
    VISUALIZER_MODE_CPU_BRUTE_FORCE = 2,
    VISUALIZER_MODE_CUDA_LBVH = 3,
    VISUALIZER_MODE_CUDA_UNIFORM_GRID_3D = 4,
    VISUALIZER_MODE_CUDA_HASH = 5
} VisualizerMode;

typedef struct RenderVertex {
    float x;
    float y;
    float depth;
    float local_x;
    float local_y;
    float r;
    float g;
    float b;
} RenderVertex;

typedef struct VisualizerMetrics {
    unsigned long long collision_count;
    unsigned long long candidate_pair_count;
    /* Whole simulation step on the device: integrate + broad-phase + collision
     * response + mouse + VBO write (CPU mode: integrate + collide + clamp +
     * mouse on the host). GL interop map/unmap included. */
    float gpu_time_ms;
    /* Broad-phase only: build + query for the selected method (CPU mode: just
     * the brute-force collide pass). Lets the methods be compared without the
     * fixed per-frame physics/render overhead muddying the numbers. */
    float broadphase_ms;
} VisualizerMetrics;

typedef struct VisualizerCamera {
    float yaw;
    float pitch;
    float distance;
} VisualizerCamera;

int cuda_visualizer_create(unsigned int vbo, int object_count, int width, int height);
void cuda_visualizer_destroy(void);
int cuda_visualizer_reset(int clustered, RadiusProfile radius_profile);
int cuda_visualizer_set_mode(int mode);
int cuda_visualizer_step(
    float dt,
    float mouse_x,
    float mouse_y,
    int mouse_active,
    const VisualizerCamera* camera,
    VisualizerMetrics* metrics);

#ifdef __cplusplus
}
#endif
