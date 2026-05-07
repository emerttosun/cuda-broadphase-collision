#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VisualizerMode {
    VISUALIZER_MODE_CUDA_BRUTE_FORCE = 0,
    VISUALIZER_MODE_CUDA_UNIFORM_GRID = 1,
    VISUALIZER_MODE_CPU_BRUTE_FORCE = 2,
    VISUALIZER_MODE_CUDA_LBVH = 3,
    VISUALIZER_MODE_CUDA_UNIFORM_GRID_3D = 4
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
    float gpu_time_ms;
} VisualizerMetrics;

int cuda_visualizer_create(unsigned int vbo, int object_count, int width, int height);
void cuda_visualizer_destroy(void);
int cuda_visualizer_reset(int clustered);
int cuda_visualizer_set_mode(int mode);
int cuda_visualizer_step(float dt, float mouse_x, float mouse_y, int mouse_active, VisualizerMetrics* metrics);

#ifdef __cplusplus
}
#endif
