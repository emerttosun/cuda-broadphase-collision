#pragma once

#include "Circle.h"
#include "CollisionResult.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef CollisionResult (*BroadphaseFn)(
    const Circle* circles,
    size_t count,
    const void* params);

typedef struct BroadphaseMethod {
    const char* name;
    BroadphaseFn run;
    const void* params;
    float reported_grid_cell_size;
} BroadphaseMethod;

typedef struct CudaGridParams {
    float scene_width;
    float scene_height;
    float cell_size;
    float max_radius;
    int dense_cell_threshold;
} CudaGridParams;

#ifdef __cplusplus
}
#endif
