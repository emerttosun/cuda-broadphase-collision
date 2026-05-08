#pragma once

#include "Broadphase.h"
#include "Circle.h"
#include "CollisionResult.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CudaLbvhParams {
    float scene_width;
    float scene_height;
} CudaLbvhParams;

CollisionResult run_cuda_lbvh(const Circle* circles, size_t count, const void* params);

#ifdef __cplusplus
}
#endif
