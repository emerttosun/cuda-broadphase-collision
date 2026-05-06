#pragma once

#include <cuda_runtime.h>

// Unified math operations to prevent code duplication between CPU and GPU

__device__ __host__ __forceinline__ bool circles_collide(float ax, float ay, float ar, float bx, float by, float br) {
    const float dx = ax - bx;
    const float dy = ay - by;
    const float radius_sum = ar + br;
    const float distance_squared = dx * dx + dy * dy;
    return distance_squared <= radius_sum * radius_sum;
}
