#pragma once

#include "Circle.h"

#if defined(__CUDACC__)
#define CMP674_HD_INLINE static __device__ __host__ inline
#else
#define CMP674_HD_INLINE static inline
#endif

CMP674_HD_INLINE int circles_overlap(const Circle* a, const Circle* b) {
    const float dx = a->x - b->x;
    const float dy = a->y - b->y;
    const float radius_sum = a->radius + b->radius;
    const float distance_squared = dx * dx + dy * dy;
    return distance_squared <= radius_sum * radius_sum;
}
