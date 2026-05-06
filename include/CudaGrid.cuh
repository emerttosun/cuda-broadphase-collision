#pragma once

#include "Broadphase.h"
#include "Circle.h"
#include "CollisionResult.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

CollisionResult run_cuda_uniform_grid(const Circle* circles, size_t count, const void* params);

#ifdef __cplusplus
}
#endif
