#pragma once

#include "GridStats.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CollisionResult {
    unsigned long long collision_count;
    unsigned long long candidate_pair_count;
    double kernel_time_ms;
    double total_time_ms;
    int has_grid_stats;
    GridStats grid_stats;
} CollisionResult;

#ifdef __cplusplus
}
#endif
