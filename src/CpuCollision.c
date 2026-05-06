#include "CpuCollision.h"
#include "CollisionMath.h"
#include "Timer.h"

#include <string.h>

CollisionResult run_cpu_brute_force(const Circle* circles, size_t count, const void* params) {
    (void)params;

    CollisionResult result;
    memset(&result, 0, sizeof(result));

    if (circles == NULL || count == 0) {
        return result;
    }

    const double start_ms = timer_now_ms();
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            result.candidate_pair_count++;
            if (circles_overlap(&circles[i], &circles[j])) {
                result.collision_count++;
            }
        }
    }
    const double end_ms = timer_now_ms();

    result.kernel_time_ms = end_ms - start_ms;
    result.total_time_ms = result.kernel_time_ms;
    return result;
}
