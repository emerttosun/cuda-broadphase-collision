#include "CpuBruteForceDetector.h"
#include "CollisionMath.cuh"
#include <chrono>

void CpuBruteForceDetector::update_data(const CirclesSoA& circles) {
    m_circles = &circles;
}

CollisionResult CpuBruteForceDetector::run_detection() {
    CollisionResult result;
    if (!m_circles || m_circles->count == 0) return result;

    const size_t count = m_circles->count;
    const float* x = m_circles->host_x.data();
    const float* y = m_circles->host_y.data();
    const float* r = m_circles->host_radius.data();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            result.candidate_pair_count++;
            if (circles_collide(x[i], y[i], r[i], x[j], y[j], r[j])) {
                result.collision_count++;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    result.kernel_execution_time_ms = elapsed.count();
    result.total_time_ms = result.kernel_execution_time_ms; // CPU has 0 transfer time

    return result;
}
