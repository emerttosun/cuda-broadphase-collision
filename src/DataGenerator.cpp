#include "DataGenerator.h"
#include <cmath>
#include <random>

static float clamp_float(float value, float min_value, float max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

CirclesSoA generate_uniform_circles(size_t count, const BenchmarkConfig& config) {
    CirclesSoA circles;
    circles.resize(count);

    std::mt19937 rng(config.seed ^ static_cast<unsigned int>(count));
    std::uniform_real_distribution<float> dist_x(0.0f, config.scene_width);
    std::uniform_real_distribution<float> dist_y(0.0f, config.scene_height);
    std::uniform_real_distribution<float> dist_r(config.min_radius, config.max_radius);

    for (size_t i = 0; i < count; ++i) {
        circles.host_x[i] = dist_x(rng);
        circles.host_y[i] = dist_y(rng);
        circles.host_radius[i] = dist_r(rng);
    }

    return circles;
}

CirclesSoA generate_clustered_circles(size_t count, const BenchmarkConfig& config) {
    CirclesSoA circles;
    circles.resize(count);

    int cluster_count = config.cluster_count > 0 ? config.cluster_count : 4;
    if (cluster_count > 8) cluster_count = 8;
    
    struct Center { float x, y; };
    Center centers[8];
    
    std::mt19937 rng(config.seed ^ 0x9E3779B9u ^ static_cast<unsigned int>(count));
    std::uniform_real_distribution<float> dist_cx(config.scene_width * 0.15f, config.scene_width * 0.85f);
    std::uniform_real_distribution<float> dist_cy(config.scene_height * 0.15f, config.scene_height * 0.85f);

    for (int i = 0; i < cluster_count && i < 8; ++i) {
        centers[i].x = dist_cx(rng);
        centers[i].y = dist_cy(rng);
    }

    std::uniform_int_distribution<int> dist_cluster(0, cluster_count - 1);
    std::uniform_real_distribution<float> dist_angle(0.0f, 6.28318530718f);
    std::uniform_real_distribution<float> dist_distance(0.0f, config.cluster_spread);
    std::uniform_real_distribution<float> dist_r(config.min_radius, config.max_radius);

    for (size_t i = 0; i < count; ++i) {
        int center_index = dist_cluster(rng);
        float angle = dist_angle(rng);
        float distance = dist_distance(rng);

        circles.host_x[i] = clamp_float(centers[center_index].x + std::cos(angle) * distance, 0.0f, config.scene_width);
        circles.host_y[i] = clamp_float(centers[center_index].y + std::sin(angle) * distance, 0.0f, config.scene_height);
        circles.host_radius[i] = dist_r(rng);
    }

    return circles;
}
