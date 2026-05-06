#include "DataGenerator.h"
#include "Rng.h"

#include <math.h>
#include <stdlib.h>

static float clamp_float(float value, float min_value, float max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

Circle* generate_uniform_circles(size_t count, const BenchmarkConfig* config) {
    Circle* circles = (Circle*)malloc(count * sizeof(Circle));
    if (circles == NULL) {
        return NULL;
    }

    unsigned int rng = config->seed ^ (unsigned int)count;
    for (size_t i = 0; i < count; ++i) {
        circles[i].x = rng_range_float(&rng, 0.0f, config->scene_width);
        circles[i].y = rng_range_float(&rng, 0.0f, config->scene_height);
        circles[i].radius = rng_range_float(&rng, config->min_radius, config->max_radius);
    }

    return circles;
}

Circle* generate_clustered_circles(size_t count, const BenchmarkConfig* config) {
    Circle* circles = (Circle*)malloc(count * sizeof(Circle));
    if (circles == NULL) {
        return NULL;
    }

    int cluster_count = config->cluster_count > 0 ? config->cluster_count : 4;
    if (cluster_count > 8) {
        cluster_count = 8;
    }
    Circle centers[8];
    unsigned int rng = config->seed ^ 0x9E3779B9u ^ (unsigned int)count;

    for (int i = 0; i < cluster_count && i < 8; ++i) {
        centers[i].x = rng_range_float(&rng, config->scene_width * 0.15f, config->scene_width * 0.85f);
        centers[i].y = rng_range_float(&rng, config->scene_height * 0.15f, config->scene_height * 0.85f);
        centers[i].radius = 0.0f;
    }

    for (size_t i = 0; i < count; ++i) {
        const int center_index = (int)(rng_next_u32(&rng) % (unsigned int)cluster_count);
        const float angle = rng_range_float(&rng, 0.0f, 6.28318530718f);
        const float distance = rng_range_float(&rng, 0.0f, config->cluster_spread);

        circles[i].x = clamp_float(centers[center_index].x + cosf(angle) * distance, 0.0f, config->scene_width);
        circles[i].y = clamp_float(centers[center_index].y + sinf(angle) * distance, 0.0f, config->scene_height);
        circles[i].radius = rng_range_float(&rng, config->min_radius, config->max_radius);
    }

    return circles;
}
