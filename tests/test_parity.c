#include "Benchmark.h"
#include "CpuCollision.h"
#include "CudaBruteForce.cuh"
#include "CudaGrid.cuh"
#include "CudaLbvh.cuh"
#include "DataGenerator.h"

#include <stdio.h>
#include <stdlib.h>

static int counts_match(unsigned long long expected, unsigned long long actual) {
    unsigned long long diff = expected > actual ? expected - actual : actual - expected;
    unsigned long long tolerance = expected / 1000u;
    if (tolerance < 1u) {
        tolerance = 1u;
    }
    return diff <= tolerance;
}

static int check_method(
    const char* case_name,
    const char* method_name,
    unsigned long long expected,
    CollisionResult actual) {
    if (!counts_match(expected, actual.collision_count)) {
        fprintf(stderr,
                "%s/%s mismatch: expected %llu collisions, got %llu\n",
                case_name,
                method_name,
                expected,
                actual.collision_count);
        return 0;
    }

    printf("PASS %-18s %-16s collisions=%llu candidates=%llu total=%.3f ms\n",
           case_name,
           method_name,
           actual.collision_count,
           actual.candidate_pair_count,
           actual.total_time_ms);
    return 1;
}

static int run_case(const char* case_name, const Circle* circles, size_t count, const BenchmarkConfig* config) {
    CudaGridParams grid_params;
    grid_params.scene_width = config->scene_width;
    grid_params.scene_height = config->scene_height;
    grid_params.cell_size = 10.0f;
    grid_params.max_radius = config->max_radius;
    grid_params.dense_cell_threshold = config->dense_cell_threshold;

    CudaLbvhParams lbvh_params;
    lbvh_params.scene_width = config->scene_width;
    lbvh_params.scene_height = config->scene_height;

    const CollisionResult cpu = run_cpu_brute_force(circles, count, NULL);
    int ok = 1;
    ok = check_method(case_name, "cpu_brute_force", cpu.collision_count, cpu) && ok;
    ok = check_method(case_name, "cuda_brute_force", cpu.collision_count,
                      run_cuda_brute_force(circles, count, NULL)) && ok;
    ok = check_method(case_name, "cuda_uniform_grid", cpu.collision_count,
                      run_cuda_uniform_grid(circles, count, &grid_params)) && ok;
    ok = check_method(case_name, "cuda_lbvh", cpu.collision_count,
                      run_cuda_lbvh(circles, count, &lbvh_params)) && ok;
    return ok;
}

static int run_generated_case(const char* name, int clustered, size_t count, const BenchmarkConfig* config) {
    Circle* circles = clustered ? generate_clustered_circles(count, config)
                                : generate_uniform_circles(count, config);
    if (circles == NULL) {
        fprintf(stderr, "Failed to allocate %s case.\n", name);
        return 0;
    }

    const int ok = run_case(name, circles, count, config);
    free(circles);
    return ok;
}

int main(void) {
    BenchmarkConfig config = {0};
    config.scene_width = 1000.0f;
    config.scene_height = 1000.0f;
    config.min_radius = 1.0f;
    config.max_radius = 2.0f;
    config.cluster_count = 4;
    config.cluster_spread = 60.0f;
    config.dense_cell_threshold = 32;
    config.seed = 674u;

    const Circle tiny[] = {
        {10.0f, 10.0f, 2.0f},
        {13.0f, 10.0f, 2.0f},
        {30.0f, 30.0f, 2.0f},
        {30.5f, 32.0f, 2.0f},
        {80.0f, 80.0f, 1.0f}
    };

    int ok = 1;
    ok = run_case("tiny_known", tiny, sizeof(tiny) / sizeof(tiny[0]), &config) && ok;
    ok = run_generated_case("uniform_100", 0, 100u, &config) && ok;
    ok = run_generated_case("clustered_1000", 1, 1000u, &config) && ok;
    ok = run_generated_case("uniform_5000", 0, 5000u, &config) && ok;
    ok = run_generated_case("clustered_5000", 1, 5000u, &config) && ok;

    if (!ok) {
        fprintf(stderr, "Collision parity test failed.\n");
        return 1;
    }

    printf("Collision parity test passed.\n");
    return 0;
}
