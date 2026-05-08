#pragma once

#include "GridStats.h"
#include "RadiusProfile.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_BENCHMARK_OBJECT_COUNTS 5
#define MAX_GRID_CELL_SIZES 4
#define MAX_RADIUS_PROFILES 3

typedef struct BenchmarkConfig {
    size_t object_counts[MAX_BENCHMARK_OBJECT_COUNTS];
    int object_count_len;
    float grid_cell_sizes[MAX_GRID_CELL_SIZES];
    int grid_cell_size_len;
    RadiusProfile radius_profiles[MAX_RADIUS_PROFILES];
    int radius_profile_len;
    RadiusProfile active_radius_profile;
    float scene_width;
    float scene_height;
    float min_radius;
    float max_radius;
    int cluster_count;
    float cluster_spread;
    int dense_cell_threshold;
    unsigned int seed;
    const char* output_csv_path;
} BenchmarkConfig;

typedef struct BenchmarkResult {
    size_t object_count;
    const char* distribution_type;
    const char* radius_profile;
    const char* method_name;
    unsigned long long collision_count;
    unsigned long long candidate_pair_count;
    double kernel_time_ms;
    double total_time_ms;
    double speedup_vs_cpu;
    float grid_cell_size;
    int has_grid_stats;
    GridStats grid_stats;
} BenchmarkResult;

BenchmarkConfig benchmark_default_config(void);
int run_benchmarks(const BenchmarkConfig* config);

#ifdef __cplusplus
}
#endif
