#pragma once

#include <cstddef>
#include <string>

#define MAX_BENCHMARK_OBJECT_COUNTS 5
#define MAX_GRID_CELL_SIZES 4

struct GridStats {
    int max_objects_in_cell;
    double avg_objects_per_non_empty_cell;
    int dense_cell_count;
};

struct BenchmarkConfig {
    size_t object_counts[MAX_BENCHMARK_OBJECT_COUNTS];
    int object_count_len;
    float grid_cell_sizes[MAX_GRID_CELL_SIZES];
    int grid_cell_size_len;
    float scene_width;
    float scene_height;
    float min_radius;
    float max_radius;
    int cluster_count;
    float cluster_spread;
    int dense_cell_threshold;
    unsigned int seed;
    const char* output_csv_path;
};

struct BenchmarkResult {
    size_t object_count;
    const char* distribution_type;
    const char* method_name;
    unsigned long long collision_count;
    unsigned long long candidate_pair_count;
    double execution_time_ms;
    double memory_transfer_time_ms;
    double kernel_execution_time_ms;
    double speedup_vs_cpu;
    float grid_cell_size;
    GridStats grid_stats;
};

BenchmarkConfig benchmark_default_config();
int run_benchmarks(const BenchmarkConfig* config);
