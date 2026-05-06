#include "Benchmark.h"
#include "DataGenerator.h"
#include "CpuBruteForceDetector.h"
#include "CudaBruteForceDetector.h"
#include "CudaGridDetector.h"
#include "CudaLBVHDetector.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <vector>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

static bool ensure_results_directory() {
    if (MKDIR("results") == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

static void write_csv_header(std::ofstream& file) {
    file << "object_count,distribution_type,method_name,collision_count,"
         << "candidate_pair_count,total_time_ms,memory_transfer_time_ms,"
         << "kernel_execution_time_ms,speedup_vs_cpu,grid_cell_size,"
         << "max_objects_in_cell,avg_objects_per_non_empty_cell,dense_cell_count\n";
}

static void write_csv_result(std::ofstream& file, const BenchmarkResult& result) {
    file << result.object_count << ","
         << result.distribution_type << ","
         << result.method_name << ","
         << result.collision_count << ","
         << result.candidate_pair_count << ","
         << std::fixed << std::setprecision(6) << result.execution_time_ms << ","
         << result.memory_transfer_time_ms << ","
         << result.kernel_execution_time_ms << ","
         << result.speedup_vs_cpu << ",";

    if (result.method_name == std::string("cuda_uniform_grid")) {
        file << result.grid_cell_size << ","
             << result.grid_stats.max_objects_in_cell << ","
             << result.grid_stats.avg_objects_per_non_empty_cell << ","
             << result.grid_stats.dense_cell_count << "\n";
    } else {
        file << "0,0,0,0\n";
    }
}

BenchmarkConfig benchmark_default_config(void) {
    BenchmarkConfig config = {};
    config.object_counts[0] = 1000;
    config.object_counts[1] = 5000;
    config.object_counts[2] = 10000;
    config.object_counts[3] = 50000;
    config.object_counts[4] = 100000;
    config.object_count_len = 5;

    config.grid_cell_sizes[0] = 5.0f;
    config.grid_cell_sizes[1] = 10.0f;
    config.grid_cell_sizes[2] = 20.0f;
    config.grid_cell_sizes[3] = 40.0f;
    config.grid_cell_size_len = 4;

    config.scene_width = 1000.0f;
    config.scene_height = 1000.0f;
    config.min_radius = 1.0f;
    config.max_radius = 2.0f;
    config.cluster_count = 4;
    config.cluster_spread = 60.0f;
    config.dense_cell_threshold = 128;
    config.seed = 42;
    config.output_csv_path = "results/timings.csv";

    return config;
}

static bool run_distribution(
    std::ofstream& csv,
    const BenchmarkConfig& config,
    size_t object_count,
    const std::string& distribution_name,
    const CirclesSoA& circles) 
{
    std::cout << "Running " << object_count << " objects, " << distribution_name << " distribution...\n";

    CpuBruteForceDetector cpu_detector;
    cpu_detector.update_data(circles);
    CollisionResult cpu_res = cpu_detector.run_detection();

    BenchmarkResult row = {};
    row.object_count = object_count;
    row.distribution_type = distribution_name.c_str();
    row.method_name = "cpu_brute_force";
    row.collision_count = cpu_res.collision_count;
    row.candidate_pair_count = cpu_res.candidate_pair_count;
    row.execution_time_ms = cpu_res.total_time_ms;
    row.memory_transfer_time_ms = cpu_res.memory_transfer_time_ms;
    row.kernel_execution_time_ms = cpu_res.kernel_execution_time_ms;
    row.speedup_vs_cpu = 1.0;
    write_csv_result(csv, row);
    
    std::cout << "  CPU brute force: collisions=" << cpu_res.collision_count 
              << " time=" << cpu_res.total_time_ms << " ms\n";

    // CUDA Brute Force
    CudaBruteForceDetector cuda_brute;
    cuda_brute.update_data(circles);
    CollisionResult cuda_brute_res = cuda_brute.run_detection();
    
    row.method_name = "cuda_brute_force";
    row.collision_count = cuda_brute_res.collision_count;
    row.candidate_pair_count = cuda_brute_res.candidate_pair_count;
    row.execution_time_ms = cuda_brute_res.total_time_ms;
    row.memory_transfer_time_ms = cuda_brute_res.memory_transfer_time_ms;
    row.kernel_execution_time_ms = cuda_brute_res.kernel_execution_time_ms;
    row.speedup_vs_cpu = cpu_res.total_time_ms / cuda_brute_res.total_time_ms;
    write_csv_result(csv, row);

    std::cout << "  CUDA brute force: collisions=" << cuda_brute_res.collision_count 
              << " time=" << cuda_brute_res.total_time_ms << " ms (Kernel: " << cuda_brute_res.kernel_execution_time_ms << " ms)\n";

    // CUDA LBVH
    CudaLBVHDetector lbvh;
    lbvh.set_scene_bounds(config.scene_width, config.scene_height);
    lbvh.update_data(circles);
    CollisionResult lbvh_res = lbvh.run_detection();

    row.method_name = "cuda_lbvh";
    row.collision_count = lbvh_res.collision_count;
    row.candidate_pair_count = lbvh_res.candidate_pair_count;
    row.execution_time_ms = lbvh_res.total_time_ms;
    row.memory_transfer_time_ms = lbvh_res.memory_transfer_time_ms;
    row.kernel_execution_time_ms = lbvh_res.kernel_execution_time_ms;
    row.speedup_vs_cpu = cpu_res.total_time_ms / lbvh_res.total_time_ms;
    write_csv_result(csv, row);

    std::cout << "  CUDA LBVH: collisions=" << lbvh_res.collision_count 
              << " time=" << lbvh_res.total_time_ms << " ms (Kernel: " << lbvh_res.kernel_execution_time_ms << " ms)\n";


    // CUDA Grid (varying cell sizes)
    for (int i = 0; i < config.grid_cell_size_len; ++i) {
        float cell_size = config.grid_cell_sizes[i];
        CudaGridDetector grid;
        grid.set_grid_params(config.scene_width, config.scene_height, cell_size, config.dense_cell_threshold);
        grid.update_data(circles);
        CollisionResult grid_res = grid.run_detection();

        row.method_name = "cuda_uniform_grid";
        row.collision_count = grid_res.collision_count;
        row.candidate_pair_count = grid_res.candidate_pair_count;
        row.execution_time_ms = grid_res.total_time_ms;
        row.memory_transfer_time_ms = grid_res.memory_transfer_time_ms;
        row.kernel_execution_time_ms = grid_res.kernel_execution_time_ms;
        row.speedup_vs_cpu = cpu_res.total_time_ms / grid_res.total_time_ms;
        row.grid_cell_size = cell_size;
        row.grid_stats.max_objects_in_cell = grid_res.extra_stats["max_objects_in_cell"];
        row.grid_stats.avg_objects_per_non_empty_cell = grid_res.extra_stats["avg_objects_per_non_empty_cell"];
        row.grid_stats.dense_cell_count = grid_res.extra_stats["dense_cell_count"];
        write_csv_result(csv, row);

        std::cout << "  CUDA Grid (cell=" << cell_size << "): collisions=" << grid_res.collision_count 
                  << " time=" << grid_res.total_time_ms << " ms (Kernel: " << grid_res.kernel_execution_time_ms << " ms)\n";
    }

    csv.flush();
    return true;
}

int run_benchmarks(const BenchmarkConfig* config) {
    if (!config) return 0;
    if (!ensure_results_directory()) {
        std::cerr << "Failed to create results directory.\n";
        return 0;
    }

    std::ofstream csv(config->output_csv_path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV output: " << config->output_csv_path << "\n";
        return 0;
    }

    write_csv_header(csv);

    for (int i = 0; i < config->object_count_len; ++i) {
        size_t object_count = config->object_counts[i];

        CirclesSoA uniform = generate_uniform_circles(object_count, *config);
        if (!run_distribution(csv, *config, object_count, "uniform", uniform)) return 0;

        CirclesSoA clustered = generate_clustered_circles(object_count, *config);
        if (!run_distribution(csv, *config, object_count, "clustered", clustered)) return 0;
    }

    std::cout << "Benchmark complete. CSV written to " << config->output_csv_path << "\n";
    return 1;
}
