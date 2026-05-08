#include "Benchmark.h"
#include "Broadphase.h"
#include "CpuCollision.h"
#include "CudaBruteForce.cuh"
#include "CudaGrid.cuh"
#include "CudaLbvh.cuh"
#include "DataGenerator.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif

#define MAX_METHODS_PER_DISTRIBUTION (3 + MAX_GRID_CELL_SIZES)

static int ensure_results_directory(void) {
#ifdef _WIN32
    if (_mkdir("results") == 0 || errno == EEXIST) {
        return 1;
    }
#else
    if (mkdir("results", 0755) == 0 || errno == EEXIST) {
        return 1;
    }
#endif
    return 0;
}

static void write_csv_header(FILE* file) {
    fprintf(
        file,
        "object_count,distribution_type,radius_profile,method_name,collision_count,"
        "candidate_pair_count,kernel_time_ms,total_time_ms,speedup_vs_cpu,grid_cell_size,"
        "max_objects_in_cell,avg_objects_per_non_empty_cell,dense_cell_count\n");
}

static void write_csv_result(FILE* file, const BenchmarkResult* row) {
    fprintf(
        file,
        "%zu,%s,%s,%s,%llu,%llu,%.6f,%.6f,%.6f,%.2f,%d,%.6f,%d\n",
        row->object_count,
        row->distribution_type,
        row->radius_profile,
        row->method_name,
        row->collision_count,
        row->candidate_pair_count,
        row->kernel_time_ms,
        row->total_time_ms,
        row->speedup_vs_cpu,
        row->grid_cell_size,
        row->has_grid_stats ? row->grid_stats.max_objects_in_cell : 0,
        row->has_grid_stats ? row->grid_stats.avg_objects_per_non_empty_cell : 0.0,
        row->has_grid_stats ? row->grid_stats.dense_cell_count : 0);
}

static double speedup_from_cpu(double cpu_total_ms, double method_total_ms) {
    if (method_total_ms <= 0.0) {
        return 0.0;
    }
    return cpu_total_ms / method_total_ms;
}

static int build_methods(
    const BenchmarkConfig* config,
    BroadphaseMethod* methods,
    CudaGridParams* grid_params_storage,
    CudaLbvhParams* lbvh_params_storage,
    int max_methods) {
    int n = 0;
    if (n >= max_methods) {
        return n;
    }
    methods[n].name = "cpu_brute_force";
    methods[n].run = run_cpu_brute_force;
    methods[n].params = NULL;
    methods[n].reported_grid_cell_size = 0.0f;
    n++;

    if (n >= max_methods) {
        return n;
    }
    methods[n].name = "cuda_brute_force";
    methods[n].run = run_cuda_brute_force;
    methods[n].params = NULL;
    methods[n].reported_grid_cell_size = 0.0f;
    n++;

    for (int i = 0; i < config->grid_cell_size_len && n < max_methods; ++i) {
        grid_params_storage[i].scene_width = config->scene_width;
        grid_params_storage[i].scene_height = config->scene_height;
        grid_params_storage[i].cell_size = config->grid_cell_sizes[i];
        grid_params_storage[i].max_radius = radius_profile_max_radius(config->active_radius_profile);
        grid_params_storage[i].dense_cell_threshold = config->dense_cell_threshold;

        methods[n].name = "cuda_uniform_grid";
        methods[n].run = run_cuda_uniform_grid;
        methods[n].params = &grid_params_storage[i];
        methods[n].reported_grid_cell_size = config->grid_cell_sizes[i];
        n++;
    }

    if (n >= max_methods) {
        return n;
    }
    lbvh_params_storage->scene_width = config->scene_width;
    lbvh_params_storage->scene_height = config->scene_height;
    methods[n].name = "cuda_lbvh";
    methods[n].run = run_cuda_lbvh;
    methods[n].params = lbvh_params_storage;
    methods[n].reported_grid_cell_size = 0.0f;
    n++;

    return n;
}

static int run_distribution(
    FILE* csv,
    const BenchmarkConfig* config,
    size_t object_count,
    const char* distribution_name,
    RadiusProfile radius_profile,
    const Circle* circles,
    const BroadphaseMethod* methods,
    int method_count) {
    if (circles == NULL || methods == NULL || method_count <= 0) {
        return 0;
    }

    printf("Running %zu objects, %s distribution, %s radius...\n",
           object_count,
           distribution_name,
           radius_profile_name(radius_profile));

    double cpu_total_ms = 0.0;
    int cpu_seen = 0;

    for (int m = 0; m < method_count; ++m) {
        const CollisionResult res = methods[m].run(circles, object_count, methods[m].params);

        BenchmarkResult row;
        memset(&row, 0, sizeof(row));
        row.object_count = object_count;
        row.distribution_type = distribution_name;
        row.radius_profile = radius_profile_name(radius_profile);
        row.method_name = methods[m].name;
        row.collision_count = res.collision_count;
        row.candidate_pair_count = res.candidate_pair_count;
        row.kernel_time_ms = res.kernel_time_ms;
        row.total_time_ms = res.total_time_ms;
        row.grid_cell_size = methods[m].reported_grid_cell_size;
        row.has_grid_stats = res.has_grid_stats;
        row.grid_stats = res.grid_stats;

        if (!cpu_seen && strcmp(methods[m].name, "cpu_brute_force") == 0) {
            cpu_total_ms = res.total_time_ms;
            cpu_seen = 1;
            row.speedup_vs_cpu = 1.0;
        } else if (cpu_seen) {
            row.speedup_vs_cpu = speedup_from_cpu(cpu_total_ms, res.total_time_ms);
        } else {
            row.speedup_vs_cpu = 0.0;
        }

        write_csv_result(csv, &row);

        if (row.has_grid_stats) {
            printf("  %-18s cell=%.2f: collisions=%llu candidates=%llu kernel=%.3f ms total=%.3f ms speedup=%.2fx max_cell=%d dense=%d\n",
                   row.method_name,
                   row.grid_cell_size,
                   row.collision_count,
                   row.candidate_pair_count,
                   row.kernel_time_ms,
                   row.total_time_ms,
                   row.speedup_vs_cpu,
                   row.grid_stats.max_objects_in_cell,
                   row.grid_stats.dense_cell_count);
        } else {
            printf("  %-18s          : collisions=%llu candidates=%llu kernel=%.3f ms total=%.3f ms speedup=%.2fx\n",
                   row.method_name,
                   row.collision_count,
                   row.candidate_pair_count,
                   row.kernel_time_ms,
                   row.total_time_ms,
                   row.speedup_vs_cpu);
        }
    }

    fflush(csv);
    (void)config;
    return 1;
}

BenchmarkConfig benchmark_default_config(void) {
    BenchmarkConfig config;
    memset(&config, 0, sizeof(config));

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

    config.radius_profiles[0] = RADIUS_PROFILE_NARROW;
    config.radius_profiles[1] = RADIUS_PROFILE_MIXED;
    config.radius_profiles[2] = RADIUS_PROFILE_EXTREME;
    config.radius_profile_len = 3;
    config.active_radius_profile = RADIUS_PROFILE_NARROW;

    config.scene_width = 1000.0f;
    config.scene_height = 1000.0f;
    config.min_radius = radius_profile_min_radius(config.active_radius_profile);
    config.max_radius = radius_profile_max_radius(config.active_radius_profile);
    config.cluster_count = 4;
    config.cluster_spread = 60.0f;
    config.dense_cell_threshold = 128;
    config.seed = 42;
    config.output_csv_path = "results/timings.csv";

    return config;
}

int run_benchmarks(const BenchmarkConfig* config) {
    if (config == NULL) {
        return 0;
    }
    if (!ensure_results_directory()) {
        fprintf(stderr, "Failed to create results directory.\n");
        return 0;
    }

    FILE* csv = fopen(config->output_csv_path, "w");
    if (csv == NULL) {
        fprintf(stderr, "Failed to open CSV output: %s\n", config->output_csv_path);
        return 0;
    }

    write_csv_header(csv);

    for (int i = 0; i < config->object_count_len; ++i) {
        const size_t object_count = config->object_counts[i];

        for (int r = 0; r < config->radius_profile_len; ++r) {
            BenchmarkConfig case_config = *config;
            case_config.active_radius_profile = config->radius_profiles[r];
            case_config.min_radius = radius_profile_min_radius(case_config.active_radius_profile);
            case_config.max_radius = radius_profile_max_radius(case_config.active_radius_profile);

            BroadphaseMethod methods[MAX_METHODS_PER_DISTRIBUTION];
            CudaGridParams grid_params_storage[MAX_GRID_CELL_SIZES];
            CudaLbvhParams lbvh_params_storage;
            const int method_count = build_methods(
                &case_config,
                methods,
                grid_params_storage,
                &lbvh_params_storage,
                MAX_METHODS_PER_DISTRIBUTION);

            Circle* uniform = generate_uniform_circles(object_count, &case_config);
            if (uniform == NULL) {
                fprintf(stderr, "Failed to allocate uniform circles for %zu objects.\n", object_count);
                fclose(csv);
                return 0;
            }
            if (!run_distribution(csv, &case_config, object_count, "uniform", case_config.active_radius_profile, uniform, methods, method_count)) {
                free(uniform);
                fclose(csv);
                return 0;
            }
            free(uniform);

            Circle* clustered = generate_clustered_circles(object_count, &case_config);
            if (clustered == NULL) {
                fprintf(stderr, "Failed to allocate clustered circles for %zu objects.\n", object_count);
                fclose(csv);
                return 0;
            }
            if (!run_distribution(csv, &case_config, object_count, "clustered", case_config.active_radius_profile, clustered, methods, method_count)) {
                free(clustered);
                fclose(csv);
                return 0;
            }
            free(clustered);
        }
    }

    fclose(csv);
    printf("Benchmark complete. CSV written to %s\n", config->output_csv_path);
    return 1;
}
