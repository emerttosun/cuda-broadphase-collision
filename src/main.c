#include "Benchmark.h"

#include <stdio.h>

int main(void) {
    BenchmarkConfig config = benchmark_default_config();

    printf("CUDA-Based Broad-Phase Collision Detection with CPU-GPU Performance Analysis\n");
    printf("Output CSV: %s\n", config.output_csv_path);
    printf("Note: run this binary on a CUDA-capable NVIDIA GPU system.\n\n");

    if (!run_benchmarks(&config)) {
        fprintf(stderr, "Benchmark failed.\n");
        return 1;
    }

    return 0;
}
