#include "Benchmark.h"
#include <iostream>
#include <cstring>

extern int run_application(int argc, char** argv);

int main(int argc, char** argv) {
    bool benchmark_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = true;
            break;
        }
    }

    if (benchmark_mode) {
        BenchmarkConfig config = benchmark_default_config();
        std::cout << "CUDA-Based Broad-Phase Collision Detection Benchmark Mode\n";
        std::cout << "Output CSV: " << config.output_csv_path << "\n\n";

        if (!run_benchmarks(&config)) {
            std::cerr << "Benchmark failed.\n";
            return 1;
        }
        return 0;
    } else {
        return run_application(argc, argv);
    }
}
