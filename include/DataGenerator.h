#pragma once

#include "Benchmark.h"
#include "CirclesSoA.h"
#include <cstddef>

CirclesSoA generate_uniform_circles(size_t count, const BenchmarkConfig& config);
CirclesSoA generate_clustered_circles(size_t count, const BenchmarkConfig& config);
