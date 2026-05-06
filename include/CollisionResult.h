#pragma once

#include <string>
#include <unordered_map>

struct CollisionResult {
    unsigned long long collision_count = 0;
    unsigned long long candidate_pair_count = 0;
    
    // Transparent performance matrix metrics
    double memory_transfer_time_ms = 0.0;
    double kernel_execution_time_ms = 0.0;
    double total_time_ms = 0.0;
    
    // Extensible metrics for advanced methods (e.g., LBVH tree build time, Grid dense cell count)
    std::unordered_map<std::string, double> extra_stats;
};
