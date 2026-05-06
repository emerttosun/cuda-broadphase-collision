#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GridStats {
    int max_objects_in_cell;
    double avg_objects_per_non_empty_cell;
    int dense_cell_count;
} GridStats;

#ifdef __cplusplus
}
#endif
