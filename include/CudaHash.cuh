#pragma once

#include "Broadphase.h"
#include "Circle.h"
#include "CollisionResult.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file CudaHash.cuh
 * @brief Hierarchical Spatial Hashing (HSH) broad-phase collision detection.
 *
 * Implements a multi-level spatial hash (Teschner et al. 2003 + hierarchical
 * extension per Eitz/Lixu 2007) that places each object into the smallest
 * grid level whose cells fit its diameter. This removes the
 * cell_size >= 2 * max_radius constraint of a single uniform grid and
 * therefore stays competitive on radius distributions where small and
 * large objects coexist (e.g. the "mixed" and "extreme" radius profiles).
 *
 * Each level uses an independent open hash table of @c hash_table_size
 * buckets; level @c L has cell size @c base_cell_size * 2^L. A composite
 * sort key @c (level * hash_table_size + hash(cx,cy)) places all levels
 * into one sorted array with a single Thrust radix sort pass and a single
 * cell-range build kernel.
 */

/**
 * @brief Maximum number of grid levels supported by the HSH builder.
 *
 * Empirically 4-6 levels suffice for the project's radius profiles
 * (max/min radius up to 16x). The value is exposed publicly so callers
 * can size storage if they wish.
 */
#define CUDA_HASH_MAX_LEVELS 8

/**
 * @brief Configuration for ::run_cuda_hash.
 *
 * Set @c num_levels or @c hash_table_size to 0 to enable automatic
 * derivation from @c min_radius, @c max_radius and the input count.
 */
typedef struct CudaHashParams {
    /** @brief Smallest radius expected in the input set. Used to pick the
     *  level-0 cell size (= 2 * min_radius). */
    float min_radius;
    /** @brief Largest radius expected in the input set. Used to pick the
     *  number of levels (top level cell size >= 2 * max_radius). */
    float max_radius;
    /** @brief Override for the number of grid levels. 0 selects auto
     *  (clamped to [1, ::CUDA_HASH_MAX_LEVELS]). */
    int num_levels;
    /** @brief Override for the per-level hash table size. 0 selects auto
     *  (next prime >= 2 * count). */
    int hash_table_size;
    /** @brief Threshold (objects per non-empty bucket) above which a bucket
     *  is reported as "dense" in ::CollisionResult::grid_stats. Mirrors the
     *  uniform grid stats so the CSV is comparable. */
    int dense_cell_threshold;
} CudaHashParams;

/**
 * @brief Run hierarchical spatial hashing broad-phase + narrow-phase pass.
 *
 * Performs the full pipeline:
 *  1. Per-object level assignment + hash key computation.
 *  2. Thrust radix sort by composite (level, hash) key.
 *  3. Cell-range build (sorted scan).
 *  4. Per-object query that walks the object's own level and all higher
 *     levels, performing exact circle-overlap tests on candidates.
 *
 * @param[in]  circles  Host array of input circles. Must have @c count
 *                      entries. May be @c NULL only if @c count is 0.
 * @param[in]  count    Number of circles.
 * @param[in]  params   Pointer to ::CudaHashParams. Must not be @c NULL.
 *
 * @return Populated ::CollisionResult. The collision_count and
 *         candidate_pair_count fields use the same semantics as the other
 *         broad-phase methods in this project (each unordered pair counted
 *         at most once). @c kernel_time_ms covers the GPU pipeline only;
 *         @c total_time_ms includes host-side allocation and transfers.
 *         @c has_grid_stats is set; @c grid_stats reports bucket
 *         occupancy across all levels combined.
 *
 * @note    The pipeline executes on the default CUDA stream. Allocations
 *          are released before return. The input @c circles buffer is
 *          copied to device memory each call (same as other methods in
 *          this project), so the function is safe to call repeatedly with
 *          different inputs.
 */
CollisionResult run_cuda_hash(const Circle* circles, size_t count, const void* params);

#ifdef __cplusplus
}
#endif
