/**
 * @file CudaHash.cu
 * @brief Hierarchical Spatial Hashing (HSH) GPU broad-phase implementation.
 *
 * Algorithm overview
 * ==================
 * Each input circle is placed at level
 *     L(r) = clamp(ceil(log2(2*r / base_cell_size)), 0, num_levels-1)
 * where @c base_cell_size = 2 * min_radius. Cell size at level L is
 *     cell_size(L) = base_cell_size * 2^L
 * so that diameter(2*r) <= cell_size(L), i.e. each object spans at most a
 * 2x2 footprint at its own level (and a 2x2 or smaller footprint at any
 * strictly higher level).
 *
 * Bucket key for object i at level L = L(r_i):
 *     hash(cx,cy) = ((u32)cx * 73856093u) ^ ((u32)cy * 19349663u)   [Teschner 2003]
 *     bucket = hash(cx,cy) mod M           with M = next_prime(2N)
 *     composite = L * M + bucket
 *
 * The composite key is sorted with Thrust (alongside the original object
 * index), then a single scan kernel writes [cell_start, cell_end) per
 * non-empty composite key. The total cell-range table has size
 * num_levels * M, stored as one flat int array.
 *
 * Query
 * -----
 * Each thread maps to one input object @c i. For each query level
 * @c L_q in [L_self, num_levels-1] it computes the cells covered by its
 * AABB at @c L_q (1, 2, or 4 cells), looks up the composite bucket, and
 * iterates members. Two filters apply:
 *   - hash collision: candidate's recomputed (cx,cy) at L_q must match
 *     the queried cell;
 *   - pair double-counting at self-level: only count if @c j > @c i.
 * At strictly higher levels no index filter is needed: the bigger object
 * lives at L_q and never queries down to L_self, so each cross-level
 * pair is observed exactly once (from the smaller object).
 *
 * Complexity
 * ----------
 * Build: O(N + N log N) dominated by the radix sort. Query: O(N * k)
 * where k is the average number of candidates per object (a function
 * of density and radius distribution). For uniformly mixed-radius
 * scenes, k is bounded by a small constant per level, so the total
 * pipeline is effectively O(N log N).
 *
 * References
 * ----------
 * - M. Teschner et al., "Optimized Spatial Hashing for Collision
 *   Detection of Deformable Objects", VMV 2003.
 * - M. Eitz, L. Lixu, "Hierarchical Hashing Scheme for Nearest
 *   Neighbor Search and Broad-Phase Collision Detection", JGT 2007.
 * - S. Green, "Particle Simulation using CUDA", NVIDIA Whitepaper 2010
 *   (sort+ranges build pattern, also used by CudaGrid.cu in this repo).
 */

#include "CudaHash.cuh"
#include "CollisionMath.h"
#include "CudaUtils.cuh"
#include "Timer.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** @brief Teschner et al. 2003 prime for the x cell coordinate. */
#define CUDA_HASH_PRIME_X 73856093u
/** @brief Teschner et al. 2003 prime for the y cell coordinate. */
#define CUDA_HASH_PRIME_Y 19349663u

/* ------------------------------------------------------------------ */
/*  Device-side helpers                                                */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute the spatial hash bucket index for a 2D cell coordinate.
 * @param cx       Cell x-coordinate (signed integer; negative supported).
 * @param cy       Cell y-coordinate (signed integer; negative supported).
 * @param table_M  Hash table size (per level). Must be > 0.
 * @return Bucket in [0, table_M).
 */
__device__ static inline unsigned int hash_bucket_2d(int cx, int cy, unsigned int table_M) {
    const unsigned int h = ((unsigned int)cx) * CUDA_HASH_PRIME_X
                         ^ ((unsigned int)cy) * CUDA_HASH_PRIME_Y;
    return h % table_M;
}

/**
 * @brief Resolve the storage level for an object of the given radius.
 *
 * Inline twin of the host-side helper so the assign and query kernels
 * stay in agreement without additional auxiliary arrays.
 *
 * @param radius             Circle radius (positive).
 * @param base_cell_size     Cell size at level 0.
 * @param max_level_inclusive @c num_levels - 1.
 * @return Level in [0, max_level_inclusive].
 */
__device__ static inline int hash_select_level(
    float radius,
    float base_cell_size,
    int max_level_inclusive) {
    const float diameter = 2.0f * radius;
    if (diameter <= base_cell_size) {
        return 0;
    }
    const float ratio = diameter / base_cell_size;
    int L = (int)ceilf(log2f(ratio));
    if (L < 0) {
        L = 0;
    }
    if (L > max_level_inclusive) {
        L = max_level_inclusive;
    }
    return L;
}

/* ------------------------------------------------------------------ */
/*  Kernels                                                            */
/* ------------------------------------------------------------------ */

/**
 * @brief Compute composite (level, hash-bucket) key and copy index for
 *        every input object.
 */
__global__ static void hash_assign_kernel(
    const Circle* circles,
    size_t count,
    float base_cell_size,
    int num_levels,
    unsigned int table_M,
    int* composite_keys,
    int* indices) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const Circle c = circles[i];
    const int L = hash_select_level(c.radius, base_cell_size, num_levels - 1);
    const float cell_size_L = base_cell_size * (float)(1 << L);
    const int cx = (int)floorf(c.x / cell_size_L);
    const int cy = (int)floorf(c.y / cell_size_L);
    const unsigned int bucket = hash_bucket_2d(cx, cy, table_M);

    composite_keys[i] = (int)((unsigned int)L * table_M + bucket);
    indices[i] = (int)i;
}

/**
 * @brief Initialise the flat cell-range arrays to "empty" sentinels.
 */
__global__ static void hash_init_cell_ranges_kernel(
    int* cell_start,
    int* cell_end,
    int total_buckets) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_buckets) {
        return;
    }
    cell_start[i] = -1;
    cell_end[i] = -1;
}

/**
 * @brief Scan the sorted composite-key array and write [start, end) per
 *        non-empty bucket. Boundaries are detected by neighbour
 *        comparison (same pattern as ::build_cell_ranges_kernel in
 *        CudaGrid.cu).
 */
__global__ static void hash_build_cell_ranges_kernel(
    const int* sorted_composite_keys,
    int* cell_start,
    int* cell_end,
    size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const int current_key = sorted_composite_keys[i];
    if (i == 0 || sorted_composite_keys[i - 1] != current_key) {
        cell_start[current_key] = (int)i;
    }
    if (i == count - 1 || sorted_composite_keys[i + 1] != current_key) {
        cell_end[current_key] = (int)i + 1;
    }
}

/**
 * @brief Per-object query kernel.
 *
 * One thread per input object. Walks the object's own storage level
 * and every strictly higher level, scanning the 1, 2 or 4 cells its
 * AABB overlaps and verifying candidates against true cell coordinates
 * (filtering out hash collisions).
 */
__global__ static void hash_query_kernel(
    const Circle* circles,
    const int* sorted_indices,
    const int* cell_start,
    const int* cell_end,
    size_t count,
    float base_cell_size,
    int num_levels,
    unsigned int table_M,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const Circle self = circles[i];
    const int L_self = hash_select_level(self.radius, base_cell_size, num_levels - 1);

    unsigned long long local_collisions = 0;
    unsigned long long local_candidates = 0;

    for (int L_q = L_self; L_q < num_levels; ++L_q) {
        const float cell_size_q = base_cell_size * (float)(1 << L_q);
        const float inv_cell = 1.0f / cell_size_q;

        /* Scan a fixed 3x3 window around self's cell at this level.
         *
         * Justification (level invariant @c 2*r <= cs at every level):
         *   - same-level case (L_q == L_self): r_self, r_other <= cs/2,
         *     so |self.x - other.x| <= cs, hence cell index distance <= 1;
         *   - cross-level case (L_q > L_self): r_self <= cs(L_self)/2 <=
         *     cs(L_q)/4, r_other <= cs(L_q)/2, sum <= 3*cs(L_q)/4 < cs(L_q),
         *     hence cell index distance <= 1.
         *
         * Using @c floorf(self.x * inv_cell) directly (rather than an
         * AABB-extension subtraction) avoids floating-point cancellation
         * around cell boundaries that would otherwise drop a cell at
         * specific (x, r) combinations. */
        const int self_cx_q = (int)floorf(self.x * inv_cell);
        const int self_cy_q = (int)floorf(self.y * inv_cell);
        const int cx_min = self_cx_q - 1;
        const int cx_max = self_cx_q + 1;
        const int cy_min = self_cy_q - 1;
        const int cy_max = self_cy_q + 1;

        for (int cy = cy_min; cy <= cy_max; ++cy) {
            for (int cx = cx_min; cx <= cx_max; ++cx) {
                const unsigned int bucket = hash_bucket_2d(cx, cy, table_M);
                const int composite = (int)((unsigned int)L_q * table_M + bucket);
                const int start = cell_start[composite];
                const int end = cell_end[composite];
                if (start < 0 || end < 0) {
                    continue;
                }

                for (int p = start; p < end; ++p) {
                    const int j = sorted_indices[p];

                    /* Filter hash collisions: members of this bucket may
                     * include cells with the same hash but different
                     * (cx,cy). Recompute and compare. */
                    const Circle other = circles[j];
                    const int other_cx = (int)floorf(other.x * inv_cell);
                    const int other_cy = (int)floorf(other.y * inv_cell);
                    if (other_cx != cx || other_cy != cy) {
                        continue;
                    }

                    /* At self-level avoid double-counting. At strictly
                     * higher levels the bigger object never queries
                     * down, so no filter is needed. */
                    if (L_q == L_self && j <= (int)i) {
                        continue;
                    }

                    ++local_candidates;
                    if (circles_overlap(&self, &other)) {
                        ++local_collisions;
                    }
                }
            }
        }
    }

    if (local_candidates > 0) {
        atomicAdd(candidate_pair_count, local_candidates);
    }
    if (local_collisions > 0) {
        atomicAdd(collision_count, local_collisions);
    }
}

/* ------------------------------------------------------------------ */
/*  Host-side helpers                                                  */
/* ------------------------------------------------------------------ */

/**
 * @brief Choose the number of levels from a radius interval.
 *
 * num_levels is the smallest integer such that
 *     base * 2^(num_levels-1) >= 2 * max_radius
 * with @c base = 2 * min_radius. Always returns at least 1, capped at
 * ::CUDA_HASH_MAX_LEVELS.
 */
static int hash_auto_num_levels(float min_radius, float max_radius) {
    if (min_radius <= 0.0f || max_radius <= 0.0f || max_radius < min_radius) {
        return 1;
    }
    const float ratio = max_radius / min_radius;
    int L = (int)ceilf(log2f(ratio)) + 1;
    if (L < 1) {
        L = 1;
    }
    if (L > CUDA_HASH_MAX_LEVELS) {
        L = CUDA_HASH_MAX_LEVELS;
    }
    return L;
}

/**
 * @brief Test whether @c n is prime (trial division). Used by
 *        ::hash_next_prime to pick a hash table size with low collision
 *        bias.
 */
static int hash_is_prime(unsigned int n) {
    if (n < 2u) {
        return 0;
    }
    if (n == 2u) {
        return 1;
    }
    if ((n % 2u) == 0u) {
        return 0;
    }
    for (unsigned int d = 3u; d * d <= n; d += 2u) {
        if ((n % d) == 0u) {
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Smallest prime >= @c lower_bound. Bounded scan; returns the
 *        input unchanged if it is already prime.
 */
static unsigned int hash_next_prime(unsigned int lower_bound) {
    if (lower_bound <= 2u) {
        return 2u;
    }
    unsigned int n = (lower_bound % 2u == 0u) ? (lower_bound + 1u) : lower_bound;
    while (!hash_is_prime(n)) {
        n += 2u;
    }
    return n;
}

/**
 * @brief Aggregate cell occupancy across all levels into a ::GridStats
 *        record so HSH results are comparable to the uniform grid in CSV
 *        output.
 */
static GridStats hash_compute_stats_on_host(
    const int* cell_start,
    const int* cell_end,
    int total_buckets,
    int dense_cell_threshold) {
    GridStats stats;
    stats.max_objects_in_cell = 0;
    stats.avg_objects_per_non_empty_cell = 0.0;
    stats.dense_cell_count = 0;

    int non_empty_cells = 0;
    unsigned long long total_objects_in_non_empty_cells = 0;

    for (int i = 0; i < total_buckets; ++i) {
        if (cell_start[i] < 0 || cell_end[i] < 0) {
            continue;
        }

        const int objects_in_cell = cell_end[i] - cell_start[i];
        if (objects_in_cell <= 0) {
            continue;
        }

        ++non_empty_cells;
        total_objects_in_non_empty_cells += (unsigned long long)objects_in_cell;
        if (objects_in_cell > stats.max_objects_in_cell) {
            stats.max_objects_in_cell = objects_in_cell;
        }
        if (objects_in_cell >= dense_cell_threshold) {
            ++stats.dense_cell_count;
        }
    }

    if (non_empty_cells > 0) {
        stats.avg_objects_per_non_empty_cell =
            (double)total_objects_in_non_empty_cells / (double)non_empty_cells;
    }

    return stats;
}

/* ------------------------------------------------------------------ */
/*  Public entry point                                                 */
/* ------------------------------------------------------------------ */

extern "C" CollisionResult run_cuda_hash(const Circle* circles, size_t count, const void* params) {
    CollisionResult result;
    memset(&result, 0, sizeof(result));
    result.has_grid_stats = 1;

    if (params == NULL) {
        fprintf(stderr, "run_cuda_hash: params must point to CudaHashParams.\n");
        return result;
    }

    const CudaHashParams* p = (const CudaHashParams*)params;
    if (circles == NULL || count == 0) {
        return result;
    }
    if (p->min_radius <= 0.0f || p->max_radius < p->min_radius) {
        fprintf(stderr,
                "run_cuda_hash: invalid radius range (min=%.3f max=%.3f).\n",
                p->min_radius,
                p->max_radius);
        return result;
    }
    if (count > (size_t)2147483647) {
        fprintf(stderr, "run_cuda_hash: count exceeds int range.\n");
        return result;
    }

    const double total_start = timer_now_ms();

    /* Auto-derive structural parameters. */
    const float base_cell_size = 2.0f * p->min_radius;
    int num_levels = (p->num_levels > 0) ? p->num_levels
                                          : hash_auto_num_levels(p->min_radius, p->max_radius);
    if (num_levels < 1) {
        num_levels = 1;
    }
    if (num_levels > CUDA_HASH_MAX_LEVELS) {
        num_levels = CUDA_HASH_MAX_LEVELS;
    }

    unsigned int table_M;
    if (p->hash_table_size > 0) {
        table_M = (unsigned int)p->hash_table_size;
    } else {
        const unsigned int target = (unsigned int)(2u * (unsigned int)count + 1u);
        table_M = hash_next_prime(target < 31u ? 31u : target);
    }
    const long long total_buckets_ll = (long long)num_levels * (long long)table_M;
    if (total_buckets_ll > (long long)2147483647) {
        fprintf(stderr,
                "run_cuda_hash: total bucket count %lld exceeds int range; "
                "reduce hash_table_size or num_levels.\n",
                total_buckets_ll);
        return result;
    }
    const int total_buckets = (int)total_buckets_ll;

    Circle* d_circles = NULL;
    int* d_composite_keys = NULL;
    int* d_indices = NULL;
    int* d_cell_start = NULL;
    int* d_cell_end = NULL;
    unsigned long long* d_collision_count = NULL;
    unsigned long long* d_candidate_pair_count = NULL;
    int* h_cell_start = NULL;
    int* h_cell_end = NULL;
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaMalloc((void**)&d_circles, count * sizeof(Circle)));
    CUDA_CHECK(cudaMalloc((void**)&d_composite_keys, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_indices, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cell_start, (size_t)total_buckets * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cell_end, (size_t)total_buckets * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&d_candidate_pair_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_circles, circles, count * sizeof(Circle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threads_per_block = 256;
    const int object_blocks = (int)((count + threads_per_block - 1) / threads_per_block);
    const int bucket_blocks = (total_buckets + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaEventRecord(start));
    hash_init_cell_ranges_kernel<<<bucket_blocks, threads_per_block>>>(
        d_cell_start,
        d_cell_end,
        total_buckets);
    CUDA_CHECK(cudaGetLastError());

    hash_assign_kernel<<<object_blocks, threads_per_block>>>(
        d_circles,
        count,
        base_cell_size,
        num_levels,
        table_M,
        d_composite_keys,
        d_indices);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> keys_ptr(d_composite_keys);
    thrust::device_ptr<int> indices_ptr(d_indices);
    thrust::sort_by_key(keys_ptr, keys_ptr + count, indices_ptr);

    hash_build_cell_ranges_kernel<<<object_blocks, threads_per_block>>>(
        d_composite_keys,
        d_cell_start,
        d_cell_end,
        count);
    CUDA_CHECK(cudaGetLastError());

    hash_query_kernel<<<object_blocks, threads_per_block>>>(
        d_circles,
        d_indices,
        d_cell_start,
        d_cell_end,
        count,
        base_cell_size,
        num_levels,
        table_M,
        d_collision_count,
        d_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_elapsed, start, stop));
    result.kernel_time_ms = (double)kernel_elapsed;
    CUDA_CHECK(cudaMemcpy(&result.collision_count, d_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result.candidate_pair_count, d_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    h_cell_start = (int*)malloc((size_t)total_buckets * sizeof(int));
    h_cell_end = (int*)malloc((size_t)total_buckets * sizeof(int));
    if (h_cell_start != NULL && h_cell_end != NULL) {
        CUDA_CHECK(cudaMemcpy(h_cell_start, d_cell_start, (size_t)total_buckets * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_cell_end, d_cell_end, (size_t)total_buckets * sizeof(int), cudaMemcpyDeviceToHost));
        result.grid_stats = hash_compute_stats_on_host(
            h_cell_start,
            h_cell_end,
            total_buckets,
            p->dense_cell_threshold);
    }

    free(h_cell_start);
    free(h_cell_end);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_circles));
    CUDA_CHECK(cudaFree(d_composite_keys));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_cell_start));
    CUDA_CHECK(cudaFree(d_cell_end));
    CUDA_CHECK(cudaFree(d_collision_count));
    CUDA_CHECK(cudaFree(d_candidate_pair_count));

    result.total_time_ms = timer_now_ms() - total_start;
    return result;
}
