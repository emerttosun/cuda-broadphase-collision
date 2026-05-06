#pragma once

#if defined(__CUDACC__)
#define CMP674_RNG_INLINE static __device__ __host__ inline
#else
#define CMP674_RNG_INLINE static inline
#endif

CMP674_RNG_INLINE unsigned int rng_next_u32(unsigned int* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

CMP674_RNG_INLINE float rng_unit_float(unsigned int* state) {
    return (float)(rng_next_u32(state) & 0x00FFFFFFu) / (float)0x01000000u;
}

CMP674_RNG_INLINE float rng_range_float(unsigned int* state, float min_value, float max_value) {
    return min_value + (max_value - min_value) * rng_unit_float(state);
}
