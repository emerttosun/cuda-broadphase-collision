#pragma once

#include "Rng.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RadiusProfile {
    RADIUS_PROFILE_NARROW = 0,
    RADIUS_PROFILE_MIXED = 1,
    RADIUS_PROFILE_EXTREME = 2,
    RADIUS_PROFILE_COUNT = 3
} RadiusProfile;

static inline const char* radius_profile_name(RadiusProfile profile) {
    switch (profile) {
        case RADIUS_PROFILE_MIXED:
            return "mixed";
        case RADIUS_PROFILE_EXTREME:
            return "extreme";
        case RADIUS_PROFILE_NARROW:
        default:
            return "narrow";
    }
}

static inline float radius_profile_min_radius(RadiusProfile profile) {
    (void)profile;
    return 2.0f;
}

static inline float radius_profile_max_radius(RadiusProfile profile) {
    switch (profile) {
        case RADIUS_PROFILE_MIXED:
            return 16.0f;
        case RADIUS_PROFILE_EXTREME:
            return 32.0f;
        case RADIUS_PROFILE_NARROW:
        default:
            return 5.0f;
    }
}

static inline RadiusProfile radius_profile_next(RadiusProfile profile) {
    return (RadiusProfile)(((int)profile + 1) % (int)RADIUS_PROFILE_COUNT);
}

static inline float sample_radius_for_profile(unsigned int* rng, RadiusProfile profile) {
    const float min_radius = radius_profile_min_radius(profile);
    const float max_radius = radius_profile_max_radius(profile);
    if (profile == RADIUS_PROFILE_NARROW) {
        return rng_range_float(rng, min_radius, max_radius);
    }

    const float log_min = logf(min_radius);
    const float log_max = logf(max_radius);
    return expf(rng_range_float(rng, log_min, log_max));
}

#ifdef __cplusplus
}
#endif
