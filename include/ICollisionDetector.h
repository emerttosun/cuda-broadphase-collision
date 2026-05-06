#pragma once

#include "CirclesSoA.h"
#include "CollisionResult.h"
#include <string>

class ICollisionDetector {
public:
    virtual ~ICollisionDetector() = default;

    // Uploads/prepares the data. For CPU this might do nothing,
    // for GPU this will handle Host-to-Device transfers.
    virtual void update_data(const CirclesSoA& circles) = 0;

    // Executes the collision detection algorithm and returns detailed stats.
    virtual CollisionResult run_detection() = 0;

    // Returns the academic name of the method.
    virtual std::string get_name() const = 0;
    
    // Optionally return a pointer to device VBO data if zero-copy rendering is supported
    virtual void* get_device_vbo() const { return nullptr; }
};
