#pragma once

#include "ICollisionDetector.h"

class CpuBruteForceDetector : public ICollisionDetector {
public:
    CpuBruteForceDetector() = default;
    ~CpuBruteForceDetector() override = default;

    void update_data(const CirclesSoA& circles) override;
    CollisionResult run_detection() override;
    std::string get_name() const override { return "CPU Brute Force"; }

private:
    const CirclesSoA* m_circles = nullptr;
};
