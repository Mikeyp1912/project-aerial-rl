

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "aerial/rl/action_adapter.hpp"

namespace
{
    void CheckTarget(
        const Eigen::Vector2d& actual,
        const Eigen::Vector2d& expected
    )
    {
        assert((actual - expected).norm() < 1e-12);
    }
}

int main()
{
    using namespace aerial::rl;

    PIDTargetActionParameters parameters;

    const Eigen::Vector2d episode_origin(0.4, -0.3);
    const Action zero{0.0, 0.0, 0.0, 0.0};

    // Existing behaviour must remain unchanged.
    CheckTarget(
        CalculatePIDHorizontalTarget(
            parameters, episode_origin, zero
        ),
        episode_origin
    );

    // World-relative zero action targets the fixed position.
    parameters.reference = PIDTargetReference::WorldRelative;
    parameters.world_target_position = {0.0, 0.0};

    CheckTarget(
        CalculatePIDHorizontalTarget(
            parameters, episode_origin, zero
        ),
        Eigen::Vector2d::Zero()
    );

    // Nonzero world target and normalized offsets.
    parameters.world_target_position = {1.0, -2.0};

    const Action offset{0.2, -0.4, 0.0, 0.0};

    CheckTarget(
        CalculatePIDHorizontalTarget(
            parameters, episode_origin, offset
        ),
        Eigen::Vector2d(2.0, -4.0)
    );

    // Out-of-range actions are clamped.
    const Action excessive{2.0, -3.0, 0.0, 0.0};

    CheckTarget(
        CalculatePIDHorizontalTarget(
            parameters, episode_origin, excessive
        ),
        Eigen::Vector2d(6.0, -7.0)
    );

    // Invalid configuration is rejected.
    parameters.maximum_horizontal_offset = -1.0;

    bool threw = false;

    try {
        CalculatePIDHorizontalTarget(
            parameters, episode_origin, zero
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    std::cout << "PID target reference tests PASSED\n";
    return 0;
}
