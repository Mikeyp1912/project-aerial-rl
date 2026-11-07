

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/rl/target_tracking_reward.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-12;

    const Eigen::Vector3d target(0.0, 0.0, 2.0);

    physics::DroneState state;
    state.position = target;

    rl::TargetTrackingRewardParameters parameters;

    const double at_target =
        rl::CalculateTargetTrackingReward(
            state, target, parameters
        );

    assert(std::abs(at_target - 1.0) < tolerance);

    state.position.x() = 1.0;

    const double one_metre =
        rl::CalculateTargetTrackingReward(
            state, target, parameters
        );

    assert(
        std::abs(one_metre - std::exp(-0.5)) <
        tolerance
    );

    state.position.x() = 2.0;

    const double two_metres =
        rl::CalculateTargetTrackingReward(
            state, target, parameters
        );

    assert(
        std::abs(two_metres - std::exp(-2.0)) <
        tolerance
    );

    assert(at_target > one_metre);
    assert(one_metre > two_metres);

    // Approaching the target must improve reward.
    state.position.x() = 1.5;

    const double approaching =
        rl::CalculateTargetTrackingReward(
            state, target, parameters
        );

    assert(approaching > two_metres);

    // Moving quickly should reduce reward.
    state.position = target;
    state.velocity.x() = 2.0;

    const double moving =
        rl::CalculateTargetTrackingReward(
            state, target, parameters
        );

    assert(moving < at_target);

    // Invalid sigma must be rejected.
    parameters.position_sigma = 0.0;

    bool threw = false;

    try
    {
        rl::CalculateTargetTrackingReward(
            state, target, parameters
        );
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }

    assert(threw);

    std::cout
        << "Target tracking reward tests PASSED\n";
}
