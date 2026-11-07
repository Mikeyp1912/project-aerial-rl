

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/rl/flight_environment.hpp"
#include "aerial/rl/target_tracking_reward.hpp"

int main()
{
    using namespace aerial::rl;

    FlightEnvironmentParameters parameters;

    parameters.reward_mode =
        RewardMode::BoundedTargetTracking;

    parameters.task = FlightTask::TargetTracking;

    parameters.target_tracking.randomize_target = true;

    parameters.maximum_episode_time = 2.0;

    FlightEnvironment environment(parameters);

    environment.Reset(42);

    const auto target = environment.GetTargetPosition();

    const auto commands = aerial::physics::MotorCommands{
        0.0, 0.0, 0.0, 0.0
    };

    const auto result = environment.Step(commands);

    const double expected = CalculateTargetTrackingReward(
        environment.GetTrueState(),
        target,
        parameters.target_tracking_reward
    );

    assert(std::abs(result.reward - expected) < 1e-10);

    // The existing reward mode must still work.
    parameters.reward_mode = RewardMode::QuadraticHover;

    FlightEnvironment legacy_environment(parameters);
    legacy_environment.Reset(42);

    const auto legacy_result =
        legacy_environment.Step(commands);

    auto hover_parameters = parameters.hover_reward;

    hover_parameters.target_position =
        legacy_environment.GetTargetPosition();

    const auto expected_legacy = CalculateHoverReward(
        legacy_environment.GetTrueState(),
        hover_parameters
    );

    assert(
        std::abs(
            legacy_result.reward - expected_legacy.total
        ) < 1e-10
    );

    // Invalid bounded reward parameters must be rejected.
    parameters.reward_mode =
        RewardMode::BoundedTargetTracking;

    parameters.target_tracking_reward.position_sigma = 0.0;

    bool rejected = false;

    try
    {
        FlightEnvironment invalid_environment(parameters);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }

    assert(rejected);

    std::cout << "RL reward mode tests PASSED\n";
}
