

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include "aerial/rl/flight_environment.hpp"
#include "aerial/rl/hover_reward.hpp"

int main()
{
    using namespace aerial::rl;

    FlightEnvironmentParameters parameters;

    parameters.task = FlightTask::TargetTracking;
    parameters.target_tracking.randomize_target = true;
    parameters.target_tracking.maximum_target_offset =
        Eigen::Vector3d(2.0, 2.0, 1.0);

    FlightEnvironment environment(parameters);

    // Same seed must reproduce the same target.
    environment.Reset(123);
    const Eigen::Vector3d first =
        environment.GetTargetPosition();

    environment.Reset(123);
    const Eigen::Vector3d repeated =
        environment.GetTargetPosition();

    assert(first.isApprox(repeated, 1e-12));

    // Target must remain within configured bounds.
    const Eigen::Vector3d reference =
        parameters.hover_reward.target_position;

    for (std::uint32_t seed = 0; seed < 100; ++seed)
    {
        environment.Reset(seed);

        const Eigen::Vector3d target =
            environment.GetTargetPosition();

        assert(target.allFinite());
        assert(target.z() >= 0.5);

        for (int axis = 0; axis < 3; ++axis)
        {
            assert(std::abs(target[axis] - reference[axis]) <=
                parameters.target_tracking
                    .maximum_target_offset[axis] + 1e-12);
        }
    }

    // Different seeds should produce different targets.
    environment.Reset(123);
    const Eigen::Vector3d target_a =
        environment.GetTargetPosition();

    environment.Reset(456);
    const Eigen::Vector3d target_b =
        environment.GetTargetPosition();

    assert(!target_a.isApprox(target_b, 1e-12));

    // Target remains constant throughout an episode.
    environment.Reset(789);
    const Eigen::Vector3d episode_target =
        environment.GetTargetPosition();

    environment.Step(
        aerial::physics::MotorCommands{0.0, 0.0, 0.0, 0.0}
    );

    assert(environment.GetTargetPosition().isApprox(
        episode_target, 1e-12
    ));

    // Hover mode must ignore target randomization.
    parameters.task = FlightTask::Hover;

    FlightEnvironment hover_environment(parameters);

    hover_environment.Reset(123);

    assert(hover_environment.GetTargetPosition().isApprox(
        reference, 1e-12
    ));

    hover_environment.Reset(456);

    assert(hover_environment.GetTargetPosition().isApprox(
        reference, 1e-12
    ));

    // Fixed target tracking must preserve the reference target.
    parameters.task = FlightTask::TargetTracking;
    parameters.target_tracking.randomize_target = false;

    FlightEnvironment fixed_environment(parameters);

    fixed_environment.Reset(123);

    assert(fixed_environment.GetTargetPosition().isApprox(
        reference, 1e-12
    ));

    // Reject invalid randomization bounds.
    parameters.target_tracking.maximum_target_offset.x() = -1.0;

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

    {
        FlightEnvironmentParameters config;

        config.task = FlightTask::TargetTracking;
        config.target_tracking.randomize_target = true;

        FlightEnvironment tracking(config);

        tracking.Reset(123);

        const Eigen::Vector3d target =
            tracking.GetTargetPosition();

        const auto result = tracking.Step(
            aerial::physics::MotorCommands{0.0, 0.0, 0.0, 0.0}
        );

        auto expected_parameters = config.hover_reward;
        expected_parameters.target_position = target;

        const auto expected = CalculateHoverReward(
            tracking.GetTrueState(),
            expected_parameters
        );

        assert(std::abs(result.reward - expected.total) < 1e-10);
    }

    {
        FlightEnvironmentParameters config;

        config.task = FlightTask::TargetTracking;
        config.observation_mode = ObservationMode::HoverAugmented;
        config.target_tracking.randomize_target = true;

        FlightEnvironment tracking(config);

        tracking.Reset(456);

        const auto observation = tracking.GetPolicyObservation();
        const auto raw = tracking.GetRLObservation();
        const auto target = tracking.GetTargetPosition();

        assert(observation.size() == 19);

        const double scale =
            config.observation_normalization.position_scale;

        for (int axis = 0; axis < 3; ++axis)
        {
            double expected =
                (raw[axis] - target[axis]) / scale;

            if (config.observation_normalization.clip)
            {
                expected = std::clamp(expected, -1.0, 1.0);
            }

            assert(std::abs(
                observation[16 + axis] - expected
            ) < 1e-10);
        }
    }

    {
        FlightEnvironmentParameters config;

        config.task = FlightTask::TargetTracking;
        config.action_mode = ActionMode::PIDTargets;
        config.target_tracking.randomize_target = true;

        config.pid_targets.reference =
            PIDTargetReference::WorldRelative;

        config.pid_targets.hover_centered_altitude = true;
        config.pid_targets.maximum_horizontal_offset = 1.0;

        bool rejected = false;

        try
        {
            FlightEnvironment invalid(config);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }

        assert(rejected);
    }

    std::cout << "Target tracking tests PASSED.\n";
    return 0;
}
