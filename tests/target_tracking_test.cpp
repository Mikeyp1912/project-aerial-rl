

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>


#include "aerial/rl/flight_environment.hpp"

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

    std::cout << "Target tracking tests PASSED.\n";
    return 0;
}
