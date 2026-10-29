// /tests/rl_episode_termination_test.cpp

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Geometry>

#include "aerial/rl/episode_termination.hpp"

int main()
{
    using namespace aerial;

    constexpr double pi = 3.14159265358979323846;

    rl::EpisodeTerminationParameters parameters;

    const Eigen::Vector3d target{0.0, 0.0, 2.0};

    physics::DroneState state;
    state.position = target;

    // Valid hover state must not terminate.
    {
        const auto result = rl::CheckEpisodeTermination(
            state, parameters, target
        );

        assert(!result.terminated);
        assert(result.reason == rl::TerminationReason::None);
    }

    // Falling below the minimum altitude terminates.
    {
        auto failed = state;
        failed.position.z() = -0.1;

        const auto result = rl::CheckEpisodeTermination(
            failed, parameters, target
        );

        assert(result.terminated);
        assert(result.reason ==
               rl::TerminationReason::MinimumAltitude);
    }

    // Exceeding the maximum altitude terminates.
    {
        auto failed = state;
        failed.position.z() = 10.1;

        const auto result = rl::CheckEpisodeTermination(
            failed, parameters, target
        );

        assert(result.terminated);
        assert(result.reason ==
               rl::TerminationReason::MaximumAltitude);
    }

    // Horizontal distance is measured relative to the target.
    {
        auto failed = state;
        failed.position.x() = 10.1;

        const auto result = rl::CheckEpisodeTermination(
            failed, parameters, target
        );

        assert(result.terminated);
        assert(result.reason ==
               rl::TerminationReason::HorizontalBoundary);
    }

    // Excessive roll terminates the episode.
    {
        auto failed = state;

        failed.orientation =
            Eigen::AngleAxisd(
                85.0 * pi / 180.0,
                Eigen::Vector3d::UnitX()
            );

        const auto result = rl::CheckEpisodeTermination(
            failed, parameters, target
        );

        assert(result.terminated);
        assert(result.reason ==
               rl::TerminationReason::ExcessiveTilt);
    }

    // Yaw alone must not count as excessive tilt.
    {
        auto yawed = state;

        yawed.orientation =
            Eigen::AngleAxisd(
                pi,
                Eigen::Vector3d::UnitZ()
            );

        const auto result = rl::CheckEpisodeTermination(
            yawed, parameters, target
        );

        assert(!result.terminated);
    }

    // Non-finite physics state must be classified as failure.
    {
        auto failed = state;

        failed.velocity.x() =
            std::numeric_limits<double>::quiet_NaN();

        const auto result = rl::CheckEpisodeTermination(
            failed, parameters, target
        );

        assert(result.terminated);
        assert(result.reason ==
               rl::TerminationReason::NonFiniteState);
    }

    // Invalid termination settings must be rejected.
    {
        auto invalid = parameters;
        invalid.maximum_horizontal_distance = -1.0;

        bool threw = false;

        try
        {
            rl::ValidateEpisodeTerminationParameters(invalid);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    std::cout << "RL episode termination tests PASSED\n";

    return 0;
}
