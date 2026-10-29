// /tests/rl_episode_statistics_test.cpp



#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/rl/episode_statistics.hpp"
#include "aerial/rl/flight_environment.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-12;

    // A fresh tracker must contain zero statistics.
    {
        rl::EpisodeStatisticsTracker tracker;

        const auto& stats = tracker.GetStatistics();

        assert(stats.steps == 0);
        assert(stats.duration == 0.0);
        assert(stats.cumulative_reward == 0.0);
        assert(stats.mean_position_error == 0.0);
        assert(stats.mean_velocity == 0.0);
        assert(!stats.terminated);
        assert(!stats.truncated);
    }

    // Verify cumulative reward and running means.
    {
        rl::EpisodeStatisticsTracker tracker;

        const Eigen::Vector3d target{0.0, 0.0, 2.0};

        physics::DroneState state;
        state.position = target;

        tracker.RecordStep(
            state, target, 1.0, 0.1,
            false, false, rl::TerminationReason::None
        );

        state.position.x() = 2.0;
        state.velocity.x() = 4.0;

        tracker.RecordStep(
            state, target, -2.0, 0.2,
            false, false, rl::TerminationReason::None
        );

        const auto& stats = tracker.GetStatistics();

        assert(stats.steps == 2);
        assert(std::abs(stats.duration - 0.2) < tolerance);
        assert(std::abs(stats.cumulative_reward + 1.0) < tolerance);

        assert(std::abs(stats.mean_position_error - 1.0) < tolerance);
        assert(std::abs(stats.maximum_position_error - 2.0) < tolerance);

        assert(std::abs(stats.mean_velocity - 2.0) < tolerance);
        assert(std::abs(stats.maximum_velocity - 4.0) < tolerance);
    }

    // Terminated episodes must preserve their failure reason.
    {
        rl::EpisodeStatisticsTracker tracker;

        physics::DroneState state;
        state.position.z() = -0.1;

        tracker.RecordStep(
            state,
            Eigen::Vector3d(0.0, 0.0, 2.0),
            -20.0,
            0.1,
            true,
            false,
            rl::TerminationReason::MinimumAltitude
        );

        const auto& stats = tracker.GetStatistics();

        assert(stats.terminated);
        assert(!stats.truncated);
        assert(stats.termination_reason ==
               rl::TerminationReason::MinimumAltitude);

        bool threw = false;

        try
        {
            tracker.RecordStep(
                state,
                Eigen::Vector3d(0.0, 0.0, 2.0),
                0.0,
                0.2,
                false,
                false,
                rl::TerminationReason::None
            );
        }
        catch (const std::logic_error&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Reset must clear all accumulated statistics.
    {
        rl::EpisodeStatisticsTracker tracker;

        physics::DroneState state;

        tracker.RecordStep(
            state,
            Eigen::Vector3d::Zero(),
            1.0,
            0.1,
            false,
            false,
            rl::TerminationReason::None
        );

        tracker.Reset();

        const auto& stats = tracker.GetStatistics();

        assert(stats.steps == 0);
        assert(stats.cumulative_reward == 0.0);
        assert(stats.duration == 0.0);
    }

    // Integration: environment statistics must match actual steps.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.maximum_episode_time = 0.1;

        rl::FlightEnvironment environment(parameters);

        const physics::MotorCommands commands{
            0.0, 0.0, 0.0, 0.0
        };

        double expected_reward = 0.0;

        for (int i = 0; i < 12; ++i)
        {
            const auto result = environment.Step(commands);
            expected_reward += result.reward;
        }

        const auto& stats = environment.GetEpisodeStatistics();

        assert(stats.steps == 12);
        assert(std::abs(stats.duration - 0.1) < tolerance);
        assert(std::abs(
            stats.cumulative_reward - expected_reward
        ) < tolerance);

        assert(stats.truncated);
        assert(!stats.terminated);

        environment.Reset(123);

        const auto& reset_stats =
            environment.GetEpisodeStatistics();

        assert(reset_stats.steps == 0);
        assert(reset_stats.duration == 0.0);
        assert(reset_stats.cumulative_reward == 0.0);
    }

    std::cout << "RL episode statistics tests PASSED\n";

    return 0;
}
