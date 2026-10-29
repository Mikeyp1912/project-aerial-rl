// /include/aerial/rl/policy_runner.hpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aerial/rl/policy_runner.hpp"

namespace aerial::rl
{
    struct PolicyEvaluationParameters
    {
        std::size_t episode_count = 10;
        std::uint32_t base_seed = 42;
    };

    struct PolicyEvaluationResult
    {
        std::vector<EpisodeStatistics> episodes;

        double mean_reward = 0.0;
        double minimum_reward = 0.0;
        double maximum_reward = 0.0;

        double mean_duration = 0.0;
        double mean_position_error = 0.0;
        double mean_velocity = 0.0;

        double termination_rate = 0.0;
        double truncation_rate = 0.0;

        std::size_t minimum_altitude_failures = 0;
        std::size_t maximum_altitude_failures = 0;
        std::size_t horizontal_boundary_failures = 0;
        std::size_t excessive_tilt_failures = 0;
        std::size_t non_finite_failures = 0;
    };

    PolicyEvaluationResult EvaluatePolicy(
        FlightEnvironment& environment,
        Policy& policy,
        const PolicyEvaluationParameters& parameters = {}
    );
}

