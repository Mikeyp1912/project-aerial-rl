// /tests/rl_policy_evaluator_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/rl/policy_evaluator.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-9;

    // Reject an evaluation with zero episodes.
    {
        rl::FlightEnvironment environment;
        rl::ConstantPolicy policy;

        rl::PolicyEvaluationParameters parameters;
        parameters.episode_count = 0;

        bool threw = false;

        try
        {
            rl::EvaluatePolicy(environment, policy, parameters);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Short episodes should truncate consistently.
    {
        rl::FlightEnvironmentParameters environment_parameters;
        environment_parameters.maximum_episode_time = 0.05;

        rl::FlightEnvironment environment(environment_parameters);
        rl::ConstantPolicy policy;

        rl::PolicyEvaluationParameters evaluation_parameters;
        evaluation_parameters.episode_count = 5;
        evaluation_parameters.base_seed = 100;

        const auto result = rl::EvaluatePolicy(
            environment,
            policy,
            evaluation_parameters
        );

        assert(result.episodes.size() == 5);
        assert(std::abs(result.truncation_rate - 1.0) < tolerance);
        assert(std::abs(result.termination_rate) < tolerance);

        assert(result.minimum_altitude_failures == 0);
        assert(result.maximum_altitude_failures == 0);
        assert(result.horizontal_boundary_failures == 0);
        assert(result.excessive_tilt_failures == 0);
        assert(result.non_finite_failures == 0);

        double reward_sum = 0.0;

        for (const auto& episode : result.episodes)
        {
            assert(episode.steps == 6);
            assert(episode.truncated);
            assert(!episode.terminated);

            reward_sum += episode.cumulative_reward;
        }

        assert(std::abs(
            result.mean_reward - reward_sum / 5.0
        ) < tolerance);
    }

    // Identical seeds and policies must reproduce results.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.maximum_episode_time = 0.1;

        rl::FlightEnvironment environment(parameters);
        rl::ConstantPolicy policy;

        rl::PolicyEvaluationParameters evaluation;
        evaluation.episode_count = 3;
        evaluation.base_seed = 42;

        const auto first = rl::EvaluatePolicy(
            environment, policy, evaluation
        );

        const auto second = rl::EvaluatePolicy(
            environment, policy, evaluation
        );

        assert(first.episodes.size() == second.episodes.size());

        assert(std::abs(
            first.mean_reward - second.mean_reward
        ) < tolerance);

        assert(std::abs(
            first.mean_position_error -
            second.mean_position_error
        ) < tolerance);

        for (std::size_t i = 0; i < first.episodes.size(); ++i)
        {
            assert(std::abs(
                first.episodes[i].cumulative_reward -
                second.episodes[i].cumulative_reward
            ) < tolerance);
        }
    }

    // A deliberately high minimum altitude should cause failure.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.termination.minimum_altitude = 0.99;
        parameters.maximum_episode_time = 1.0;

        rl::FlightEnvironment environment(parameters);
        rl::ConstantPolicy policy;

        rl::PolicyEvaluationParameters evaluation;
        evaluation.episode_count = 3;

        const auto result = rl::EvaluatePolicy(
            environment, policy, evaluation
        );

        assert(result.termination_rate == 1.0);
        assert(result.minimum_altitude_failures == 3);
        assert(result.truncation_rate == 0.0);

        for (const auto& episode : result.episodes)
        {
            assert(episode.terminated);
            assert(episode.termination_reason ==
                   rl::TerminationReason::MinimumAltitude);
        }
    }

    std::cout << "RL policy evaluator tests PASSED\n";

    return 0;
}

