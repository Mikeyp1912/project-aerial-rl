// /src/rl/policy_evaluator.cpp

#include "aerial/rl/policy_evaluator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace aerial::rl
{
    PolicyEvaluationResult EvaluatePolicy(
            FlightEnvironment& environment,
            Policy& policy,
            const PolicyEvaluationParameters& parameters
            )
    {
        if (parameters.episode_count == 0) {
            throw std::invalid_argument("Policy evaluation requires at least one episode");
        }

        if (parameters.episode_count >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max() -
                    parameters.base_seed) + 1ULL
                    ) {
            throw std::invalid_argument("Policy evaluation seed range overflows");
        }

        PolicyRunner runner(environment, policy);

        PolicyEvaluationResult result;
        result.episodes.reserve(parameters.episode_count);

        double total_reward = 0.0;
        double total_duration = 0.0;
        double total_position_error = 0.0;
        double total_velocity = 0.0;

        std::size_t terminated_count = 0;
        std::size_t truncated_count = 0;

        for (std::size_t episode = 0; episode < parameters.episode_count; ++episode) {
            const std::uint32_t seed = parameters.base_seed + static_cast<std::uint32_t>(episode);

            runner.Reset(seed);

            while (true) {
                const FlightStepResult step = runner.Step();

                if (step.terminated || step.truncated) {
                    break;
                }
            }

            const EpisodeStatistics statistics = environment.GetEpisodeStatistics();

            result.episodes.push_back(statistics);

            total_reward += statistics.cumulative_reward;
            total_duration += statistics.duration;
            total_position_error += statistics.mean_position_error;
            total_velocity += statistics.mean_velocity;

            if (episode == 0) {
                result.minimum_reward = statistics.cumulative_reward;
                result.maximum_reward = statistics.cumulative_reward;
            }
            else {
                result.minimum_reward = std::min(
                        result.minimum_reward,
                        statistics.cumulative_reward
                        );

                result.maximum_reward = std::max(
                        result.maximum_reward,
                        statistics.cumulative_reward
                        );
            }

            if (statistics.terminated) {
                ++terminated_count;
            }

            if (statistics.truncated) {
                ++truncated_count;
            }

            switch (statistics.termination_reason) {
                case TerminationReason::MinimumAltitude:
                    ++result.minimum_altitude_failures;
                    break;

                case TerminationReason::MaximumAltitude:
                    ++result.maximum_altitude_failures;
                    break;

                case TerminationReason::HorizontalBoundary:
                    ++result.horizontal_boundary_failures;
                    break;

                case TerminationReason::ExcessiveTilt:
                    ++result.excessive_tilt_failures;
                    break;

                case TerminationReason::NonFiniteState:
                    ++result.non_finite_failures;
                    break;

                case TerminationReason::None:
                    break;
            }
        }

        const double count = static_cast<double>(parameters.episode_count);

        result.mean_reward = total_reward / count;
        result.mean_duration = total_duration / count;
        result.mean_position_error = total_position_error / count;
        result.mean_velocity = total_velocity / count;

        result.termination_rate = static_cast<double>(terminated_count) / count;

        result.truncation_rate = static_cast<double>(truncated_count) / count;

        return result;
    }
}
