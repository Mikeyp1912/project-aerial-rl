// /src/rl/epsiode_statistics.cpp

#include "aerial/rl/episode_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    void EpisodeStatisticsTracker::Reset()
    {
        statistics_ = EpisodeStatistics{};

        position_error_sum_ = 0.0;
        velocity_sum_ = 0.0;
    }

    void EpisodeStatisticsTracker::RecordStep(
            const physics::DroneState& state,
            const Eigen::Vector3d& target_position,
            double reward,
            double simulation_time,
            bool terminated,
            bool truncated,
            TerminationReason reason
            )
    {
        if (!std::isfinite(reward) ||
            !std::isfinite(simulation_time) ||
            simulation_time < 0.0 ||
            simulation_time <= statistics_.duration ||
            !target_position.allFinite())
        {
            throw std::invalid_argument(
                "Invalid episode statistics input"
            );
        }

        if (statistics_.terminated || statistics_.truncated)
        {
            throw std::logic_error(
                "Cannot record steps after an episode has finished"
            );
        }

        if (!terminated && reason != TerminationReason::None)
        {
            throw std::invalid_argument(
                "Termination reason requires a terminated episode"
            );
        }

        if (terminated && reason == TerminationReason::None)
        {
            throw std::invalid_argument(
                "Terminated episode requires a reason"
            );
        }

        const bool finite_state =
            state.position.allFinite() &&
            state.velocity.allFinite();

        if (!finite_state && reason != TerminationReason::NonFiniteState)
        {
            throw std::invalid_argument(
                "Non-finite state requires NonFiniteState termination"
            );
        }

        // Accumulate the reward and step count even if the final
        // physics state is non-finite
        statistics_.steps++;
        statistics_.duration = simulation_time;
        statistics_.cumulative_reward += reward;

        if (finite_state) {
            const double position_error = (state.position - target_position).norm();

            const double velocity = state.velocity.norm();

            if (std::isfinite(position_error) &&
                    std::isfinite(velocity)) {
                position_error_sum_ += position_error;
                velocity_sum_ += velocity;

                statistics_.maximum_position_error =
                    std::max(
                            statistics_.maximum_position_error,
                            position_error
                            );

                statistics_.maximum_velocity =
                    std::max(
                            statistics_.maximum_velocity,
                            velocity
                            );
            }
        }

        // The mean uses the number of recorded steps
        statistics_.mean_position_error =
            position_error_sum_ / static_cast<double>(statistics_.steps);

        statistics_.mean_velocity =
            velocity_sum_ / static_cast<double>(statistics_.steps);

        statistics_.terminated = terminated;
        statistics_.truncated = truncated;
        statistics_.termination_reason = reason;
    }

    const EpisodeStatistics& EpisodeStatisticsTracker::GetStatistics() const
    {
        return statistics_;
    }
}
