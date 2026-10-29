// /include/aerial/rl/episode_statistics.hpp

#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "aerial/physics/drone_state.hpp"
#include "aerial/rl/episode_termination.hpp"

namespace aerial::rl
{
    struct EpisodeStatistics
    {
        std::uint64_t steps = 0;

        double duration = 0.0;
        double cumulative_reward = 0.0;

        double mean_position_error = 0.0;
        double maximum_position_error = 0.0;

        double mean_velocity = 0.0;
        double maximum_velocity = 0.0;

        bool terminated = false;
        bool truncated = false;

        TerminationReason termination_reason = TerminationReason::None;
    };

    class EpisodeStatisticsTracker
    {
        public:
            void Reset();

            void RecordStep(
                    const physics::DroneState& state,
                    const Eigen::Vector3d& target_position,
                    double reward,
                    double simulation_time,
                    bool terminated,
                    bool truncated,
                    TerminationReason reason
                    );

            const EpisodeStatistics& GetStatistics() const;

        private:
            EpisodeStatistics statistics_;

            double position_error_sum_ = 0.0;
            double velocity_sum_ = 0.0;
    };
}
