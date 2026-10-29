// /include/aerial/rl/episode_termination.hpp

#pragma once

#include "aerial/physics/drone_state.hpp"

namespace aerial::rl
{
    enum class TerminationReason
    {
        None,
        MinimumAltitude,
        MaximumAltitude,
        HorizontalBoundary,
        ExcessiveTilt,
        NonFiniteState
    };

    struct EpisodeTerminationParameters
    {
        double minimum_altitude = 0.0;
        double maximum_altitude = 10.0;

        double maximum_horizontal_distance = 10.0;
        double maximum_tilt_angle = 1.3962634015954636; 

        double failure_penalty = -20.0;
    };

    struct EpisodeTerminationResult
    {
        bool terminated = false;
        TerminationReason reason = TerminationReason::None;
    };

    void ValidateEpisodeTerminationParameters(
            const EpisodeTerminationParameters& parameters
            );

    EpisodeTerminationResult CheckEpisodeTermination(
            const physics::DroneState& state,
            const EpisodeTerminationParameters& parameters,
            const Eigen::Vector3d& target_position
            );
}
