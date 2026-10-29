// /src/rl/episode_termination.cpp

#include "aerial/rl/episode_termination.hpp"

#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    void ValidateEpisodeTerminationParameters(
            const EpisodeTerminationParameters& parameters
            )
    {
        if (!std::isfinite(parameters.minimum_altitude) ||
            !std::isfinite(parameters.maximum_altitude) ||
            !std::isfinite(parameters.maximum_horizontal_distance) ||
            !std::isfinite(parameters.maximum_tilt_angle) ||
            !std::isfinite(parameters.failure_penalty) ||
            parameters.maximum_altitude <= parameters.minimum_altitude ||
            parameters.maximum_horizontal_distance <= 0.0 ||
            parameters.maximum_tilt_angle <= 0.0 ||
            parameters.maximum_tilt_angle > 3.14159265358979323846 ||
            parameters.failure_penalty > 0.0)
        {
            throw std::invalid_argument(
                "Invalid episode termination parameters"
            );
        }
    }

    EpisodeTerminationResult CheckEpisodeTermination(
            const physics::DroneState& state,
            const EpisodeTerminationParameters& parameters,
            const Eigen::Vector3d& target_position
            )
    {
        ValidateEpisodeTerminationParameters(parameters);

        if (!target_position.allFinite()) {
            throw std::invalid_argument("Termination target must be finite");
        }

        if (!state.position.allFinite() ||
            !state.velocity.allFinite() ||
            !state.angular_velocity.allFinite() ||
            !state.orientation.coeffs().allFinite() ||
            state.orientation.norm() < 1e-12) {
            return {true, TerminationReason::NonFiniteState};
        }

        if (state.position.z() < parameters.minimum_altitude) {
            return {true, TerminationReason::MinimumAltitude};
        }

        if (state.position.z() > parameters.maximum_altitude) {
            return {true, TerminationReason::MaximumAltitude};
        }

        const Eigen::Vector2d horizontal_error =
            (state.position - target_position).head<2>();

        if (horizontal_error.norm() > parameters.maximum_horizontal_distance) {
            return {true, TerminationReason::HorizontalBoundary};
        }

        const Eigen::Quaterniond orientation = state.orientation.normalized();

        const Eigen::Vector3d body_up = orientation * Eigen::Vector3d::UnitZ();

        const double cosine_limit = std::cos(parameters.maximum_tilt_angle);

        if (body_up.z() < cosine_limit) {
            return {true, TerminationReason::ExcessiveTilt};
        }

        return {};
    }
}
