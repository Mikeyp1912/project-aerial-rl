// /src/rl/hover_reward.cpp

#include "aerial/rl/hover_reward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    HoverRewardComponents CalculateHoverReward(
            const physics::DroneState& state,
            const HoverRewardParameters& parameters
            )
    {
        if (!state.position.allFinite() ||
            !state.velocity.allFinite() ||
            !state.angular_velocity.allFinite() ||
            !state.orientation.coeffs().allFinite() ||
            state.orientation.norm() < 1e-12 ||
            !parameters.target_position.allFinite())
        {
            throw std::invalid_argument(
                "Invalid hover reward state or target"
            );
        }

        const double weights[] = {
            parameters.alive_reward,
            parameters.position_weight,
            parameters.velocity_weight,
            parameters.tilt_weight,
            parameters.angular_velocity_weight
        };

        for (double weight : weights) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument("Hover reward weights must be finite and non-negative");
            }
        }

        HoverRewardComponents reward;

        reward.alive = parameters.alive_reward;

        const Eigen::Vector3d position_error =
            state.position - parameters.target_position;

        reward.position =
            -parameters.position_weight *
            position_error.squaredNorm();

        reward.velocity =
            -parameters.velocity_weight *
            state.velocity.squaredNorm();

        const Eigen::Quaterniond orientation = state.orientation.normalized();

        // World-frame direction of the drone's body +Z axis
        const Eigen::Vector3d body_up = orientation * Eigen::Vector3d::UnitZ();

        // Zero for level flight, increasing toward inverted flight
        const double tilt_error = 1.0 - std::clamp(body_up.z(), -1.0, 1.0);

        reward.tilt = -parameters.tilt_weight * tilt_error;

        reward.angular_velocity =
            -parameters.angular_velocity_weight *
            state.angular_velocity.squaredNorm();

        reward.total =
            reward.alive +
            reward.position +
            reward.velocity +
            reward.tilt +
            reward.angular_velocity;

        if (!std::isfinite(reward.total))
        {
            throw std::invalid_argument(
                "Non-finite hover reward"
            );
        }

        return reward;
    }
}
