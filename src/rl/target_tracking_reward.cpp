

#include "aerial/rl/target_tracking_reward.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    double CalculateTargetTrackingReward(
        const physics::DroneState& state,
        const Eigen::Vector3d& target_position,
        const TargetTrackingRewardParameters& parameters
    )
    {
        if (!state.position.allFinite() ||
            !state.velocity.allFinite() ||
            !state.angular_velocity.allFinite() ||
            !state.orientation.coeffs().allFinite() ||
            state.orientation.norm() < 1e-12 ||
            !target_position.allFinite())
        {
            throw std::invalid_argument(
                "Invalid target tracking reward state"
            );
        }

        const double weights[] = {
            parameters.position_weight,
            parameters.velocity_weight,
            parameters.tilt_weight,
            parameters.angular_velocity_weight
        };

        if (!std::isfinite(parameters.position_sigma) ||
            parameters.position_sigma <= 0.0)
        {
            throw std::invalid_argument(
                "Position sigma must be finite and positive"
            );
        }

        for (double weight : weights)
        {
            if (!std::isfinite(weight) || weight < 0.0)
            {
                throw std::invalid_argument(
                    "Reward weights must be finite and non-negative"
                );
            }
        }

        const double squared_error =
            (state.position - target_position).squaredNorm();

        const double position_reward =
            parameters.position_weight *
            std::exp(
                -squared_error /
                (2.0 * parameters.position_sigma *
                 parameters.position_sigma)
            );

        const double velocity_penalty =
            parameters.velocity_weight *
            state.velocity.squaredNorm();

        const Eigen::Vector3d body_up =
            state.orientation.normalized() *
            Eigen::Vector3d::UnitZ();

        const double tilt_error =
            1.0 - std::clamp(body_up.z(), -1.0, 1.0);

        const double tilt_penalty =
            parameters.tilt_weight * tilt_error;

        const double angular_penalty =
            parameters.angular_velocity_weight *
            state.angular_velocity.squaredNorm();

        const double reward =
            position_reward -
            velocity_penalty -
            tilt_penalty -
            angular_penalty;

        if (!std::isfinite(reward))
        {
            throw std::invalid_argument(
                "Non-finite target tracking reward"
            );
        }

        return reward;
    }
}
