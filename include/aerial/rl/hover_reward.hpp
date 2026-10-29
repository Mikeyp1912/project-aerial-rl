// /include/aerial/rl/hover_reward.hpp

#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_state.hpp"

namespace aerial::rl
{
    struct HoverRewardParameters
    {
        Eigen::Vector3d target_position{0.0, 0.0, 2.0};

        double alive_reward = 1.0;

        double position_weight = 1.0;
        double velocity_weight = 0.2;
        double tilt_weight = 0.5;
        double angular_velocity_weight = 0.1;
    };

    struct HoverRewardComponents
    {
        double alive = 0.0;
        double position = 0.0;
        double velocity = 0.0;
        double tilt = 0.0;
        double angular_velocity = 0.0;

        double total = 0.0;
    };

    HoverRewardComponents CalculateHoverReward(
            const physics::DroneState& state,
            const HoverRewardParameters& parameters
            );
}
