

#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_state.hpp"

namespace aerial::rl
{
    struct TargetTrackingRewardParameters
    {
        double position_sigma = 1.0;
        double position_weight = 1.0;
        double velocity_weight = 0.02;
        double tilt_weight = 0.05;
        double angular_velocity_weight = 0.01;
    };

    double CalculateTargetTrackingReward(
        const physics::DroneState& state,
        const Eigen::Vector3d& target_position,
        const TargetTrackingRewardParameters& parameters = {}
    );
}
