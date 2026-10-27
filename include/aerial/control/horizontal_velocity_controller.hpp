// /include/aerial/control/horizontal_velocity_controller.hpp

#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_state.hpp"

namespace aerial::control
{
    struct HorizontalVelocityControllerParameters
    {
        double velocity_kp = 2.0;

        double max_horizontal_acceleration = 4.0;

        double max_tilt_angle = 0.349066; // 20 degrees
    };

    struct HorizontalVelocityControllerOutput
    {
        Eigen::Vector2d desired_acceleration = Eigen::Vector2d::Zero();

        double target_roll = 0.0;
        double target_pitch = 0.0;

        bool saturated = false;
    };

    HorizontalVelocityControllerOutput UpdateHorizontalVelocityController(
            const physics::DroneState& drone_state,
            const Eigen::Vector2d& target_velocity,
            double target_yaw,
            const HorizontalVelocityControllerParameters& parameters
            );
}
