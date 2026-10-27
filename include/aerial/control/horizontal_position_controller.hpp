// /include/aerial/control/horizontal_position_controller.hpp

#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_state.hpp"

namespace aerial::control
{
    struct HorizontalPositionControllerParameters
    {
        double position_kp = 1.0;

        double max_horizontal_velocity = 3.0;
    };

    struct HorizontalPositionControllerOutput
    {
        Eigen::Vector2d desired_velocity = Eigen::Vector2d::Zero();

        Eigen::Vector2d position_error = Eigen::Vector2d::Zero();

        bool saturated = false;
    };

    HorizontalPositionControllerOutput UpdateHorizontalPositionController(
            const physics::DroneState& drone_state,
            const Eigen::Vector2d& target_position,
            const HorizontalPositionControllerParameters& parameters
            );
}
