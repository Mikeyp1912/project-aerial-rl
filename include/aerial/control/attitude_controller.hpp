// /include/aerial/control/attitude_controller.hpp

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace aerial::control
{
    struct AttitudeControllerParameters
    {
        double kp = 4.0;
        double max_angular_rate = 4.0;
    };

    Eigen::Vector3d CalculateDesiredAngularRate(
            const Eigen::Quaterniond& current_orientation,
            const Eigen::Quaterniond& target_orientation,
            const AttitudeControllerParameters& parameters
            );
}

