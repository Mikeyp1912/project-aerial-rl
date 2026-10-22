// /include/aerial/control/rate_controller.hpp

#pragma once

#include <Eigen/Core>

namespace aerial::control
{
    struct RateControllerParameters
    {
        // Proportional gains: N m / (rad/s)
        Eigen::Vector3d kp = Eigen::Vector3d(0.1, 0.1, 0.05);

        // Integral gains
        Eigen::Vector3d ki = Eigen::Vector3d::Zero();

        // Derivative gains
        Eigen::Vector3d kd = Eigen::Vector3d(0.01, 0.01, 0.005);

        // Maximum integral error accumulation
        Eigen::Vector3d integral_limit = Eigen::Vector3d(1.0, 1.0, 1.0);

        // Maximum absolut body torque (N m)
        Eigen::Vector3d torque_limit = Eigen::Vector3d(0.5, 0.5, 0.1);
    };

    struct RateControllerState
    {
        Eigen::Vector3d integral_error = Eigen::Vector3d::Zero();

        Eigen::Vector3d previous_angular_velocity = Eigen::Vector3d::Zero();

        bool initialized = false;
    };

    Eigen::Vector3d CalculateBodyTorque(
            const Eigen::Vector3d& desired_angular_velocity,
            const Eigen::Vector3d& actual_angular_velocity,
            double dt,
            const RateControllerParameters& parameters,
            RateControllerState& state
            );

    void ResetRateController(RateControllerState& state);
}
