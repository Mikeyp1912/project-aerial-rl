// /include/aerial/physics/motor_model.hpp

#pragma once

#include <array>
#include <Eigen/Core>

#include "aerial/physics/drone_parameters.hpp"

namespace aerial::physics
{
    struct MotorOutput
    {
        // Total force in body coordinates (N)
        Eigen::Vector3d force = Eigen::Vector3d::Zero();

        // Total torque in body coordinates (N m)
        Eigen::Vector3d torque = Eigen::Vector3d::Zero();
    };

    MotorOutput CalculateMotorOutput(
            const DroneParameters& parameters,
            const std::array<double, 4>& motor_thrusts
            );
}
