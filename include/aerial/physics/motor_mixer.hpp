// /include/aerial/physics/motor_mixer.hpp

#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/motor_commands.hpp"

namespace aerial::physics
{
    struct MotorMixerOutput
    {
        MotorCommands commands{};
        bool saturated = false;
    };

    MotorMixerOutput MixMotors(
            const DroneParameters& parameters,
            double collective_thrust,
            const Eigen::Vector3d& body_torque
            );
}
