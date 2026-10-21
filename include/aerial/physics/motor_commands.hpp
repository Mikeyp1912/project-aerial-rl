// /include/aerial/physics/motor_commands.hpp

#pragma once

#include <array>

#include "aerial/physics/drone_parameters.hpp"

namespace aerial::physics
{
    using MotorCommands = std::array<double, 4>;

    std::array<double, 4> CommandsToThrust(
            const DroneParameters& parameters,
            const MotorCommands& commands
            );
}
