// /src/physics/motor_commands.cpp

#include "aerial/physics/motor_commands.hpp"

#include <algorithm>

namespace aerial::physics
{
    std::array<double, 4> CommandsToThrust(
            const DroneParameters& paramters,
            const MotorCommands& commands
            )
    {
        std::array<double, 4> thrusts{};

        const double max_thrust = std::max(0.0, paramters.max_motor_thrust);

        for (std::size_t i = 0; i < commands.size(); ++i) {
            const double command = std::clamp(commands[i], 0.0, 1.0);

            thrusts[i] = command * max_thrust;
        }

        return thrusts;
    }
}
