// /tests/motor_commands_test.cpp

#include <cmath>
#include <iostream>

#include "aerial/physics/motor_commands.hpp"

namespace
{
    bool Near(double a, double b)
    {
        return std::abs(a - b) < 1e-9;
    }
}

int main()
{
    using aerial::physics::DroneParameters;
    using aerial::physics::MotorCommands;
    using aerial::physics::CommandsToThrust;

    DroneParameters parameters;
    parameters.max_motor_thrust = 5.0;

    // Test 1: Normalized commands.
    const MotorCommands commands = {
        0.0, 0.25, 0.5, 1.0
    };

    const auto thrusts =
        CommandsToThrust(parameters, commands);

    if (!Near(thrusts[0], 0.0) ||
        !Near(thrusts[1], 1.25) ||
        !Near(thrusts[2], 2.5) ||
        !Near(thrusts[3], 5.0))
    {
        std::cerr << "Command conversion test failed\n";
        return 1;
    }

    // Test 2: Commands outside [0, 1] are clamped.
    const MotorCommands invalid_commands = {
        -1.0, 0.0, 1.0, 2.0
    };

    const auto clamped =
        CommandsToThrust(parameters, invalid_commands);

    if (!Near(clamped[0], 0.0) ||
        !Near(clamped[1], 0.0) ||
        !Near(clamped[2], 5.0) ||
        !Near(clamped[3], 5.0))
    {
        std::cerr << "Command clamping test failed\n";
        return 1;
    }

    // Test 3: Zero commands produce zero thrust.
    const MotorCommands zero_commands = {
        0.0, 0.0, 0.0, 0.0
    };

    const auto zero_thrusts =
        CommandsToThrust(parameters, zero_commands);

    for (double thrust : zero_thrusts)
    {
        if (!Near(thrust, 0.0))
        {
            std::cerr << "Zero command test failed\n";
            return 1;
        }
    }

    // Test 4: Invalid maximum thrust cannot produce
    // negative motor thrust.
    parameters.max_motor_thrust = -5.0;

    const auto invalid_thrusts =
        CommandsToThrust(parameters, commands);

    for (double thrust : invalid_thrusts)
    {
        if (!Near(thrust, 0.0))
        {
            std::cerr << "Invalid maximum thrust test failed\n";
            return 1;
        }
    }

    std::cout << "Motor command tests passed\n";

    return 0;
}
