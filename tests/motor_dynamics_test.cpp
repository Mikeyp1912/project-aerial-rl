// /tests/motor_dynamics_test.cpp



#include <array>
#include <cmath>
#include <iostream>

#include "aerial/physics/motor_dynamics.hpp"

namespace
{
    bool Near(double a, double b, double tolerance = 1e-9)
    {
        return std::abs(a - b) < tolerance;
    }
}

int main()
{
    using aerial::physics::DroneParameters;
    using aerial::physics::MotorState;
    using aerial::physics::UpdateMotorDynamics;

    DroneParameters parameters;
    parameters.max_motor_thrust = 5.0;
    parameters.motor_time_constant = 0.05;

    MotorState motors;

    const std::array<double, 4> targets = {
        5.0, 4.0, 3.0, 2.0
    };

    // One time constant should reach about 63.2% of target.
    UpdateMotorDynamics(
        motors,
        parameters,
        targets,
        0.05
    );

    const double fraction = 1.0 - std::exp(-1.0);

    for (std::size_t i = 0; i < targets.size(); ++i)
    {
        if (!Near(motors.thrusts[i], targets[i] * fraction))
        {
            std::cerr << "Motor response test failed\n";
            return 1;
        }
    }

    // Repeated steps should approach the target.
    constexpr double dt = 1.0 / 120.0;

    for (int i = 0; i < 240; ++i)
    {
        UpdateMotorDynamics(motors, parameters, targets, dt);
    }

    for (std::size_t i = 0; i < targets.size(); ++i)
    {
        if (!Near(motors.thrusts[i], targets[i]))
        {
            std::cerr << "Motor convergence test failed\n";
            return 1;
        }
    }

    // Zero target should make thrust decay.
    const std::array<double, 4> zero_targets{};

    UpdateMotorDynamics(
        motors,
        parameters,
        zero_targets,
        0.05
    );

    if (!Near(motors.thrusts[0], 5.0 * std::exp(-1.0)))
    {
        std::cerr << "Motor decay test failed\n";
        return 1;
    }

    // Invalid timestep must not change the state.
    const auto previous = motors.thrusts;

    UpdateMotorDynamics(motors, parameters, targets, 0.0);
    UpdateMotorDynamics(motors, parameters, targets, -1.0);

    for (std::size_t i = 0; i < previous.size(); ++i)
    {
        if (!Near(motors.thrusts[i], previous[i]))
        {
            std::cerr << "Invalid timestep test failed\n";
            return 1;
        }
    }

    // Targets above maximum must be clamped.
    MotorState limited;

    const std::array<double, 4> excessive = {
        100.0, 100.0, 100.0, 100.0
    };

    UpdateMotorDynamics(limited, parameters, excessive, 10.0);

    for (double thrust : limited.thrusts)
    {
        if (!Near(thrust, parameters.max_motor_thrust))
        {
            std::cerr << "Motor thrust limit test failed\n";
            return 1;
        }
    }

    std::cout << "Motor dynamics tests passed\n";
    return 0;
}
