// /tests/motor_model_test.cpp

#include <array>
#include <cmath>
#include <iostream>

#include "aerial/physics/motor_model.hpp"

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
    using aerial::physics::CalculateMotorOutput;

    DroneParameters parameters;

    // Equal thrust should produce zero net torque.
    const std::array<double, 4> hover = {
        2.4525, 2.4525, 2.4525, 2.4525
    };

    const auto hover_output =
        CalculateMotorOutput(parameters, hover);

    if (!Near(hover_output.force.z(), 9.81) ||
        hover_output.torque.norm() > 1e-9)
    {
        std::cerr << "Hover motor test failed\n";
        return 1;
    }

    // Increasing thrust on the left should produce positive roll torque.
    const std::array<double, 4> roll = {
        3.0, 3.0, 2.0, 2.0
    };

    const auto roll_output =
        CalculateMotorOutput(parameters, roll);

    if (roll_output.torque.x() <= 0.0)
    {
        std::cerr << "Roll torque test failed\n";
        return 1;
    }

    // Increasing thrust at the front should produce negative pitch torque.
    const std::array<double, 4> pitch = {
        3.0, 2.0, 2.0, 3.0
    };

    const auto pitch_output =
        CalculateMotorOutput(parameters, pitch);

    if (pitch_output.torque.y() >= 0.0)
    {
        std::cerr << "Pitch torque test failed\n";
        return 1;
    }

    // Differential thrust between spin directions produces yaw torque.
    const std::array<double, 4> yaw = {
        3.0, 2.0, 3.0, 2.0
    };

    const auto yaw_output =
        CalculateMotorOutput(parameters, yaw);

    if (Near(yaw_output.torque.z(), 0.0))
    {
        std::cerr << "Yaw torque test failed\n";
        return 1;
    }

    // Thrust should be limited to the configured maximum.
    const std::array<double, 4> excessive = {
        100.0, 100.0, 100.0, 100.0
    };

    const auto excessive_output =
        CalculateMotorOutput(parameters, excessive);

    if (!Near(
        excessive_output.force.z(),
        4.0 * parameters.max_motor_thrust))
    {
        std::cerr << "Motor thrust limit test failed\n";
        return 1;
    }

    std::cout << "Motor model tests passed\n";
    return 0;
}
