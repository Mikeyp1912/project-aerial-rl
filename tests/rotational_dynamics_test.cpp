// /tests/rotational_dynamics_test.cpp

#include <cmath>
#include <iostream>

#include "aerial/physics/rotational_dynamics.hpp"

namespace
{
    bool Near(double a, double b, double tolerance = 1e-9)
    {
        return std::abs(a - b) < tolerance;
    }
}

int main()
{
    using aerial::physics::DroneState;
    using aerial::physics::DroneParameters;
    using aerial::physics::IntegrateRotation;

    DroneParameters parameters;

    // Test 1: Zero torque and zero angular velocity.
    DroneState stationary;

    IntegrateRotation(
        stationary,
        parameters,
        Eigen::Vector3d::Zero(),
        0.1
    );

    if (!Near(stationary.angular_velocity.norm(), 0.0) ||
        !Near(stationary.orientation.w(), 1.0))
    {
        std::cerr << "Stationary rotation test failed\n";
        return 1;
    }

    // Test 2: Constant torque about the X axis.
    DroneState rolling;

    const Eigen::Vector3d roll_torque(0.01, 0.0, 0.0);

    constexpr double dt = 1.0 / 120.0;

    for (int i = 0; i < 120; ++i)
    {
        IntegrateRotation(
            rolling,
            parameters,
            roll_torque,
            dt
        );
    }

    // alpha_x = torque_x / inertia_x = 1 rad/s^2.
    // After one second, omega_x should be 1 rad/s.
    if (!Near(rolling.angular_velocity.x(), 1.0))
    {
        std::cerr << "Roll angular velocity test failed\n";
        return 1;
    }

    // Semi-implicit Euler angular displacement.
    const double expected_angle = 0.5 * (1.0 + dt);

    const Eigen::Quaterniond expected_orientation(
        Eigen::AngleAxisd(
            expected_angle,
            Eigen::Vector3d::UnitX()
        )
    );

    if (rolling.orientation.angularDistance(expected_orientation) > 1e-8)
    {
        std::cerr << "Roll orientation test failed\n";
        return 1;
    }

    // Test 3: Orientation must remain normalized.
    if (!Near(rolling.orientation.norm(), 1.0))
    {
        std::cerr << "Quaternion normalization test failed\n";
        return 1;
    }

    // Test 4: Invalid timestep must not change state.
    const Eigen::Vector3d previous_velocity =
        rolling.angular_velocity;

    const Eigen::Quaterniond previous_orientation =
        rolling.orientation;

    IntegrateRotation(
        rolling,
        parameters,
        roll_torque,
        0.0
    );

    if ((rolling.angular_velocity - previous_velocity).norm() > 1e-9 ||
        rolling.orientation.angularDistance(previous_orientation) > 1e-9)
    {
        std::cerr << "Invalid timestep test failed\n";
        return 1;
    }

    // Test 5: Zero inertia must not change state.
    DroneParameters invalid_parameters;
    invalid_parameters.inertia.x() = 0.0;

    IntegrateRotation(
        rolling,
        invalid_parameters,
        roll_torque,
        dt
    );

    if ((rolling.angular_velocity - previous_velocity).norm() > 1e-9 ||
        rolling.orientation.angularDistance(previous_orientation) > 1e-9)
    {
        std::cerr << "Invalid inertia test failed\n";
        return 1;
    }

    std::cout << "Rotational dynamics tests passed\n";

    return 0;
}
