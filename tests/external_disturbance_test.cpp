// /tests/external_disturbance_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "aerial/physics/simulation_environment.hpp"

namespace
{
    bool Near(double a, double b, double tolerance = 1e-9)
    {
        return std::abs(a - b) < tolerance;
    }
}

int main()
{
    using namespace aerial::physics;

    constexpr double dt = 1.0 / 120.0;

    Simulation simulation;

    const MotorCommands zero_commands{};

    // Test 1: External force accelerates the drone.
    simulation.SetExternalForce(Eigen::Vector3d(2.0, 0.0, 0.0));

    simulation.Step(zero_commands, dt);

    const double expected_velocity =
        2.0 / simulation.GetParameters().mass * dt;

    assert(Near(
        simulation.GetDroneState().velocity.x(),
        expected_velocity
    ));

    // Test 2: Force persists across steps.
    simulation.Step(zero_commands, dt);

    assert(Near(
        simulation.GetDroneState().velocity.x(),
        2.0 * expected_velocity
    ));

    // Test 3: Clearing disturbances stops applying force.
    simulation.ClearExternalDisturbances();

    assert(simulation.GetExternalForce().isZero());
    assert(simulation.GetExternalTorque().isZero());

    const double previous_velocity =
        simulation.GetDroneState().velocity.x();

    simulation.Step(zero_commands, dt);

    assert(Near(
        simulation.GetDroneState().velocity.x(),
        previous_velocity
    ));

    // Test 4: External torque generates angular velocity.
    simulation.Reset();

    simulation.SetExternalTorque(
        Eigen::Vector3d(0.01, 0.0, 0.0)
    );

    simulation.Step(zero_commands, dt);

    const double expected_angular_velocity =
        0.01 / simulation.GetParameters().inertia.x() * dt;

    assert(Near(
        simulation.GetDroneState().angular_velocity.x(),
        expected_angular_velocity
    ));

    // Test 5: Reset clears disturbances but preserves wind.
    simulation.SetWindVelocity(Eigen::Vector3d(1.0, 0.0, 0.0));
    simulation.SetExternalForce(Eigen::Vector3d(3.0, 0.0, 0.0));

    simulation.Reset();

    assert(simulation.GetExternalForce().isZero());
    assert(simulation.GetExternalTorque().isZero());
    assert(Near(simulation.GetWindVelocity().x(), 1.0));

    // Test 6: Invalid inputs must be rejected.
    simulation.SetExternalForce(
        Eigen::Vector3d(
            std::numeric_limits<double>::quiet_NaN(),
            0.0,
            0.0
        )
    );

    assert(simulation.GetExternalForce().isZero());

    simulation.SetExternalTorque(
        Eigen::Vector3d(
            0.0,
            std::numeric_limits<double>::infinity(),
            0.0
        )
    );

    assert(simulation.GetExternalTorque().isZero());

    std::cout << "External disturbance tests PASSED\n";

    return 0;
}
