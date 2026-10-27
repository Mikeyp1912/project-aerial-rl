// /tests/wind_simulation_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "aerial/physics/aerodynamic_drag.hpp"
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

    DroneParameters parameters;
    parameters.linear_drag = Eigen::Vector3d(0.5, 1.0, 2.0);

    DroneState state;

    // Test 1: Wind generates force on a stationary drone.
    const Eigen::Vector3d wind(4.0, -2.0, 1.0);

    const Eigen::Vector3d force =
        CalculateAerodynamicDrag(state, parameters, wind);

    assert(Near(force.x(), 2.0));
    assert(Near(force.y(), -2.0));
    assert(Near(force.z(), 2.0));

    // Test 2: No drag when the drone moves with the air.
    state.velocity = wind;

    assert(
        CalculateAerodynamicDrag(state, parameters, wind).isZero()
    );

    // Test 3: Wind affects full simulation dynamics.
    Simulation simulation(parameters);

    simulation.SetWindVelocity(Eigen::Vector3d(2.0, 0.0, 0.0));

    assert(Near(simulation.GetWindVelocity().x(), 2.0));

    const MotorCommands zero_commands{};
    constexpr double dt = 1.0 / 120.0;

    simulation.Step(zero_commands, dt);

    // Initial horizontal velocity is zero.
    // Drag force in X = 0.5 * 2.0 = 1 N.
    const double expected_velocity_x =
        (1.0 / parameters.mass) * dt;

    assert(Near(
        simulation.GetDroneState().velocity.x(),
        expected_velocity_x
    ));

    // Test 4: Reset preserves wind configuration.
    simulation.Reset();

    assert(Near(simulation.GetWindVelocity().x(), 2.0));
    assert(Near(simulation.GetDroneState().velocity.x(), 0.0));

    // Test 5: Invalid wind configuration is rejected.
    simulation.SetWindVelocity(
        Eigen::Vector3d(
            std::numeric_limits<double>::quiet_NaN(),
            0.0,
            0.0
        )
    );

    assert(Near(simulation.GetWindVelocity().x(), 2.0));

    // Test 6: Default simulation has no wind.
    Simulation default_simulation;

    assert(default_simulation.GetWindVelocity().isZero());

    std::cout << "Wind simulation tests PASSED\n";

    return 0;
}
