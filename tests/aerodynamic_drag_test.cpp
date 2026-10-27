// /tests/aerodynamic_drag_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>

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
    DroneState state;

    // Default coefficients preserve drag-free physics.
    state.velocity = Eigen::Vector3d(3.0, -2.0, 1.0);

    assert(CalculateAerodynamicDrag(state, parameters).isZero());

    // Drag must oppose velocity in each axis.
    parameters.linear_drag = Eigen::Vector3d(0.5, 1.0, 2.0);

    const Eigen::Vector3d drag =
        CalculateAerodynamicDrag(state, parameters);

    assert(Near(drag.x(), -1.5));
    assert(Near(drag.y(), 2.0));
    assert(Near(drag.z(), -2.0));

    // Stationary drone experiences no drag.
    state.velocity.setZero();
    assert(CalculateAerodynamicDrag(state, parameters).isZero());

    // Verify that drag affects the full simulation.
    DroneParameters drag_parameters;
    drag_parameters.linear_drag = Eigen::Vector3d(1.0, 0.0, 0.0);

    Simulation simulation(drag_parameters);

    auto initial_state = simulation.GetDroneState();
    initial_state.velocity = Eigen::Vector3d(2.0, 0.0, 0.0);

    simulation.Reset(initial_state);

    const MotorCommands zero_commands{};
    constexpr double dt = 1.0 / 120.0;

    simulation.Step(zero_commands, dt);

    const double expected_velocity =
        2.0 - (2.0 / drag_parameters.mass) * dt;

    assert(Near(
        simulation.GetDroneState().velocity.x(),
        expected_velocity
    ));

    std::cout << "Aerodynamic drag tests PASSED\n";
    return 0;
}
