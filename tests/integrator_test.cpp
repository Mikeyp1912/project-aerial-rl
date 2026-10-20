// /tests/integrator_test.cpp

#include <cmath>
#include <iostream>

#include "aerial/physics/integrator.hpp"

namespace
{
    bool Near(double a, double b)
    {
        return std::abs(a - b) < 1e-9;
    }

    bool CheckPosition(
            const aerial::physics::DroneState& state,
            double x,
            double y,
            double z
            ) 
    {
        return Near(state.position.x(), x) && Near(state.position.y(), y) && Near(state.position.z(), z);
    }
}

int main() {
    using aerial::physics::DroneState;
    using aerial::physics::Integrate;
    using aerial::physics::DroneParameters;

    const DroneParameters parameters;
    const Eigen::Vector3d hover_force(
        0.0, 0.0, parameters.mass * aerial::physics::gravity
    );

    DroneState state;
    state.position = {0.0, 0.0, 1.0};
    state.velocity = {2.0, -1.0, 0.5};

    Integrate(state, parameters, hover_force, 0.5);

    if (!CheckPosition(state, 1.0, -0.5, 1.25)) {
        std::cerr << "Single-step integration failed\n";
        return 1;
    }

    DroneState repeated;
    repeated.position = {0.0, 0.0, 1.0};
    repeated.velocity = {2.0, -1.0, 0.5};

    constexpr double dt = 1.0 / 120.0;

    for (int i = 0; i < 120; i++) {
        Integrate(repeated, parameters, hover_force, dt);
    }

    if (!CheckPosition(repeated, 2.0, -1.0, 1.5)) {
        std::cerr << "Fixed-step integration failed\n";
        return 1;
    }

    const Eigen::Vector3d previous = repeated.position;

    Integrate(repeated, parameters, hover_force, 0.0);
    Integrate(repeated, parameters, hover_force, -1.0);

    if ((repeated.position - previous).norm() > 1e-9) {
        std::cerr << "Non-positive timestep changed state\n";
        return 1;
    }

    // Test gravity after one second using 120 fixed steps
    DroneState falling;
    falling.position = {0.0, 0.0, 10.0};

    constexpr double gravity_dt = 1.0 / 120.0;

    for (int i = 0; i < 120; i++) {
        Integrate(
            falling,
            parameters,
            Eigen::Vector3d::Zero(),
            gravity_dt
        );
    }

    if (!Near(falling.velocity.z(), -9.81)) {
        std::cerr << "Gravity velocity test failed\n";
        return 1;
    }

    // Semi-implicit Euler gives a slightly different position
    // from the continuous analytical solution
    const double expected_height = 10.0 - 0.5 * 9.81 * (1.0 + gravity_dt);

    if (!Near(falling.position.z(), expected_height)) {
        std::cerr << "Gravity position test failed\n";
        return 1;
    }

    // Check that gravity doesn't affect X or Y
    if (!Near(falling.position.x(), 0.0) || !Near(falling.position.y(), 0.0)) {
        std::cerr << "Gravity horizontal motion test failed\n";
        return 1;
    }

    
    // Test stationary hover.
    DroneState hovering;
    hovering.position = {0.0, 0.0, 5.0};

    for (int i = 0; i < 120; ++i)
    {
        Integrate(hovering, parameters, hover_force, dt);
    }

    if (!CheckPosition(hovering, 0.0, 0.0, 5.0) ||
        !Near(hovering.velocity.z(), 0.0))
    {
        std::cerr << "Hover test failed\n";
        return 1;
    }

    // Test upward acceleration with a known net force.
    DroneState ascending;
    ascending.position = {0.0, 0.0, 5.0};

    const Eigen::Vector3d upward_force(
        0.0,
        0.0,
        parameters.mass * (aerial::physics::gravity + 2.0)
    );

    for (int i = 0; i < 120; ++i)
    {
        Integrate(ascending, parameters, upward_force, dt);
    }

    if (!Near(ascending.velocity.z(), 2.0))
    {
        std::cerr << "Upward acceleration test failed\n";
        return 1;
    }

    const double expected_ascending_height =
        5.0 + 0.5 * 2.0 * (1.0 + dt);

    if (!Near(ascending.position.z(), expected_ascending_height))
    {
        std::cerr << "Upward position test failed\n";
        return 1;
    }

    // Test mass dependence.
    DroneParameters heavy_parameters;
    heavy_parameters.mass = 2.0;

    DroneState heavy;
    Integrate(
        heavy,
        heavy_parameters,
        Eigen::Vector3d(0.0, 0.0, 19.62),
        1.0
    );

    if (!Near(heavy.velocity.z(), 0.0))
    {
        std::cerr << "Mass-dependent hover test failed\n";
        return 1;
    }

    // Invalid mass must not change the state.
    DroneParameters invalid_parameters;
    invalid_parameters.mass = 0.0;

    const Eigen::Vector3d previous_position = heavy.position;
    const Eigen::Vector3d previous_velocity = heavy.velocity;

    Integrate(
        heavy,
        invalid_parameters,
        Eigen::Vector3d::Zero(),
        1.0
    );

    if ((heavy.position - previous_position).norm() > 1e-9 ||
        (heavy.velocity - previous_velocity).norm() > 1e-9)
    {
        std::cerr << "Invalid mass test failed\n";
        return 1;
    }


    std::cout << "Integrator tests passed\n";
    return 0;
}
