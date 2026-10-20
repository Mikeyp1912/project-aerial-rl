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

    DroneState state;
    state.position = {0.0, 0.0, 1.0};
    state.velocity = {2.0, -1.0, 0.5};

    Integrate(state, 0.5, Eigen::Vector3d::Zero());

    if (!CheckPosition(state, 1.0, -0.5, 1.25)) {
        std::cerr << "Single-step integration failed\n";
        return 1;
    }

    DroneState repeated;
    repeated.position = {0.0, 0.0, 1.0};
    repeated.velocity = {2.0, -1.0, 0.5};

    constexpr double dt = 1.0 / 120.0;

    for (int i = 0; i < 120; i++) {
        Integrate(repeated, dt, Eigen::Vector3d::Zero());
    }

    if (!CheckPosition(repeated, 2.0, -1.0, 1.5)) {
        std::cerr << "Fixed-step integration failed\n";
        return 1;
    }

    const Eigen::Vector3d previous = repeated.position;

    Integrate(repeated, 0.0, Eigen::Vector3d::Zero());
    Integrate(repeated, -1.0, Eigen::Vector3d::Zero());

    if ((repeated.position - previous).norm() > 1e-9) {
        std::cerr << "Non-positive timestep changed state\n";
        return 1;
    }

    // Test gravity after one second using 120 fixed steps
    DroneState falling;
    falling.position = {0.0, 0.0, 10.0};

    constexpr double gravity_dt = 1.0 / 120.0;

    for (int i = 0; i < 120; i++) {
        Integrate(falling, gravity_dt);
    }

    if (!Near(falling.velocity.z(), -9.81)) {
        std::cerr << "Gravity velocity test failed\n";
        return 1;
    }

    // Semi-implicit Euler gives a slightly different position
    // from the continuous analytical solution
    const double expected_height = 10.0 - 0.5 * 9.81 * (1.0 + gravity_dt);

    if (!Near(falling.position.z(), expected_height)) {
        std::cerr << "Gravuty position test failed\n";
        return 1;
    }

    // Check that gravity doesn't affect X or Y
    if (!Near(falling.position.x(), 0.0) || !Near(falling.position.y(), 0.0)) {
        std::cerr << "Gravity horizontal motion test failed\n";
        return 1;
    }

    std::cout << "Integrator tests passed\n";
    return 0;
}
