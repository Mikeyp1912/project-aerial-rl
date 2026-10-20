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

    Integrate(state, 0.5);

    if (!CheckPosition(state, 1.0, -0.5, 1.25)) {
        std::cerr << "Single-step integration failed\n";
        return 1;
    }

    DroneState repeated;
    repeated.position = {0.0, 0.0, 1.0};
    repeated.velocity = {2.0, -1.0, 0.5};

    constexpr double dt = 1.0 / 120.0;

    for (int i = 0; i < 120; i++) {
        Integrate(repeated, dt);
    }

    if (!CheckPosition(repeated, 2.0, -1.0, 1.5)) {
        std::cerr << "Fixed-step integration failed\n";
        return 1;
    }

    const Eigen::Vector3d previous = repeated.position;

    Integrate(repeated, 0.0);
    Integrate(repeated, -1.0);

    if ((repeated.position - previous).norm() > 1e-9) {
        std::cerr << "Non-positive timestep changed state\n";
        return 1;
    }

    std::cout << "Integrator tests passed\n";
    return 0;
}
