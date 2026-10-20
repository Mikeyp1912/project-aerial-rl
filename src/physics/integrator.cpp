// src/physics/integrator.cpp
#include "aerial/physics/integrator.hpp"

namespace aerial::physics
{
    void Integrate(
            DroneState& state,
            double dt,
            const Eigen::Vector3d& acceleration)
    {
        if (dt <= 0.0) {
            return;
        }

        // Semi-implicit Euler integration
        state.velocity += acceleration * dt;
        state.position += state.velocity * dt;
    }
}
