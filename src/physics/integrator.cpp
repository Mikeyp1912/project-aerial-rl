// src/physics/integrator.cpp
#include "aerial/physics/integrator.hpp"

namespace aerial::physics
{
    void Integrate(DroneState& state, double dt)
    {
        if (dt <= 0.0) {
            return;
        }

        state.position += state.velocity * dt;
    }
}
