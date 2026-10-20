// /include/aerial/physics/integrator.hpp
#pragma once

#include "aerial/physics/drone_state.hpp"

namespace aerial::physics
{
    void Integrate(DroneState& state, double dt);
}
