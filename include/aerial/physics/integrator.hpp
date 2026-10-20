// /include/aerial/physics/integrator.hpp
#pragma once

#include "aerial/physics/drone_state.hpp"

namespace aerial::physics
{
    inline constexpr double gravity = 9.81;

    void Integrate(
            DroneState& state,
            double dt,
            const Eigen::Vector3d& acceleration = Eigen::Vector3d(0.0, 0.0, -gravity));
}
