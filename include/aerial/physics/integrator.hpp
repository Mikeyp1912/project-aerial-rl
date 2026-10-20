// /include/aerial/physics/integrator.hpp
#pragma once

#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/drone_state.hpp"

namespace aerial::physics
{
    inline constexpr double gravity = 9.81;

    void Integrate(
            DroneState& state,
            const DroneParameters& parameters,
            const Eigen::Vector3d& Force,
            double dt
            );
}
