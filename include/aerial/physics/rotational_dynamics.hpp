// /include/aerial/physics/rotational_dynamics.hpp

#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_state.hpp"
#include "aerial/physics/drone_parameters.hpp"

namespace aerial::physics
{
    void IntegrateRotation(
            DroneState& state,
            const DroneParameters& parameters,
            const Eigen::Vector3d& torque,
            double dt
            );
}
