// /include/aerial/physics/aerodynamic_drag.hpp

#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/drone_state.hpp"

namespace aerial::physics
{
    Eigen::Vector3d CalculateAerodynamicDrag(
            const DroneState& state,
            const DroneParameters& parameters,
            const Eigen::Vector3d& wind_velocity = Eigen::Vector3d::Zero()
            );
}
