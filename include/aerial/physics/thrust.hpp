// /include/aerial/physics/thrust.hpp
#pragma once

#include <Eigen/Core>

#include "aerial/physics/drone_state.hpp"

namespace aerial::physics
{
    Eigen::Vector3d CalculateThrustForce(
            const DroneState& state,
            double thrust
            );
}
