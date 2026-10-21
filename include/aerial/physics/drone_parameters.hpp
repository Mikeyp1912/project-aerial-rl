// /include/aerial/physics/drone_parameters.hpp
#pragma once

#include <Eigen/Core>

namespace aerial::physics
{
    struct DroneParameters
    {
        // Total drone mass (kg) 
        double mass = 1.0;

        // Principle moments of inertia (kg m^2)
        // X = roll, Y = pitch, Z = yaw
        Eigen::Vector3d inertia = {0.01, 0.01, 0.02};
    };
}
