// /src/physics/aerodynamic_drag.cpp

#include "aerial/physics/aerodynamic_drag.hpp"

namespace aerial::physics
{
    Eigen::Vector3d CalculateAerodynamicDrag(
            const DroneState& state,
            const DroneParameters& parameters,
            const Eigen::Vector3d& wind_velocity
            )
    {
        const Eigen::Vector3d relative_velocity = state.velocity - wind_velocity;

        return -parameters.linear_drag.cwiseProduct(relative_velocity);
    }
}
