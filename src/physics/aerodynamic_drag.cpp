// /src/physics/aerodynamic_drag.cpp

#include "aerial/physics/aerodynamic_drag.hpp"

namespace aerial::physics
{
    Eigen::Vector3d CalculateAerodynamicDrag(
            const DroneState& state,
            const DroneParameters& parameters
            )
    {
        return -parameters.linear_drag.cwiseProduct(state.velocity);
    }
}
