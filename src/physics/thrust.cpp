// /src/physics/thrust.cpp

#include "aerial/physics/thrust.hpp" 

namespace aerial::physics
{
    Eigen::Vector3d CalculateThrustForce(
            const DroneState& state,
            double thrust
            )
    {
        // Thrust acts along the drone's body +Z axis
        const Eigen::Vector3d body_thrust(0.0, 0.0, thrust);

        // Rotate body-frame thrust into world coordinates
        return state.orientation * body_thrust;
    }
}
