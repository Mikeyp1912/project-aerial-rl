// src/physics/integrator.cpp
#include "aerial/physics/integrator.hpp"
#include "aerial/physics/drone_parameters.hpp"

namespace aerial::physics
{
    void Integrate(
            DroneState& state,
            const DroneParameters& parameters,
            const Eigen::Vector3d& force,
            double dt
            )
    {
        if (dt <= 0.0 || parameters.mass <= 0.0) {
            return;
        }

        // Force is expressed in world coordinates (newtons)
        const Eigen::Vector3d gravity_acceleration(0.0, 0.0, -gravity);

        const Eigen::Vector3d acceleration = force / parameters.mass + gravity_acceleration;

        // Semi-implicit Euler integration
        state.velocity += acceleration * dt;
        state.position += state.velocity * dt;
       
    }
}
