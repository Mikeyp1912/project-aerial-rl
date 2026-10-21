// /src/physics/rotational_dynamics.cpp

#include "aerial/physics/rotational_dynamics.hpp"

#include <Eigen/Geometry>

namespace aerial::physics
{
    void IntegrateRotation(
            DroneState &state,
            const DroneParameters &parameters,
            const Eigen::Vector3d &torque,
            double dt
            )
    {
        if (dt <= 0.0 || (parameters.inertia.array() <= 0.0).any()) {
            return;
        }

        // Angular momentum in body coordinates
        const Eigen::Vector3d angular_momentum = parameters.inertia.cwiseProduct(state.angular_velocity);

        // Gyroscopic coupling term
        const Eigen::Vector3d gyroscopic = state.angular_velocity.cross(angular_momentum);

        // Euler's rigid-body equation, assuming diagonal inertia
        const Eigen::Vector3d angular_acceleration = (torque - gyroscopic).cwiseQuotient(parameters.inertia);

        // Semi-implicit Euler integration
        state.angular_velocity += angular_acceleration * dt;

        // Incremental rotation in body coordinates
        const Eigen::Vector3d rotation_vector = state.angular_velocity * dt;

        const double angle = rotation_vector.norm();

        if (angle > 0.0) {
            const Eigen::Quaterniond delta_rotation(
                    Eigen::AngleAxisd(
                        angle, rotation_vector / angle
                        )
                    );

            // Orientation maps body coordinates to world coordinates
            // Right multiplication applies to a body-frame rotation
            state.orientation = (state.orientation * delta_rotation).normalized();
        }


    }
}
