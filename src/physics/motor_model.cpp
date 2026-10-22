// /src/physics/motor_model.cpp

#include "aerial/physics/motor_model.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Geometry>

namespace aerial::physics
{
    MotorOutput CalculateMotorOutput(
            const DroneParameters &parameters,
            const std::array<double, 4> &motor_thrusts
            )
    {
        MotorOutput output;

        // X-configuration motor positions
        const double offset = parameters.arm_length / std::sqrt(2.0);

        const std::array<Eigen::Vector3d, 4> positions = {
            Eigen::Vector3d( offset,  offset, 0.0), // Front-left
            Eigen::Vector3d(-offset,  offset, 0.0), // Rear-left
            Eigen::Vector3d(-offset, -offset, 0.0), // Rear-right
            Eigen::Vector3d( offset, -offset, 0.0), // Front-right
        };

        // +1 = CCW, -1 CW, viewed from above
        const std::array<double, 4> spin_directions = {
            1.0, -1.0, 1.0, -1.0
        };

        for (std::size_t i = 0; i < motor_thrusts.size(); ++i) {
            const double thrust = std::clamp(
                    motor_thrusts[i],
                    0.0,
                    parameters.max_motor_thrust
                    ); 

            const Eigen::Vector3d motor_force(0.0, 0.0, thrust);

            // Sum forces
            output.force += motor_force;

            // Torque from force applied at motor position
            output.torque += positions[i].cross(motor_force);

            // Reaction torque acts opposite to propeller rotation
            output.torque.z() -= spin_directions[i] * parameters.yaw_torque_coefficient * thrust;
        }

        return output;
    };
}
