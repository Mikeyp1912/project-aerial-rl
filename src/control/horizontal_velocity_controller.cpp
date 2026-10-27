// /src/control/horizontal_velocity_controller.cpp

#include "aerial/control/horizontal_velocity_controller.hpp"

#include <cmath>
#include <stdexcept>

#include "aerial/physics/integrator.hpp"

namespace aerial::control
{
    HorizontalVelocityControllerOutput UpdateHorizontalVelocityController(
            const physics::DroneState& drone_state,
            const Eigen::Vector2d& target_velocity,
            double target_yaw,
            const HorizontalVelocityControllerParameters& parameters
            )
    {
        if (!target_velocity.allFinite() ||
            !drone_state.velocity.allFinite() ||
            !std::isfinite(target_yaw) ||
            !std::isfinite(parameters.velocity_kp) ||
            !std::isfinite(parameters.max_horizontal_acceleration) ||
            !std::isfinite(parameters.max_tilt_angle) ||
            parameters.velocity_kp < 0.0 ||
            parameters.max_horizontal_acceleration < 0.0 ||
            parameters.max_tilt_angle < 0.0 ||
            parameters.max_tilt_angle >= 1.5707963267948966) {
            throw std::invalid_argument(
                "Invalid horizontal velocity controller input"
            );
        }

        HorizontalVelocityControllerOutput output;

        const Eigen::Vector2d current_velocity = drone_state.velocity.head<2>();

        const Eigen::Vector2d velocity_error = target_velocity - current_velocity;

        output.desired_acceleration = parameters.velocity_kp * velocity_error;

        const double magnitude = output.desired_acceleration.norm();

        if (magnitude > parameters.max_horizontal_acceleration && magnitude > 0.0) {
            output.desired_acceleration *= parameters.max_horizontal_acceleration / magnitude;

            output.saturated = true;
        }

        const double cos_yaw = std::cos(target_yaw);
        const double sin_yaw = std::sin(target_yaw);

        const double forward_acceleration = cos_yaw * output.desired_acceleration.x() +
            sin_yaw * output.desired_acceleration.y();

        const double left_acceleration = 
            -sin_yaw * output.desired_acceleration.x() + 
            cos_yaw * output.desired_acceleration.y();

        output.target_pitch = std::atan2(
                forward_acceleration,
                physics::gravity
                );

        output.target_roll = -std::atan2(
                left_acceleration,
                physics::gravity
                );

        const double tilt_magnitude = std::hypot(
                output.target_roll,
                output.target_pitch
                );

        if (tilt_magnitude > parameters.max_tilt_angle) {
            const double scale = parameters.max_tilt_angle / tilt_magnitude;

            output.target_roll *= scale;
            output.target_pitch *= scale;

            output.saturated = true;
        }

        return output;
    }
}
