// /src/control/attitude_controller.cpp

#include "aerial/control/attitude_controller.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::control
{
    Eigen::Vector3d CalculateDesiredAngularRate(
            const Eigen::Quaterniond& current_orientation,
            const Eigen::Quaterniond& target_orientation,
            const AttitudeControllerParameters &parameters
            )
    {
        if (!current_orientation.coeffs().allFinite() ||
            !target_orientation.coeffs().allFinite() ||
            !std::isfinite(parameters.kp) ||
            !std::isfinite(parameters.max_angular_rate) ||
            parameters.kp < 0.0 ||
            parameters.max_angular_rate < 0.0 ||
            current_orientation.norm() < 1e-12 ||
            target_orientation.norm() < 1e-12) {
            throw std::invalid_argument(
                    "Invalid attitude controller input"
                    );
        }

        const Eigen::Quaterniond current = current_orientation.normalized();

        const Eigen::Quaterniond target = target_orientation.normalized();

        Eigen::Quaterniond error = current.conjugate() * target;

        // Choose the shortest rotation
        if (error.w() < 0.0) {
            error.coeffs() *= -1.0;
        }

        // Convert quaternion error into a rotation vector
        const Eigen::Vector3d vector = error.vec();
        const double sin_half_angle = vector.norm();

        Eigen::Vector3d rotation_error = Eigen::Vector3d::Zero();

        if (sin_half_angle > 1e-12) {
            const double angle = 2.0 * std::atan2(
                    sin_half_angle,
                    std::clamp(error.w(), 0.0, 1.0)
                    );

            rotation_error = (angle / sin_half_angle) * vector;
        }
        else {
            rotation_error = 2.0 * vector;
        }

        Eigen::Vector3d desired_rate = parameters.kp * rotation_error;

        // Limit the magnitude of the angular-rate command
        const double magnitude = desired_rate.norm();

        if (magnitude > parameters.max_angular_rate && magnitude > 0.0) {
            desired_rate *= parameters.max_angular_rate / magnitude;
        }

        return desired_rate;
    }
}

