// /src/physics/motor_mixer.cpp

#include "aerial/physics/motor_mixer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace aerial::physics
{
    MotorMixerOutput MixMotors(
            const DroneParameters& parameters,
            double collective_thrust,
            const Eigen::Vector3d& body_torque
            )
    {
        const double max_thrust = parameters.max_motor_thrust;
        const double arm = parameters.arm_length / std::sqrt(2.0);
        const double yaw = parameters.yaw_torque_coefficient;

        if (!std::isfinite(max_thrust) || max_thrust <= 0.0 ||
            !std::isfinite(arm) || arm <= 0.0 ||
            !std::isfinite(yaw) || yaw <= 0.0 ||
            !std::isfinite(collective_thrust) ||
            !body_torque.allFinite()) {
            throw std::invalid_argument("Invalid motor mixer input");
        }

        const double roll = body_torque.x() / arm;
        const double pitch = body_torque.y() / arm;
        const double yaw_term = body_torque.z() / yaw;

        const std::array<double, 4> thrusts = {
            (collective_thrust + roll - pitch - yaw_term) / 4.0,
            (collective_thrust + roll + pitch + yaw_term) / 4.0,
            (collective_thrust - roll + pitch - yaw_term) / 4.0,
            (collective_thrust - roll - pitch + yaw_term) / 4.0,
        };

        MotorMixerOutput output;

        for (std::size_t i = 0; i < thrusts.size(); ++i) {
            const double limited = std::clamp(thrusts[i], 0.0, max_thrust);

            if (limited != thrusts[i]) {
                output.saturated = true;
            }

            output.commands[i] = limited / max_thrust;
        }

        return output;
    }
}
