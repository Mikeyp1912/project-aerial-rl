// /src/control/rate_controller.cpp

#include "aerial/control/rate_controller.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::control
{
    namespace
    {
        Eigen::Vector3d ClampVector(
                const Eigen::Vector3d& value,
                const Eigen::Vector3d& limits
                )
        {
            Eigen::Vector3d result;

            for (int i = 0; i < 3; ++i) {
                result[i] = std::clamp(value[i], -limits[i], limits[i]);
            }

            return result;
        }

        bool ValidGains(const Eigen::Vector3d& gains)
        {
            return gains.allFinite() && (gains.array() >= 0.0).all();
        }
    }

    Eigen::Vector3d CalculateBodyTorque(
            const Eigen::Vector3d& desired_angular_velocity,
            const Eigen::Vector3d& actual_angular_velocity,
            double dt,
            const RateControllerParameters& parameters,
            RateControllerState& state
            )
    {
        if (!desired_angular_velocity.allFinite() ||
            !actual_angular_velocity.allFinite() ||
            !std::isfinite(dt) ||
            dt <= 0.0 ||
            !ValidGains(parameters.kp) ||
            !ValidGains(parameters.ki) ||
            !ValidGains(parameters.kd) ||
            !ValidGains(parameters.integral_limit) ||
            !ValidGains(parameters.torque_limit) ||
            !state.integral_error.allFinite() ||
            (state.initialized &&
             !state.previous_angular_velocity.allFinite())) {
            throw std::invalid_argument(
                    "Invalid rate controller input"
                    );
        }

        const Eigen::Vector3d error = desired_angular_velocity - actual_angular_velocity;

        // Estimate angular acceleration from measured rates
        Eigen::Vector3d angular_acceleration = Eigen::Vector3d::Zero();

        if (state.initialized) {
            angular_acceleration = (actual_angular_velocity - state.previous_angular_velocity) / dt;
        }

        // Proportional and derivative contributions
        const Eigen::Vector3d pd_torque = 
            parameters.kp.cwiseProduct(error) - 
            parameters.kd.cwiseProduct(angular_acceleration);

        // Candidate integral state, bounded independently per axis
        const Eigen::Vector3d candidate_integral = ClampVector(
                state.integral_error + error * dt,
                parameters.integral_limit
                );

        Eigen::Vector3d next_integral = state.integral_error;

        // Conditional integration anti-windup
        for (int i = 0; i < 3; ++i) {
            const double candidate_torque =
                pd_torque[i] + 
                parameters.ki[i] * candidate_integral[i];

            const bool pushing_upper =
                candidate_torque > parameters.torque_limit[i] &&
                error[i] > 0.0;

            const bool pushing_lower =
                candidate_torque < -parameters.torque_limit[i] &&
                error[i] < 0.0;

            if (!pushing_upper && !pushing_lower) {
                next_integral[i] = candidate_integral[i];
            }
        }

        const Eigen::Vector3d raw_torque = 
            pd_torque +
            parameters.ki.cwiseProduct(next_integral);

        const Eigen::Vector3d torque = ClampVector(
                raw_torque,
                parameters.torque_limit
                );

        state.integral_error = next_integral;
        state.previous_angular_velocity = actual_angular_velocity;
        state.initialized = true;

        return torque;
    }

    void ResetRateController(RateControllerState& state)
    {
        state = RateControllerState{};
    }
}
