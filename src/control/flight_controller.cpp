// /src/control/flight_controller.cpp

#include "aerial/control/flight_controller.hpp"

#include <cmath>
#include <stdexcept>

namespace aerial::control
{
    double CalculateHoverThrust(const physics::DroneParameters& parameters)
    {
        constexpr double gravity = 9.81;

        if (!std::isfinite(parameters.mass) ||
            parameters.mass <= 0.0 ||
            !std::isfinite(parameters.max_motor_thrust) ||
            parameters.max_motor_thrust <= 0.0) {
            throw std::invalid_argument("Invalid drone parameters for hover thrust");
        }

        return parameters.mass * gravity;
    }

    FlightControllerOutput UpdateFlightController(
            const physics::DroneState &drone_state,
            const physics::DroneParameters &drone_parameters,
            const Eigen::Quaterniond &target_orientation,
            double collective_thrust,
            double dt,
            const FlightControllerParameters &controller_parameters,
            FlightControllerState &controller_state
            )
    {
        if (!drone_state.angular_velocity.allFinite() ||
            !std::isfinite(collective_thrust) ||
            !std::isfinite(dt) ||
            dt <= 0.0) {
            throw std::invalid_argument("Invalid flight controller input");
        }

        FlightControllerOutput output;

        // 1. Orientation error -> desired body angular velocity
        output.desired_angular_rate =
            CalculateDesiredAngularRate(
                    drone_state.orientation,
                    target_orientation,
                    controller_parameters.attitude
                    );

        // 2. Angular velocity error -> requested body torque
        output.requested_body_torque = 
            CalculateBodyTorque(
                    output.desired_angular_rate,
                    drone_state.angular_velocity,
                    dt,
                    controller_parameters.rate,
                    controller_state.rate
                    );

        // 3. Requested thrust and torque -> motor commands
        const physics::MotorMixerOutput mixed = 
            physics::MixMotors(
                    drone_parameters,
                    collective_thrust,
                    output.requested_body_torque
                    );

        output.motor_commands = mixed.commands;
        output.motor_saturated = mixed.saturated;

        return output;
    }

    void ResetFlightController(FlightControllerState& controller_state)
    {
        ResetRateController(controller_state.rate);
    }
}
