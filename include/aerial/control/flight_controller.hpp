// /include/aerial/control/flight_controller.hpp

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "aerial/control/attitude_controller.hpp"
#include "aerial/control/rate_controller.hpp"
#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/drone_state.hpp"
#include "aerial/physics/motor_mixer.hpp"

namespace aerial::control
{
    struct FlightControllerParameters
    {
        AttitudeControllerParameters attitude;
        RateControllerParameters rate;
    };

    struct FlightControllerState
    {
        RateControllerState rate;
    };

    struct FlightControllerOutput
    {
        Eigen::Vector3d desired_angular_rate = Eigen::Vector3d::Zero();

        Eigen::Vector3d requested_body_torque = Eigen::Vector3d::Zero();

        physics::MotorCommands motor_commands{};

        bool motor_saturated = false;
    };

    double CalculateHoverThrust(const physics::DroneParameters & parameters);

    FlightControllerOutput UpdateFlightController(
            const physics::DroneState& drone_state,
            const physics::DroneParameters& drone_parameters,
            const Eigen::Quaterniond& target_orientation,
            double collective_thrust,
            double dt,
            const FlightControllerParameters& controller_parameters,
            FlightControllerState& controller_state
            );

    void ResetFlightController(FlightControllerState& controller_state);
}
