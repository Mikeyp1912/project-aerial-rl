// /include/aerial/control/altitude_controller.hpp

#pragma once

#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/drone_state.hpp"

namespace aerial::control
{
    struct AltitudeControllerParameters
    {
        double position_kp = 1.5;
        double velocity_kp = 3.0;

        double max_vertical_velocity = 3.0;
        double max_vertical_acceleration = 5.0;

        double minimum_vertical_thrust_fraction = 0.3;
    };

    struct AltitudeControllerOutput
    {
        double desired_vertical_velocity = 0.0;
        double desired_vertical_acceleration = 0.0;

        double collective_thrust = 0.0;

        bool saturated = false;
    };

    AltitudeControllerOutput UpdateAltitudeController(
            const physics::DroneState& drone_state,
            const physics::DroneParameters& drone_parameters,
            double target_altitude,
            const AltitudeControllerParameters& controller_parameters
            );
}
