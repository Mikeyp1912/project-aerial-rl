// /include/aerial/control/flight_control_targets.hpp

#pragma once

#include <Eigen/Core>

#include "aerial/control/flight_control_mode.hpp"
#include "aerial/physics/drone_state.hpp"

namespace aerial::control
{
    struct FlightControlTargets
    {
        Eigen::Vector2d position = Eigen::Vector2d::Zero();

        double altitude = 0.0;
        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
    };

    void CaptureFlightModeTargets(
            FlightControlMode mode,
            const physics::DroneState& drone_state,
            FlightControlTargets& targets
            );
}
