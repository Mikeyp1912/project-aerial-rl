// /src/control/flight_control_targets.cpp

#include "aerial/control/flight_control_targets.hpp"

namespace aerial::control
{
    void CaptureFlightModeTargets(
            FlightControlMode mode,
            const physics::DroneState& drone_state,
            FlightControlTargets& targets
            )
    {
        switch (mode) {
            case FlightControlMode::OpenLoop:
            case FlightControlMode::AttitudeHold:
                break;

            case FlightControlMode::AltitudeHold:
                targets.altitude = drone_state.position.z();
                break;

            case FlightControlMode::PositionHold:
                targets.position = drone_state.position.head<2>();
                targets.altitude = drone_state.position.z();

                targets.roll = 0.0;
                targets.pitch = 0.0;
                break;
        }
    }
}
