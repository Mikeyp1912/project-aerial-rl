// /include/aerial/control/flight_control_mode.hpp

#pragma once

#include "aerial/control/flight_controller.hpp"

namespace aerial::control
{
    enum class FlightControlMode
    {
        OpenLoop,
        AttitudeHold,
        AltitudeHold
    };

    struct FlightControlModeState
    {
        FlightControlMode mode = FlightControlMode::OpenLoop;
        FlightControllerState controller_state{};
    };

    void ToggleFlightControlMode(FlightControlModeState& state);

    void ResetFlightControlMode(FlightControlModeState& state);
}
