// /src/control/flight_control_mode.cpp

#include "aerial/control/flight_control_mode.hpp"

namespace aerial::control
{
    void ToggleFlightControlMode(FlightControlModeState& state)
    {
        if (state.mode == FlightControlMode::OpenLoop) {
            state.mode = FlightControlMode::AttitudeHold;
        }
        else {
            state.mode = FlightControlMode::OpenLoop;
        }

        ResetFlightController(state.controller_state);
    }

    void ResetFlightControlMode(FlightControlModeState& state)
    {
        // Preserve the selected control mode.
        ResetFlightController(state.controller_state);
    }
}
