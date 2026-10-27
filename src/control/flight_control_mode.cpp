// /src/control/flight_control_mode.cpp

#include "aerial/control/flight_control_mode.hpp"

namespace aerial::control
{
    void ToggleFlightControlMode(FlightControlModeState& state)
    {
        switch (state.mode) {
            case FlightControlMode::OpenLoop:
                state.mode = FlightControlMode::AttitudeHold;
                break;

            case FlightControlMode::AttitudeHold:
                state.mode = FlightControlMode::AltitudeHold;
                break;

            case FlightControlMode::AltitudeHold:
                state.mode = FlightControlMode::PositionHold;
                break;
            case FlightControlMode::PositionHold:
                state.mode = FlightControlMode::OpenLoop;
                break;
        }

        ResetFlightController(state.controller_state);
    }

    void ResetFlightControlMode(FlightControlModeState& state)
    {
        // Preserve the selected control mode.
        ResetFlightController(state.controller_state);
    }
}
