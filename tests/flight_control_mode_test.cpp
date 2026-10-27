// /tests/flight_control_mode_test.cpp

#include <cassert>
#include <iostream>

#include "aerial/control/flight_control_mode.hpp"

using aerial::control::FlightControlMode;
using aerial::control::FlightControlModeState;

int main()
{
    FlightControlModeState state;

    assert(state.mode == FlightControlMode::OpenLoop);

    aerial::control::ToggleFlightControlMode(state);
    assert(state.mode == FlightControlMode::AttitudeHold);

    aerial::control::ResetFlightControlMode(state);
    assert(state.mode == FlightControlMode::AttitudeHold);

    aerial::control::ToggleFlightControlMode(state);
    assert(state.mode == FlightControlMode::AltitudeHold);

    aerial::control::ResetFlightControlMode(state);
    assert(state.mode == FlightControlMode::AltitudeHold);

    aerial::control::ToggleFlightControlMode(state);
    assert(state.mode == FlightControlMode::PositionHold);

    aerial::control::ResetFlightControlMode(state);
    assert(state.mode == FlightControlMode::PositionHold);

    aerial::control::ToggleFlightControlMode(state);
    assert(state.mode == FlightControlMode::OpenLoop);

    aerial::control::ResetFlightControlMode(state);
    assert(state.mode == FlightControlMode::OpenLoop);

    std::cout << "Flight control mode tests PASSED\n";

    return 0;
}
