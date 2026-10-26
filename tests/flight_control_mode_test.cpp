// /tests/flight_control_mode_test.cpp

#include <cassert>
#include <iostream>

#include "aerial/control/flight_control_mode.hpp"

using aerial::control::FlightControlMode;
using aerial::control::FlightControlModeState;

int main()
{
    FlightControlModeState state;

    // Default mode must be open-loop.
    assert(state.mode == FlightControlMode::OpenLoop);

    // Toggle into attitude hold.
    aerial::control::ToggleFlightControlMode(state);

    assert(state.mode == FlightControlMode::AttitudeHold);

    // Reset must preserve the selected mode.
    aerial::control::ResetFlightControlMode(state);

    assert(state.mode == FlightControlMode::AttitudeHold);

    // Toggle back into AltitudeHold.
    aerial::control::ToggleFlightControlMode(state);

    assert(state.mode == FlightControlMode::AltitudeHold);

    // Reset must preserve the selected mode.
    aerial::control::ResetFlightControlMode(state);

    assert(state.mode == FlightControlMode::AltitudeHold);

    // Toggle back into open-loop.
    aerial::control::ToggleFlightControlMode(state);

    assert(state.mode == FlightControlMode::OpenLoop);

    // Reset must preserve open-loop mode.
    aerial::control::ResetFlightControlMode(state);

    assert(state.mode == FlightControlMode::OpenLoop);

    std::cout << "Flight control mode tests PASSED\n";

    return 0;
}
