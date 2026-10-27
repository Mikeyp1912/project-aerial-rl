// /tests/flight_control_targets_test.cpp
#include <cassert>
#include <cmath>
#include <iostream>

#include "aerial/control/flight_control_targets.hpp"

using namespace aerial;

constexpr double tolerance = 1e-12;

bool NearlyEqual(double a, double b)
{
    return std::abs(a - b) < tolerance;
}

int main()
{
    physics::DroneState drone;

    drone.position = Eigen::Vector3d(3.0, -2.0, 5.0);

    control::FlightControlTargets targets;

    targets.position = Eigen::Vector2d(10.0, 20.0);
    targets.altitude = 8.0;
    targets.roll = 0.2;
    targets.pitch = -0.3;
    targets.yaw = 1.0;

    // Open Loop should not modify targets.
    control::CaptureFlightModeTargets(
        control::FlightControlMode::OpenLoop,
        drone,
        targets
    );

    assert(NearlyEqual(targets.altitude, 8.0));
    assert(NearlyEqual(targets.position.x(), 10.0));
    assert(NearlyEqual(targets.roll, 0.2));

    // Attitude Hold should preserve targets.
    control::CaptureFlightModeTargets(
        control::FlightControlMode::AttitudeHold,
        drone,
        targets
    );

    assert(NearlyEqual(targets.altitude, 8.0));
    assert(NearlyEqual(targets.pitch, -0.3));

    // Altitude Hold captures Z only.
    control::CaptureFlightModeTargets(
        control::FlightControlMode::AltitudeHold,
        drone,
        targets
    );

    assert(NearlyEqual(targets.altitude, 5.0));
    assert(NearlyEqual(targets.position.x(), 10.0));
    assert(NearlyEqual(targets.position.y(), 20.0));

    // Position Hold captures XYZ and levels the target.
    control::CaptureFlightModeTargets(
        control::FlightControlMode::PositionHold,
        drone,
        targets
    );

    assert(NearlyEqual(targets.position.x(), 3.0));
    assert(NearlyEqual(targets.position.y(), -2.0));
    assert(NearlyEqual(targets.altitude, 5.0));

    assert(NearlyEqual(targets.roll, 0.0));
    assert(NearlyEqual(targets.pitch, 0.0));

    // Yaw should remain unchanged.
    assert(NearlyEqual(targets.yaw, 1.0));

    std::cout << "Flight control target tests PASSED\n";

    return 0;
}
