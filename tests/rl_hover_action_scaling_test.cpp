

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/rl/action_adapter.hpp"

namespace
{
    bool Near(double a, double b)
    {
        return std::abs(a - b) < 1e-12;
    }
}

int main()
{
    using namespace aerial::rl;

    PIDTargetActionParameters parameters;

    // Legacy altitude mapping remains unchanged.
    assert(Near(
        CalculatePIDAltitudeTarget(parameters, 0.0),
        2.75
    ));

    assert(Near(
        CalculatePIDAltitudeTarget(parameters, -1.0),
        0.5
    ));

    assert(Near(
        CalculatePIDAltitudeTarget(parameters, 1.0),
        5.0
    ));

    // Enable hover-centred altitude control.
    parameters.hover_centered_altitude = true;
    parameters.hover_altitude = 2.0;
    parameters.maximum_altitude_offset = 1.0;

    assert(Near(
        CalculatePIDAltitudeTarget(parameters, 0.0),
        2.0
    ));

    assert(Near(
        CalculatePIDAltitudeTarget(parameters, -1.0),
        1.0
    ));

    assert(Near(
        CalculatePIDAltitudeTarget(parameters, 1.0),
        3.0
    ));

    // Clamping still respects the absolute altitude range.
    parameters.hover_altitude = 4.8;

    assert(Near(
        CalculatePIDAltitudeTarget(parameters, 1.0),
        5.0
    ));

    // Invalid action values must be rejected.
    bool threw = false;

    try {
        CalculatePIDAltitudeTarget(
            parameters,
            std::numeric_limits<double>::quiet_NaN()
        );
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);

    // Hover-specific horizontal scaling.
    parameters.reference = PIDTargetReference::WorldRelative;
    parameters.maximum_horizontal_offset = 1.0;
    parameters.world_target_position = Eigen::Vector2d::Zero();

    const Eigen::Vector2d initial_position(0.4, -0.3);

    const Action zero{0.0, 0.0, 0.0, 0.0};

    const auto zero_target = CalculatePIDHorizontalTarget(
        parameters,
        initial_position,
        zero
    );

    assert(zero_target.norm() < 1e-12);

    const Action offset{0.5, -0.25, 0.0, 0.0};

    const auto offset_target = CalculatePIDHorizontalTarget(
        parameters,
        initial_position,
        offset
    );

    assert(
        (offset_target - Eigen::Vector2d(0.5, -0.25)).norm()
        < 1e-12
    );

    std::cout << "Hover action scaling tests PASSED\n";

    return 0;
}
