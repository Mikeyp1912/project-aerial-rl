// /tests/horizontal_position_controller_test.cpp
// /tests/horizontal_position_controller_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/control/horizontal_position_controller.hpp"

namespace
{
    bool NearlyEqual(
        double actual,
        double expected,
        double tolerance = 1e-6
    )
    {
        return std::abs(actual - expected) < tolerance;
    }

    void TestZeroPositionError()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalPositionController(
                state,
                state.position.head<2>(),
                parameters
            );

        assert(output.position_error.norm() < 1e-6);
        assert(output.desired_velocity.norm() < 1e-6);
        assert(!output.saturated);
    }

    void TestPositiveXPositionError()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalPositionController(
                state,
                Eigen::Vector2d(2.0, 0.0),
                parameters
            );

        assert(NearlyEqual(output.desired_velocity.x(), 2.0));
        assert(NearlyEqual(output.desired_velocity.y(), 0.0));
        assert(!output.saturated);
    }

    void TestNegativeYPositionError()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalPositionController(
                state,
                Eigen::Vector2d(0.0, -1.5),
                parameters
            );

        assert(NearlyEqual(output.desired_velocity.x(), 0.0));
        assert(NearlyEqual(output.desired_velocity.y(), -1.5));
    }

    void TestCurrentPositionOffset()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        state.position.x() = 4.0;
        state.position.y() = -2.0;

        const auto output =
            aerial::control::UpdateHorizontalPositionController(
                state,
                Eigen::Vector2d(1.0, 0.0),
                parameters
            );

        assert(NearlyEqual(output.position_error.x(), -3.0));
        assert(NearlyEqual(output.position_error.y(), 2.0));

        // Error magnitude exceeds the default velocity limit.
        assert(output.saturated);

        assert(NearlyEqual(
            output.desired_velocity.norm(),
            parameters.max_horizontal_velocity
        ));
    }

    void TestVelocitySaturation()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalPositionController(
                state,
                Eigen::Vector2d(10.0, 10.0),
                parameters
            );

        assert(output.saturated);

        assert(NearlyEqual(
            output.desired_velocity.norm(),
            parameters.max_horizontal_velocity
        ));

        // Direction must be preserved.
        assert(NearlyEqual(
            output.desired_velocity.x(),
            output.desired_velocity.y()
        ));
    }

    void TestProportionalGain()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        parameters.position_kp = 0.5;

        const auto output =
            aerial::control::UpdateHorizontalPositionController(
                state,
                Eigen::Vector2d(2.0, 0.0),
                parameters
            );

        assert(NearlyEqual(output.desired_velocity.x(), 1.0));
    }

    void TestInvalidParameters()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        parameters.position_kp = -1.0;

        bool threw = false;

        try {
            aerial::control::UpdateHorizontalPositionController(
                state,
                Eigen::Vector2d::Zero(),
                parameters
            );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    void TestNonFiniteTarget()
    {
        aerial::physics::DroneState state;

        aerial::control::HorizontalPositionControllerParameters parameters;

        bool threw = false;

        try {
            aerial::control::UpdateHorizontalPositionController(
                state,
                Eigen::Vector2d(
                    std::numeric_limits<double>::quiet_NaN(),
                    0.0
                ),
                parameters
            );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }
}

int main()
{
    TestZeroPositionError();
    TestPositiveXPositionError();
    TestNegativeYPositionError();
    TestCurrentPositionOffset();
    TestVelocitySaturation();
    TestProportionalGain();
    TestInvalidParameters();
    TestNonFiniteTarget();

    std::cout << "Horizontal position controller tests PASSED\n";

    return 0;
}
