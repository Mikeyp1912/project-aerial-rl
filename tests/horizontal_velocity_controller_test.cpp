// /tests/horizontal_velocity_controller_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/control/horizontal_velocity_controller.hpp"

namespace
{
    constexpr double pi = 3.14159265358979323846;

    bool NearlyEqual(double a, double b, double tolerance = 1e-6)
    {
        return std::abs(a - b) < tolerance;
    }

    void TestZeroVelocityError()
    {
        aerial::physics::DroneState state;
        aerial::control::HorizontalVelocityControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalVelocityController(
                state,
                Eigen::Vector2d::Zero(),
                0.0,
                parameters
            );

        assert(NearlyEqual(output.target_roll, 0.0));
        assert(NearlyEqual(output.target_pitch, 0.0));
        assert(!output.saturated);
    }

    void TestPositiveXVelocity()
    {
        aerial::physics::DroneState state;
        aerial::control::HorizontalVelocityControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalVelocityController(
                state,
                Eigen::Vector2d(1.0, 0.0),
                0.0,
                parameters
            );

        assert(output.target_pitch > 0.0);
        assert(NearlyEqual(output.target_roll, 0.0));
    }

    void TestPositiveYVelocity()
    {
        aerial::physics::DroneState state;
        aerial::control::HorizontalVelocityControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalVelocityController(
                state,
                Eigen::Vector2d(0.0, 1.0),
                0.0,
                parameters
            );

        assert(output.target_roll < 0.0);
        assert(NearlyEqual(output.target_pitch, 0.0));
    }

    void TestYawRotation()
    {
        aerial::physics::DroneState state;
        aerial::control::HorizontalVelocityControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalVelocityController(
                state,
                Eigen::Vector2d(1.0, 0.0),
                pi / 2.0,
                parameters
            );

        assert(output.target_roll > 0.0);
        assert(NearlyEqual(output.target_pitch, 0.0));
    }

    void TestVelocityCorrection()
    {
        aerial::physics::DroneState state;
        aerial::control::HorizontalVelocityControllerParameters parameters;

        state.velocity.x() = 2.0;

        const auto output =
            aerial::control::UpdateHorizontalVelocityController(
                state,
                Eigen::Vector2d::Zero(),
                0.0,
                parameters
            );

        assert(output.target_pitch < 0.0);
    }

    void TestAccelerationSaturation()
    {
        aerial::physics::DroneState state;
        aerial::control::HorizontalVelocityControllerParameters parameters;

        const auto output =
            aerial::control::UpdateHorizontalVelocityController(
                state,
                Eigen::Vector2d(100.0, 100.0),
                0.0,
                parameters
            );

        assert(output.saturated);

        assert(
            output.desired_acceleration.norm() <=
            parameters.max_horizontal_acceleration + 1e-6
        );

        assert(
            std::hypot(output.target_roll, output.target_pitch) <=
            parameters.max_tilt_angle + 1e-6
        );
    }

    void TestInvalidParameters()
    {
        aerial::physics::DroneState state;
        aerial::control::HorizontalVelocityControllerParameters parameters;

        parameters.velocity_kp = -1.0;

        bool threw = false;

        try {
            aerial::control::UpdateHorizontalVelocityController(
                state,
                Eigen::Vector2d::Zero(),
                0.0,
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
    TestZeroVelocityError();
    TestPositiveXVelocity();
    TestPositiveYVelocity();
    TestYawRotation();
    TestVelocityCorrection();
    TestAccelerationSaturation();
    TestInvalidParameters();

    std::cout << "Horizontal velocity controller tests PASSED\n";

    return 0;
}
