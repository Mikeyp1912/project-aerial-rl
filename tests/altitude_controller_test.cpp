// /tests/altitude_controller_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Geometry>

#include "aerial/control/altitude_controller.hpp"
#include "aerial/physics/integrator.hpp"

namespace
{
    bool NearlyEqual(double a, double b, double tolerance = 1e-6)
    {
        return std::abs(a - b) < tolerance;
    }

    void TestHoverThrust()
    {
        aerial::physics::DroneState state;
        aerial::physics::DroneParameters drone_parameters;
        aerial::control::AltitudeControllerParameters parameters;

        const auto output =
            aerial::control::UpdateAltitudeController(
                state,
                drone_parameters,
                0.0,
                parameters
            );

        assert(NearlyEqual(
            output.collective_thrust,
            drone_parameters.mass * aerial::physics::gravity
        ));

        assert(!output.saturated);
    }

    void TestAltitudeCorrection()
    {
        aerial::physics::DroneState state;
        aerial::physics::DroneParameters drone_parameters;
        aerial::control::AltitudeControllerParameters parameters;

        state.position.z() = -1.0;

        const auto below =
            aerial::control::UpdateAltitudeController(
                state, drone_parameters, 0.0, parameters
            );

        assert(below.collective_thrust >
               drone_parameters.mass * aerial::physics::gravity);

        state.position.z() = 1.0;

        const auto above =
            aerial::control::UpdateAltitudeController(
                state, drone_parameters, 0.0, parameters
            );

        assert(above.collective_thrust <
               drone_parameters.mass * aerial::physics::gravity);
    }

    void TestVerticalVelocityCorrection()
    {
        aerial::physics::DroneState state;
        aerial::physics::DroneParameters drone_parameters;
        aerial::control::AltitudeControllerParameters parameters;

        state.velocity.z() = -2.0;

        const auto falling =
            aerial::control::UpdateAltitudeController(
                state, drone_parameters, 0.0, parameters
            );

        assert(falling.collective_thrust >
               drone_parameters.mass * aerial::physics::gravity);

        state.velocity.z() = 2.0;

        const auto rising =
            aerial::control::UpdateAltitudeController(
                state, drone_parameters, 0.0, parameters
            );

        assert(rising.collective_thrust <
               drone_parameters.mass * aerial::physics::gravity);
    }

    void TestTiltCompensation()
    {
        aerial::physics::DroneState state;
        aerial::physics::DroneParameters drone_parameters;
        aerial::control::AltitudeControllerParameters parameters;

        constexpr double pi = 3.14159265358979323846;

        state.orientation = Eigen::Quaterniond(
            Eigen::AngleAxisd(
                20.0 * pi / 180.0,
                Eigen::Vector3d::UnitX()
            )
        );

        const auto output =
            aerial::control::UpdateAltitudeController(
                state, drone_parameters, 0.0, parameters
            );

        const double expected_thrust =
            drone_parameters.mass * aerial::physics::gravity /
            std::cos(20.0 * pi / 180.0);

        assert(NearlyEqual(output.collective_thrust, expected_thrust));
        assert(!output.saturated);
    }

    void TestThrustSaturation()
    {
        aerial::physics::DroneState state;
        aerial::physics::DroneParameters drone_parameters;
        aerial::control::AltitudeControllerParameters parameters;

        state.position.z() = -100.0;

        drone_parameters.max_motor_thrust = 3.0;

        const auto output =
            aerial::control::UpdateAltitudeController(
                state, drone_parameters, 0.0, parameters
            );

        assert(output.saturated);
        assert(NearlyEqual(output.collective_thrust, 12.0));
    }
}

int main()
{
    TestHoverThrust();
    TestAltitudeCorrection();
    TestVerticalVelocityCorrection();
    TestTiltCompensation();
    TestThrustSaturation();

    std::cout << "Altitude controller tests PASSED\n";

    return 0;
}
