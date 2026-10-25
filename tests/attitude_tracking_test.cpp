// /tests/attitude_tracking_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "aerial/control/flight_controller.hpp"
#include "aerial/physics/simulation_environment.hpp"

namespace
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double degrees_to_radians = pi / 180.0;

    double QuaternionError(
        const Eigen::Quaterniond& actual,
        const Eigen::Quaterniond& target)
    {
        const Eigen::Quaterniond error =
            target * actual.conjugate();

        return Eigen::AngleAxisd(error).angle();
    }

    void TestAttitudeTracking(
        double target_roll_degrees,
        double target_pitch_degrees)
    {
        aerial::physics::Simulation simulation;

        const auto& drone_parameters = simulation.GetParameters();

        aerial::control::FlightControllerParameters controller_parameters;
        aerial::control::FlightControllerState controller_state;

        const double hover_thrust =
            aerial::control::CalculateHoverThrust(drone_parameters);

        const double roll =
            target_roll_degrees * degrees_to_radians;

        const double pitch =
            target_pitch_degrees * degrees_to_radians;

        const Eigen::Quaterniond target_orientation =
            Eigen::Quaterniond(
                Eigen::AngleAxisd(
                    pitch,
                    Eigen::Vector3d::UnitY()
                )
            ) *
            Eigen::Quaterniond(
                Eigen::AngleAxisd(
                    roll,
                    Eigen::Vector3d::UnitX()
                )
            );

        constexpr double dt = 1.0 / 120.0;
        constexpr double duration = 5.0;

        const int steps = static_cast<int>(duration / dt);

        for (int i = 0; i < steps; ++i) {
            const auto output =
                aerial::control::UpdateFlightController(
                    simulation.GetDroneState(),
                    drone_parameters,
                    target_orientation,
                    hover_thrust,
                    dt,
                    controller_parameters,
                    controller_state
                );

            simulation.Step(output.motor_commands, dt);
        }

        const double final_error = QuaternionError(
            simulation.GetDroneState().orientation,
            target_orientation
        );

        std::cout
            << "Target roll: " << target_roll_degrees
            << " deg | Target pitch: " << target_pitch_degrees
            << " deg | Final error: "
            << final_error / degrees_to_radians
            << " deg\n";

        assert(std::isfinite(final_error));

        // The attitude controller should converge to within 2 degrees.
        assert(final_error < 2.0 * degrees_to_radians);
    }
}

int main()
{
    TestAttitudeTracking(10.0, 0.0);
    TestAttitudeTracking(-10.0, 0.0);

    TestAttitudeTracking(0.0, 10.0);
    TestAttitudeTracking(0.0, -10.0);

    TestAttitudeTracking(15.0, 10.0);

    std::cout << "Attitude tracking tests PASSED\n";

    return 0;
}
