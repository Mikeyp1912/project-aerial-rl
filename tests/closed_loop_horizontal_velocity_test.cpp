// /tests/closed_loop_horizontal_velocity_test.cpp
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "aerial/control/altitude_controller.hpp"
#include "aerial/control/flight_controller.hpp"
#include "aerial/control/horizontal_velocity_controller.hpp"
#include "aerial/physics/simulation_environment.hpp"

namespace
{
    constexpr double dt = 1.0 / 120.0;
    constexpr double duration = 10.0;

    struct TestResult
    {
        Eigen::Vector2d velocity = Eigen::Vector2d::Zero();
        Eigen::Vector2d position = Eigen::Vector2d::Zero();

        double altitude = 0.0;
        double vertical_velocity = 0.0;
        double maximum_speed = 0.0;
    };

    TestResult RunScenario(
        const Eigen::Vector2d& initial_velocity,
        const Eigen::Vector2d& target_velocity,
        double target_yaw = 0.0
    )
    {
        aerial::physics::Simulation simulation;

        auto initial_state = simulation.GetDroneState();

        initial_state.velocity.x() = initial_velocity.x();
        initial_state.velocity.y() = initial_velocity.y();

        simulation.Reset(initial_state);

        const auto& drone_parameters = simulation.GetParameters();

        aerial::control::HorizontalVelocityControllerParameters
            velocity_parameters;

        aerial::control::AltitudeControllerParameters
            altitude_parameters;

        aerial::control::FlightControllerParameters
            flight_parameters;

        aerial::control::FlightControllerState
            flight_state;

        const double target_altitude = initial_state.position.z();

        TestResult result;

        const int steps = static_cast<int>(duration / dt);

        for (int i = 0; i < steps; ++i) {
            const auto& state = simulation.GetDroneState();

            // Horizontal velocity -> target roll and pitch.
            const auto velocity_output =
                aerial::control::UpdateHorizontalVelocityController(
                    state,
                    target_velocity,
                    target_yaw,
                    velocity_parameters
                );

            // Yaw * Pitch * Roll.
            const Eigen::Quaterniond target_orientation =
                Eigen::Quaterniond(
                    Eigen::AngleAxisd(
                        target_yaw,
                        Eigen::Vector3d::UnitZ()
                    )
                ) *
                Eigen::Quaterniond(
                    Eigen::AngleAxisd(
                        velocity_output.target_pitch,
                        Eigen::Vector3d::UnitY()
                    )
                ) *
                Eigen::Quaterniond(
                    Eigen::AngleAxisd(
                        velocity_output.target_roll,
                        Eigen::Vector3d::UnitX()
                    )
                );

            // Maintain altitude while moving horizontally.
            const auto altitude_output =
                aerial::control::UpdateAltitudeController(
                    state,
                    drone_parameters,
                    target_altitude,
                    altitude_parameters
                );

            // Attitude controller + motor mixer.
            const auto flight_output =
                aerial::control::UpdateFlightController(
                    state,
                    drone_parameters,
                    target_orientation,
                    altitude_output.collective_thrust,
                    dt,
                    flight_parameters,
                    flight_state
                );

            simulation.Step(flight_output.motor_commands, dt);

            const auto& updated_state = simulation.GetDroneState();

            const double horizontal_speed =
                updated_state.velocity.head<2>().norm();

            result.maximum_speed =
                std::max(result.maximum_speed, horizontal_speed);
        }

        const auto& final_state = simulation.GetDroneState();

        result.velocity = final_state.velocity.head<2>();
        result.position = final_state.position.head<2>();
        result.altitude = final_state.position.z();
        result.vertical_velocity = final_state.velocity.z();

        return result;
    }

    void TestScenario(
        const char* name,
        const Eigen::Vector2d& initial_velocity,
        const Eigen::Vector2d& target_velocity,
        double target_yaw = 0.0
    )
    {
        const auto result = RunScenario(
            initial_velocity,
            target_velocity,
            target_yaw
        );

        const double velocity_error =
            (result.velocity - target_velocity).norm();

        const double altitude_error =
            std::abs(result.altitude - 1.0);

        std::cout
            << name
            << "\n  Final velocity: ("
            << result.velocity.x() << ", "
            << result.velocity.y() << ") m/s"
            << "\n  Target velocity: ("
            << target_velocity.x() << ", "
            << target_velocity.y() << ") m/s"
            << "\n  Velocity error: "
            << velocity_error << " m/s"
            << "\n  Final altitude: "
            << result.altitude << " m"
            << "\n  Maximum speed: "
            << result.maximum_speed << " m/s\n"
            << std::flush;

        assert(result.velocity.allFinite());
        assert(result.position.allFinite());
        assert(std::isfinite(result.altitude));
        assert(std::isfinite(result.vertical_velocity));

        // Initial acceptance thresholds.
        assert(velocity_error < 0.2);
        assert(altitude_error < 0.25);
        assert(std::abs(result.vertical_velocity) < 0.2);
    }
}

int main()
{
    TestScenario(
        "Stationary hover",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero()
    );

    TestScenario(
        "Positive X velocity tracking",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(1.0, 0.0)
    );

    TestScenario(
        "Positive Y velocity tracking",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(0.0, 1.0)
    );

    TestScenario(
        "Horizontal drift recovery",
        Eigen::Vector2d(2.0, 0.0),
        Eigen::Vector2d::Zero()
    );

    TestScenario(
        "Yaw rotated velocity tracking",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(1.0, 0.0),
        3.14159265358979323846 / 2.0
    );

    std::cout
        << "\nClosed-loop horizontal velocity tests PASSED\n";

    return 0;
}
