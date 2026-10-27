// /tests/closed_loop_horizontal_position_test.cpp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "aerial/control/altitude_controller.hpp"
#include "aerial/control/flight_controller.hpp"
#include "aerial/control/horizontal_position_controller.hpp"
#include "aerial/control/horizontal_velocity_controller.hpp"
#include "aerial/physics/simulation_environment.hpp"

namespace
{
    constexpr double dt = 1.0 / 120.0;
    constexpr double duration = 12.0;
    constexpr double pi = 3.14159265358979323846;

    struct TestResult
    {
        Eigen::Vector2d position = Eigen::Vector2d::Zero();
        Eigen::Vector2d velocity = Eigen::Vector2d::Zero();

        double altitude = 0.0;
        double vertical_velocity = 0.0;

        double maximum_position_error = 0.0;
        double maximum_speed = 0.0;
        double settling_time = -1.0;
    };

    TestResult RunScenario(
        const Eigen::Vector2d& initial_position,
        const Eigen::Vector2d& initial_velocity,
        const Eigen::Vector2d& target_position,
        double target_yaw = 0.0
    )
    {
        aerial::physics::Simulation simulation;

        auto initial_state = simulation.GetDroneState();

        initial_state.position.x() = initial_position.x();
        initial_state.position.y() = initial_position.y();

        initial_state.velocity.x() = initial_velocity.x();
        initial_state.velocity.y() = initial_velocity.y();

        simulation.Reset(initial_state);

        const auto& drone_parameters = simulation.GetParameters();

        aerial::control::HorizontalPositionControllerParameters
            position_parameters;

        aerial::control::HorizontalVelocityControllerParameters
            velocity_parameters;

        aerial::control::AltitudeControllerParameters
            altitude_parameters;

        aerial::control::FlightControllerParameters
            flight_parameters;

        aerial::control::FlightControllerState flight_state;

        const double target_altitude = initial_state.position.z();

        TestResult result;

        // Settled means remaining inside both tolerances.
        constexpr double position_tolerance = 0.15;
        constexpr double velocity_tolerance = 0.15;

        // Record the last time the drone was outside tolerance.
        // If it remains inside afterward, that is its settling time.
        double last_unsettled_time = -1.0;

        const int steps = static_cast<int>(duration / dt);

        for (int i = 0; i < steps; ++i) {
            const auto& state = simulation.GetDroneState();

            // Outer loop: position -> desired velocity.
            const auto position_output =
                aerial::control::UpdateHorizontalPositionController(
                    state,
                    target_position,
                    position_parameters
                );

            // Inner horizontal loop: velocity -> roll/pitch.
            const auto velocity_output =
                aerial::control::UpdateHorizontalVelocityController(
                    state,
                    position_output.desired_velocity,
                    target_yaw,
                    velocity_parameters
                );

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

            // Maintain the starting altitude.
            const auto altitude_output =
                aerial::control::UpdateAltitudeController(
                    state,
                    drone_parameters,
                    target_altitude,
                    altitude_parameters
                );

            // Attitude/rate control and motor mixing.
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

            const auto& updated = simulation.GetDroneState();

            const double position_error =
                (target_position -
                 updated.position.head<2>().eval()).norm();

            const double speed =
                updated.velocity.head<2>().norm();

            result.maximum_position_error =
                std::max(result.maximum_position_error, position_error);

            result.maximum_speed =
                std::max(result.maximum_speed, speed);

            if (position_error >= position_tolerance ||
                speed >= velocity_tolerance) {
                last_unsettled_time = simulation.GetSimulationTime();
            }
        }

        const auto& final_state = simulation.GetDroneState();

        result.position = final_state.position.head<2>();
        result.velocity = final_state.velocity.head<2>();

        result.altitude = final_state.position.z();
        result.vertical_velocity = final_state.velocity.z();

        const double final_position_error =
            (target_position - result.position).norm();

        const double final_speed = result.velocity.norm();

        if (final_position_error < position_tolerance &&
            final_speed < velocity_tolerance) {
            result.settling_time =
                last_unsettled_time < 0.0
                    ? 0.0
                    : last_unsettled_time;
        }

        return result;
    }

    void TestScenario(
        const char* name,
        const Eigen::Vector2d& initial_position,
        const Eigen::Vector2d& initial_velocity,
        const Eigen::Vector2d& target_position,
        double target_yaw = 0.0
    )
    {
        const auto result = RunScenario(
            initial_position,
            initial_velocity,
            target_position,
            target_yaw
        );

        const double position_error =
            (target_position - result.position).norm();

        const double speed = result.velocity.norm();

        const double altitude_error =
            std::abs(result.altitude - 1.0);

        std::cout
            << name
            << "\n  Final position: ("
            << result.position.x() << ", "
            << result.position.y() << ") m"
            << "\n  Target position: ("
            << target_position.x() << ", "
            << target_position.y() << ") m"
            << "\n  Position error: " << position_error << " m"
            << "\n  Final horizontal speed: " << speed << " m/s"
            << "\n  Final altitude: " << result.altitude << " m"
            << "\n  Maximum horizontal speed: "
            << result.maximum_speed << " m/s"
            << "\n  Settling time: "
            << result.settling_time << " s\n"
            << std::flush;

        assert(result.position.allFinite());
        assert(result.velocity.allFinite());

        assert(std::isfinite(result.altitude));
        assert(std::isfinite(result.vertical_velocity));

        assert(position_error < 0.15);
        assert(speed < 0.15);

        assert(altitude_error < 0.25);
        assert(std::abs(result.vertical_velocity) < 0.2);

        assert(result.settling_time >= 0.0);
    }
}

int main()
{
    TestScenario(
        "Stationary hover",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero()
    );

    TestScenario(
        "Positive X position tracking",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(5.0, 0.0)
    );

    TestScenario(
        "Positive Y position tracking",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(0.0, 5.0)
    );

    TestScenario(
        "Horizontal drift recovery",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(2.0, 0.0),
        Eigen::Vector2d::Zero()
    );

    TestScenario(
        "Yaw rotated position tracking",
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d::Zero(),
        Eigen::Vector2d(5.0, 0.0),
        pi / 2.0
    );

    std::cout
        << "\nClosed-loop horizontal position tests PASSED\n";

    return 0;
}
