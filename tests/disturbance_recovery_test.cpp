// /tests/disturbance_recovery_test.cpp

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

    struct RecoveryResult
    {
        double final_position_error = 0.0;
        double final_speed = 0.0;
        double final_altitude_error = 0.0;
        double final_angular_speed = 0.0;

        double maximum_position_error = 0.0;
        double maximum_speed = 0.0;

        double settling_time = -1.0;
    };

    RecoveryResult RunRecovery(
        const aerial::physics::DroneState& initial_state,
        const Eigen::Vector2d& target_position,
        double target_altitude,
        double target_yaw = 0.0
    )
    {
        aerial::physics::Simulation simulation;
        simulation.Reset(initial_state);

        const auto& parameters = simulation.GetParameters();

        aerial::control::HorizontalPositionControllerParameters position_params;
        aerial::control::HorizontalVelocityControllerParameters velocity_params;
        aerial::control::AltitudeControllerParameters altitude_params;
        aerial::control::FlightControllerParameters flight_params;
        aerial::control::FlightControllerState flight_state;

        RecoveryResult result;

        constexpr double position_tolerance = 0.15;
        constexpr double velocity_tolerance = 0.15;
        constexpr double altitude_tolerance = 0.15;
        constexpr double angular_tolerance = 0.15;

        double last_unsettled_time = -1.0;

        const int steps = static_cast<int>(duration / dt);

        for (int i = 0; i < steps; ++i) {
            const auto& state = simulation.GetDroneState();

            const auto position_output =
                aerial::control::UpdateHorizontalPositionController(
                    state, target_position, position_params
                );

            const auto velocity_output =
                aerial::control::UpdateHorizontalVelocityController(
                    state,
                    position_output.desired_velocity,
                    target_yaw,
                    velocity_params
                );

            const Eigen::Quaterniond target_orientation =
                Eigen::Quaterniond(
                    Eigen::AngleAxisd(target_yaw, Eigen::Vector3d::UnitZ())
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

            const auto altitude_output =
                aerial::control::UpdateAltitudeController(
                    state, parameters, target_altitude, altitude_params
                );

            const auto flight_output =
                aerial::control::UpdateFlightController(
                    state,
                    parameters,
                    target_orientation,
                    altitude_output.collective_thrust,
                    dt,
                    flight_params,
                    flight_state
                );

            simulation.Step(flight_output.motor_commands, dt);

            const auto& updated = simulation.GetDroneState();

            const double position_error =
                (target_position - updated.position.head<2>().eval()).norm();

            const double horizontal_speed =
                updated.velocity.head<2>().norm();

            const double altitude_error =
                std::abs(target_altitude - updated.position.z());

            const double angular_speed =
                updated.angular_velocity.norm();

            result.maximum_position_error =
                std::max(result.maximum_position_error, position_error);

            result.maximum_speed =
                std::max(result.maximum_speed, horizontal_speed);

            if (position_error >= position_tolerance ||
                horizontal_speed >= velocity_tolerance ||
                altitude_error >= altitude_tolerance ||
                std::abs(updated.velocity.z()) >= velocity_tolerance ||
                angular_speed >= angular_tolerance) {

                last_unsettled_time = simulation.GetSimulationTime();
            }
        }

        const auto& final_state = simulation.GetDroneState();

        result.final_position_error =
            (target_position - final_state.position.head<2>().eval()).norm();

        result.final_speed = final_state.velocity.head<2>().norm();

        result.final_altitude_error =
            std::abs(target_altitude - final_state.position.z());

        result.final_angular_speed =
            final_state.angular_velocity.norm();

        const bool settled =
            result.final_position_error < position_tolerance &&
            result.final_speed < velocity_tolerance &&
            result.final_altitude_error < altitude_tolerance &&
            std::abs(final_state.velocity.z()) < velocity_tolerance &&
            result.final_angular_speed < angular_tolerance;

        if (settled) {
            result.settling_time =
                last_unsettled_time < 0.0 ? 0.0 : last_unsettled_time;
        }

        return result;
    }

    void TestRecovery(
        const char* name,
        const aerial::physics::DroneState& initial_state,
        const Eigen::Vector2d& target_position,
        double target_altitude,
        double target_yaw = 0.0
    )
    {
        const auto result = RunRecovery(
            initial_state, target_position, target_altitude, target_yaw
        );

        std::cout
            << name
            << "\n  Final position error: " << result.final_position_error << " m"
            << "\n  Final horizontal speed: " << result.final_speed << " m/s"
            << "\n  Final altitude error: " << result.final_altitude_error << " m"
            << "\n  Final angular speed: " << result.final_angular_speed << " rad/s"
            << "\n  Maximum position error: " << result.maximum_position_error << " m"
            << "\n  Maximum speed: " << result.maximum_speed << " m/s"
            << "\n  Settling time: " << result.settling_time << " s\n"
            << std::flush;

        assert(std::isfinite(result.final_position_error));
        assert(std::isfinite(result.final_speed));
        assert(std::isfinite(result.final_altitude_error));
        assert(std::isfinite(result.final_angular_speed));

        assert(result.final_position_error < 0.15);
        assert(result.final_speed < 0.15);
        assert(result.final_altitude_error < 0.15);
        assert(result.final_angular_speed < 0.15);
        assert(result.settling_time >= 0.0);
    }
}

int main()
{
    aerial::physics::Simulation reference;
    const auto initial = reference.GetDroneState();

    // 1. Recover from a horizontal displacement.
    {
        auto state = initial;
        state.position.x() = 3.0;
        state.position.y() = -2.0;

        TestRecovery(
            "Horizontal displacement",
            state,
            Eigen::Vector2d::Zero(),
            1.0
        );
    }

    // 2. Recover from an initial horizontal velocity.
    {
        auto state = initial;
        state.velocity.x() = 3.0;
        state.velocity.y() = -1.0;

        TestRecovery(
            "Horizontal velocity disturbance",
            state,
            Eigen::Vector2d::Zero(),
            1.0
        );
    }

    // 3. Recover from a 20-degree roll disturbance.
    {
        auto state = initial;
        state.orientation = Eigen::Quaterniond(
            Eigen::AngleAxisd(
                20.0 * pi / 180.0,
                Eigen::Vector3d::UnitX()
            )
        );

        TestRecovery(
            "Initial roll disturbance",
            state,
            Eigen::Vector2d::Zero(),
            1.0
        );
    }

    // 4. Recover from an angular velocity disturbance.
    {
        auto state = initial;
        state.angular_velocity =
            Eigen::Vector3d(1.0, -0.5, 0.2);

        TestRecovery(
            "Angular velocity disturbance",
            state,
            Eigen::Vector2d::Zero(),
            1.0
        );
    }

    // 5. Recover while maintaining a 90-degree yaw target.
    {
        auto state = initial;
        state.position.x() = 2.0;
        state.velocity.y() = 1.0;

        TestRecovery(
            "Yaw-rotated disturbance recovery",
            state,
            Eigen::Vector2d::Zero(),
            1.0,
            pi / 2.0
        );
    }

    std::cout << "\nDisturbance recovery tests PASSED\n";
    return 0;
}
