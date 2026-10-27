// /tests/closed_loop_disturbance_injection_test.cpp

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
    constexpr int total_steps = 12 * 120;
    constexpr int injection_start = 2 * 120;
    constexpr int injection_end = injection_start + 30;

    struct Result
    {
        double maximum_position_error = 0.0;
        double maximum_horizontal_speed = 0.0;
        double final_position_error = 0.0;
        double final_horizontal_speed = 0.0;
        double final_altitude_error = 0.0;
        double final_angular_speed = 0.0;
        double settling_time = -1.0;
    };

    Result RunScenario(
        const Eigen::Vector3d& force,
        const Eigen::Vector3d& torque
    )
    {
        using namespace aerial;

        physics::Simulation simulation;
        const auto& drone_parameters = simulation.GetParameters();

        control::HorizontalPositionControllerParameters position_params;
        control::HorizontalVelocityControllerParameters velocity_params;
        control::AltitudeControllerParameters altitude_params;
        control::FlightControllerParameters flight_params;
        control::FlightControllerState flight_state;

        const Eigen::Vector2d target_position = Eigen::Vector2d::Zero();
        constexpr double target_altitude = 1.0;
        constexpr double target_yaw = 0.0;

        constexpr double position_tolerance = 0.15;
        constexpr double velocity_tolerance = 0.15;
        constexpr double altitude_tolerance = 0.15;
        constexpr double angular_tolerance = 0.15;

        Result result;

        // Last time outside tolerance, relative to disturbance removal.
        double last_unsettled_time = -1.0;

        for (int i = 0; i < total_steps; ++i) {
            if (i == injection_start) {
                simulation.SetExternalForce(force);
                simulation.SetExternalTorque(torque);
            }

            if (i == injection_end) {
                simulation.ClearExternalDisturbances();
            }

            const auto& state = simulation.GetDroneState();

            const auto position_output =
                control::UpdateHorizontalPositionController(
                    state,
                    target_position,
                    position_params
                );

            const auto velocity_output =
                control::UpdateHorizontalVelocityController(
                    state,
                    position_output.desired_velocity,
                    target_yaw,
                    velocity_params
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

            const auto altitude_output =
                control::UpdateAltitudeController(
                    state,
                    drone_parameters,
                    target_altitude,
                    altitude_params
                );

            const auto flight_output =
                control::UpdateFlightController(
                    state,
                    drone_parameters,
                    target_orientation,
                    altitude_output.collective_thrust,
                    dt,
                    flight_params,
                    flight_state
                );

            simulation.Step(flight_output.motor_commands, dt);

            const auto& updated = simulation.GetDroneState();

            const double position_error =
                updated.position.head<2>().norm();

            const double horizontal_speed =
                updated.velocity.head<2>().norm();

            const double altitude_error =
                std::abs(updated.position.z() - target_altitude);

            const double vertical_speed =
                std::abs(updated.velocity.z());

            const double angular_speed =
                updated.angular_velocity.norm();

            // Measure peak response only after injection begins.
            if (i >= injection_start) {
                result.maximum_position_error = std::max(
                    result.maximum_position_error,
                    position_error
                );

                result.maximum_horizontal_speed = std::max(
                    result.maximum_horizontal_speed,
                    horizontal_speed
                );
            }

            // Recovery begins once the disturbance is removed.
            if (i >= injection_end) {
                if (position_error >= position_tolerance ||
                    horizontal_speed >= velocity_tolerance ||
                    altitude_error >= altitude_tolerance ||
                    vertical_speed >= velocity_tolerance ||
                    angular_speed >= angular_tolerance) {
                    last_unsettled_time =
                        simulation.GetSimulationTime() -
                        injection_end * dt;
                }
            }
        }

        const auto& final_state = simulation.GetDroneState();

        result.final_position_error =
            final_state.position.head<2>().norm();

        result.final_horizontal_speed =
            final_state.velocity.head<2>().norm();

        result.final_altitude_error =
            std::abs(final_state.position.z() - target_altitude);

        result.final_angular_speed =
            final_state.angular_velocity.norm();

        const bool settled =
            result.final_position_error < position_tolerance &&
            result.final_horizontal_speed < velocity_tolerance &&
            result.final_altitude_error < altitude_tolerance &&
            std::abs(final_state.velocity.z()) < velocity_tolerance &&
            result.final_angular_speed < angular_tolerance;

        if (settled) {
            result.settling_time =
                std::max(0.0, last_unsettled_time);
        }

        return result;
    }

    void TestScenario(
        const char* name,
        const Eigen::Vector3d& force,
        const Eigen::Vector3d& torque
    )
    {
        const Result result = RunScenario(force, torque);

        std::cout
            << name
            << "\n  Maximum position error: "
            << result.maximum_position_error << " m"
            << "\n  Maximum horizontal speed: "
            << result.maximum_horizontal_speed << " m/s"
            << "\n  Final position error: "
            << result.final_position_error << " m"
            << "\n  Final horizontal speed: "
            << result.final_horizontal_speed << " m/s"
            << "\n  Final altitude error: "
            << result.final_altitude_error << " m"
            << "\n  Final angular speed: "
            << result.final_angular_speed << " rad/s"
            << "\n  Settling time after removal: "
            << result.settling_time << " s\n"
            << std::flush;

        assert(std::isfinite(result.maximum_position_error));
        assert(std::isfinite(result.maximum_horizontal_speed));
        assert(std::isfinite(result.final_position_error));
        assert(std::isfinite(result.final_horizontal_speed));
        assert(std::isfinite(result.final_altitude_error));
        assert(std::isfinite(result.final_angular_speed));

        assert(result.final_position_error < 0.15);
        assert(result.final_horizontal_speed < 0.15);
        assert(result.final_altitude_error < 0.15);
        assert(result.final_angular_speed < 0.15);
        assert(result.settling_time >= 0.0);
    }
}

int main()
{
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    TestScenario(
        "Horizontal force disturbance",
        Eigen::Vector3d(4.0, 0.0, 0.0),
        zero
    );

    TestScenario(
        "Lateral force disturbance",
        Eigen::Vector3d(0.0, -4.0, 0.0),
        zero
    );

    TestScenario(
        "Roll torque disturbance",
        zero,
        Eigen::Vector3d(0.02, 0.0, 0.0)
    );

    TestScenario(
        "Pitch torque disturbance",
        zero,
        Eigen::Vector3d(0.0, 0.02, 0.0)
    );

    TestScenario(
        "Combined force and torque disturbance",
        Eigen::Vector3d(3.0, -2.0, 0.0),
        Eigen::Vector3d(0.01, -0.01, 0.0)
    );

    std::cout
        << "\nClosed-loop disturbance injection tests PASSED\n";

    return 0;
}
