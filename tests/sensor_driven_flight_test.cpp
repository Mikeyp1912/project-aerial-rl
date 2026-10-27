// /tests/sensor_driven_flight_test.cpp
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

#include "aerial/sensors/observed_state.hpp"
#include "aerial/sensors/sensor_scheduler.hpp"

namespace
{
    constexpr double dt = 1.0 / 120.0;
    constexpr int steps = 12 * 120;

    struct Result
    {
        double position_error = 0.0;
        double horizontal_speed = 0.0;
        double altitude_error = 0.0;
        double angular_speed = 0.0;
    };

    Result RunScenario(
        const aerial::sensors::SensorParameters& sensor_parameters
    )
    {
        using namespace aerial;

        physics::Simulation simulation;

        sensors::SensorScheduler scheduler(
            sensor_parameters,
            sensors::SensorSchedulerParameters{},
            1234
        );

        const auto& drone_parameters = simulation.GetParameters();

        control::HorizontalPositionControllerParameters position_params;
        control::HorizontalVelocityControllerParameters velocity_params;
        control::AltitudeControllerParameters altitude_params;
        control::FlightControllerParameters flight_params;
        control::FlightControllerState flight_state;

        const Eigen::Vector2d target_position(2.0, 1.0);
        constexpr double target_altitude = 1.0;
        constexpr double target_yaw = 0.0;

        // Initialize sensors before the first control update.
        scheduler.Update(
            simulation.GetDroneState(),
            Eigen::Vector3d::Zero(),
            simulation.GetSimulationTime()
        );

        for (int i = 0; i < steps; ++i) {
            const physics::DroneState observed =
                sensors::MakeObservedState(
                    scheduler.GetMeasurements()
                );

            const auto position_output =
                control::UpdateHorizontalPositionController(
                    observed,
                    target_position,
                    position_params
                );

            const auto velocity_output =
                control::UpdateHorizontalVelocityController(
                    observed,
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
                    observed,
                    drone_parameters,
                    target_altitude,
                    altitude_params
                );

            const auto flight_output =
                control::UpdateFlightController(
                    observed,
                    drone_parameters,
                    target_orientation,
                    altitude_output.collective_thrust,
                    dt,
                    flight_params,
                    flight_state
                );

            // Only the physics simulation receives motor commands.
            simulation.Step(flight_output.motor_commands, dt);

            // Sensors sample the updated true state.
            scheduler.Update(
                simulation.GetDroneState(),
                simulation.GetWorldAcceleration(),
                simulation.GetSimulationTime()
            );
        }

        // Evaluate performance using true state, not sensor readings.
        const auto& final_state = simulation.GetDroneState();

        Result result;

        result.position_error =
            (target_position -
             final_state.position.head<2>().eval()).norm();

        result.horizontal_speed =
            final_state.velocity.head<2>().norm();

        result.altitude_error =
            std::abs(final_state.position.z() - target_altitude);

        result.angular_speed =
            final_state.angular_velocity.norm();

        return result;
    }

    void PrintResult(const char* name, const Result& result)
    {
        std::cout
            << name
            << "\n  Position error: " << result.position_error << " m"
            << "\n  Horizontal speed: " << result.horizontal_speed << " m/s"
            << "\n  Altitude error: " << result.altitude_error << " m"
            << "\n  Angular speed: " << result.angular_speed << " rad/s\n"
            << std::flush;

        assert(std::isfinite(result.position_error));
        assert(std::isfinite(result.horizontal_speed));
        assert(std::isfinite(result.altitude_error));
        assert(std::isfinite(result.angular_speed));
    }
}

int main()
{
    using namespace aerial;

    sensors::SensorParameters ideal_parameters;

    const Result ideal = RunScenario(ideal_parameters);

    PrintResult("Ideal multi-rate sensors", ideal);

    assert(ideal.position_error < 0.15);
    assert(ideal.horizontal_speed < 0.15);
    assert(ideal.altitude_error < 0.15);
    assert(ideal.angular_speed < 0.15);

    sensors::SensorParameters noisy_parameters;

    noisy_parameters.position_noise =
        Eigen::Vector3d(0.01, 0.01, 0.02);

    noisy_parameters.velocity_noise =
        Eigen::Vector3d(0.02, 0.02, 0.02);

    noisy_parameters.orientation_noise =
        Eigen::Vector3d(0.002, 0.002, 0.003);

    noisy_parameters.angular_velocity_noise =
        Eigen::Vector3d(0.005, 0.005, 0.005);

    noisy_parameters.accelerometer_noise =
        Eigen::Vector3d(0.05, 0.05, 0.05);

    const Result noisy = RunScenario(noisy_parameters);

    PrintResult("Noisy multi-rate sensors", noisy);

    // Initial tolerances are deliberately more generous.
    assert(noisy.position_error < 0.35);
    assert(noisy.horizontal_speed < 0.35);
    assert(noisy.altitude_error < 0.35);
    assert(noisy.angular_speed < 0.35);

    std::cout << "\nSensor-driven flight tests PASSED\n";

    return 0;
}
