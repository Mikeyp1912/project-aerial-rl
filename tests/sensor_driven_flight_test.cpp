
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
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

        double rms_position_error = 0.0;
        double maximum_position_error = 0.0;
        double rms_altitude_error = 0.0;
    };

    Result RunScenario(
        const aerial::sensors::SensorParameters& sensor_parameters,
        std::uint32_t seed = 1234
    )
    {
        using namespace aerial;

        physics::Simulation simulation;

        sensors::SensorScheduler scheduler(
            sensor_parameters,
            sensors::SensorSchedulerParameters{},
            seed
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

        double squared_position_error_sum = 0.0;
        double squared_altitude_error_sum = 0.0;
        double maximum_position_error = 0.0;

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

            // Outer position loop.
            const auto position_output =
                control::UpdateHorizontalPositionController(
                    observed,
                    target_position,
                    position_params
                );

            // Horizontal velocity loop.
            const auto velocity_output =
                control::UpdateHorizontalVelocityController(
                    observed,
                    position_output.desired_velocity,
                    target_yaw,
                    velocity_params
                );

            // Desired attitude from horizontal control.
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

            // Altitude control.
            const auto altitude_output =
                control::UpdateAltitudeController(
                    observed,
                    drone_parameters,
                    target_altitude,
                    altitude_params
                );

            // Attitude/rate control and motor mixing.
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

            // Advance the true physics simulation.
            simulation.Step(flight_output.motor_commands, dt);

            // Update sensors at their configured sampling rates.
            scheduler.Update(
                simulation.GetDroneState(),
                simulation.GetWorldAcceleration(),
                simulation.GetSimulationTime()
            );

            // Calculate performance using the true state.
            const auto& true_state = simulation.GetDroneState();

            const double position_error =
                (target_position -
                 true_state.position.head<2>().eval()).norm();

            const double altitude_error =
                std::abs(
                    true_state.position.z() - target_altitude
                );

            squared_position_error_sum +=
                position_error * position_error;

            squared_altitude_error_sum +=
                altitude_error * altitude_error;

            maximum_position_error = std::max(
                maximum_position_error,
                position_error
            );
        }

        // Final performance metrics.
        const auto& final_state = simulation.GetDroneState();

        Result result;

        result.position_error =
            (target_position -
             final_state.position.head<2>().eval()).norm();

        result.horizontal_speed =
            final_state.velocity.head<2>().norm();

        result.altitude_error =
            std::abs(
                final_state.position.z() - target_altitude
            );

        result.angular_speed =
            final_state.angular_velocity.norm();

        result.rms_position_error =
            std::sqrt(squared_position_error_sum / steps);

        result.rms_altitude_error =
            std::sqrt(squared_altitude_error_sum / steps);

        result.maximum_position_error =
            maximum_position_error;

        return result;
    }

    void PrintResult(
        const char* name,
        const Result& result
    )
    {
        std::cout
            << name
            << "\n  Position error: "
            << result.position_error << " m"
            << "\n  Horizontal speed: "
            << result.horizontal_speed << " m/s"
            << "\n  Altitude error: "
            << result.altitude_error << " m"
            << "\n  Angular speed: "
            << result.angular_speed << " rad/s"
            << "\n  RMS position error: "
            << result.rms_position_error << " m"
            << "\n  Maximum position error: "
            << result.maximum_position_error << " m"
            << "\n  RMS altitude error: "
            << result.rms_altitude_error << " m\n"
            << std::flush;

        assert(std::isfinite(result.position_error));
        assert(std::isfinite(result.horizontal_speed));
        assert(std::isfinite(result.altitude_error));
        assert(std::isfinite(result.angular_speed));
        assert(std::isfinite(result.rms_position_error));
        assert(std::isfinite(result.maximum_position_error));
        assert(std::isfinite(result.rms_altitude_error));
    }

    aerial::sensors::SensorParameters MakeNoiseParameters(
        double scale
    )
    {
        aerial::sensors::SensorParameters parameters;

        parameters.position_noise =
            scale * Eigen::Vector3d(0.01, 0.01, 0.02);

        parameters.velocity_noise =
            scale * Eigen::Vector3d(0.02, 0.02, 0.02);

        parameters.orientation_noise =
            scale * Eigen::Vector3d(0.002, 0.002, 0.003);

        parameters.angular_velocity_noise =
            scale * Eigen::Vector3d(0.005, 0.005, 0.005);

        parameters.accelerometer_noise =
            scale * Eigen::Vector3d(0.05, 0.05, 0.05);

        return parameters;
    }
}

int main()
{
    using namespace aerial;

    // ---------------------------------------------------------
    // Test 1: Ideal sensor-driven flight
    // ---------------------------------------------------------

    sensors::SensorParameters ideal_parameters;

    const Result ideal = RunScenario(ideal_parameters);

    PrintResult("Ideal multi-rate sensors", ideal);

    assert(ideal.position_error < 0.15);
    assert(ideal.horizontal_speed < 0.15);
    assert(ideal.altitude_error < 0.15);
    assert(ideal.angular_speed < 0.15);

    // ---------------------------------------------------------
    // Test 2: Noisy sensor-driven flight
    // ---------------------------------------------------------

    const sensors::SensorParameters noisy_parameters =
        MakeNoiseParameters(1.0);

    const Result noisy = RunScenario(noisy_parameters);

    PrintResult("Noisy multi-rate sensors", noisy);

    assert(noisy.position_error < 0.35);
    assert(noisy.horizontal_speed < 0.35);
    assert(noisy.altitude_error < 0.35);
    assert(noisy.angular_speed < 0.35);

    // ---------------------------------------------------------
    // Test 3: Sensor noise robustness benchmark
    // ---------------------------------------------------------

    constexpr std::array<double, 5> noise_scales{
        0.0,
        0.5,
        1.0,
        2.0,
        5.0
    };

    constexpr int trials = 5;

    std::cout
        << "\nNoise robustness benchmark\n"
        << "Scale,MeanFinalError,MeanRMSError,MeanMaxError,"
        << "MeanAltitudeRMSError\n";

    for (const double scale : noise_scales) {
        double final_error_sum = 0.0;
        double rms_error_sum = 0.0;
        double maximum_error_sum = 0.0;
        double altitude_rms_sum = 0.0;

        const auto parameters = MakeNoiseParameters(scale);

        for (int trial = 0; trial < trials; ++trial) {
            const Result result = RunScenario(
                parameters,
                static_cast<std::uint32_t>(1234 + trial)
            );

            assert(std::isfinite(result.position_error));
            assert(std::isfinite(result.rms_position_error));
            assert(std::isfinite(result.maximum_position_error));
            assert(std::isfinite(result.rms_altitude_error));

            final_error_sum += result.position_error;
            rms_error_sum += result.rms_position_error;
            maximum_error_sum += result.maximum_position_error;
            altitude_rms_sum += result.rms_altitude_error;
        }

        std::cout
            << scale << ","
            << final_error_sum / trials << ","
            << rms_error_sum / trials << ","
            << maximum_error_sum / trials << ","
            << altitude_rms_sum / trials << "\n";
    }

    std::cout
        << "\nSensor-driven flight tests PASSED\n";

    return 0;
}

