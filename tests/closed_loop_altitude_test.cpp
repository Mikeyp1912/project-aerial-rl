// /tests/closed_loop_altitude_test.cpp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "aerial/control/altitude_controller.hpp"
#include "aerial/control/flight_controller.hpp"
#include "aerial/physics/simulation_environment.hpp"

namespace
{
    constexpr double dt = 1.0 / 120.0;
    constexpr double duration = 8.0;

    struct TestResult
    {
        double final_altitude = 0.0;
        double final_vertical_velocity = 0.0;
        double maximum_altitude_error = 0.0;
    };

    TestResult RunScenario(
        double initial_altitude,
        double initial_vertical_velocity,
        double target_altitude
    )
    {
        aerial::physics::Simulation simulation;

        auto state = simulation.GetDroneState();

        state.position.z() = initial_altitude;
        state.velocity.z() = initial_vertical_velocity;

        // Initialize a new simulation from the modified state.
        simulation.Reset(state);

        const auto& drone_parameters = simulation.GetParameters();

        aerial::control::AltitudeControllerParameters altitude_parameters;
        aerial::control::FlightControllerParameters flight_parameters;
        aerial::control::FlightControllerState flight_state;

        const Eigen::Quaterniond target_orientation =
            Eigen::Quaterniond::Identity();

        TestResult result;

        const int steps = static_cast<int>(duration / dt);

        for (int i = 0; i < steps; ++i) {
            const auto altitude_output =
                aerial::control::UpdateAltitudeController(
                    simulation.GetDroneState(),
                    drone_parameters,
                    target_altitude,
                    altitude_parameters
                );

            const auto flight_output =
                aerial::control::UpdateFlightController(
                    simulation.GetDroneState(),
                    drone_parameters,
                    target_orientation,
                    altitude_output.collective_thrust,
                    dt,
                    flight_parameters,
                    flight_state
                );

            simulation.Step(flight_output.motor_commands, dt);

            const auto& current_state = simulation.GetDroneState();

            const double altitude_error =
                std::abs(target_altitude - current_state.position.z());

            result.maximum_altitude_error = std::max(
                result.maximum_altitude_error,
                altitude_error
            );
        }

        result.final_altitude =
            simulation.GetDroneState().position.z();

        result.final_vertical_velocity =
            simulation.GetDroneState().velocity.z();

        return result;
    }

    void TestScenario(
        const char* name,
        double initial_altitude,
        double initial_vertical_velocity,
        double target_altitude
    )
    {
        const auto result = RunScenario(
            initial_altitude,
            initial_vertical_velocity,
            target_altitude
        );

        const double final_error =
            std::abs(result.final_altitude - target_altitude);

        std::cout
            << name
            << " | Final altitude: " << result.final_altitude
            << " m | Final velocity: "
            << result.final_vertical_velocity
            << " m/s | Final error: "
            << final_error
            << " m\n";

        assert(std::isfinite(final_error));
        assert(std::isfinite(result.final_vertical_velocity));

        assert(final_error < 0.15);
        assert(std::abs(result.final_vertical_velocity) < 0.15);
    }
}

int main()
{
    TestScenario("Stationary hover", 0.0, 0.0, 0.0);
    TestScenario("Altitude recovery", -2.0, 0.0, 0.0);
    TestScenario("Velocity recovery", 0.0, -2.0, 0.0);

    std::cout << "Closed-loop altitude tests PASSED\n";

    return 0;
}
