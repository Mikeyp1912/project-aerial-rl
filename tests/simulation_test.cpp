// /tests/simulation_test.cpp
#include <cmath>
#include <iostream>

#include "aerial/physics/simulation.hpp"
#include "aerial/physics/integrator.hpp"
#include "aerial/physics/motor_model.hpp"
#include "aerial/physics/rotational_dynamics.hpp"

namespace
{
    bool Near(double a, double b, double tolerance = 1e-9)
    {
        return std::abs(a - b) < tolerance;
    }
}

int main()
{
    using namespace aerial::physics;

    DroneParameters parameters;

    DroneState unified;
    DroneState manual;

    MotorState unified_motors;
    MotorState manual_motors;

    unified.position = {0.0, 0.0, 1.0};
    manual.position = unified.position;

    const double hover_command =
        parameters.mass * gravity /
        (4.0 * parameters.max_motor_thrust);

    const MotorCommands commands = {
        hover_command + 0.002,
        hover_command + 0.002,
        hover_command - 0.002,
        hover_command - 0.002
    };

    // Match the simulator's initial motor state.
    const double hover_thrust =
        parameters.mass * gravity / 4.0;

    unified_motors.thrusts.fill(hover_thrust);
    manual_motors.thrusts.fill(hover_thrust);

    constexpr double dt = 1.0 / 120.0;

    for (int i = 0; i < 120; ++i)
    {
        // New unified simulation.
        StepSimulation(
            unified,
            unified_motors,
            parameters,
            commands,
            dt
        );

        // Original simulation pipeline.
        const auto targets =
            CommandsToThrust(parameters, commands);

        UpdateMotorDynamics(
            manual_motors,
            parameters,
            targets,
            dt
        );

        const MotorOutput output =
            CalculateMotorOutput(
                parameters,
                manual_motors.thrusts
            );

        const Eigen::Vector3d world_force =
            manual.orientation * output.force;

        Integrate(
            manual,
            parameters,
            world_force,
            dt
        );

        IntegrateRotation(
            manual,
            parameters,
            output.torque,
            dt
        );
    }

    // Both approaches must produce the same state.
    if ((unified.position - manual.position).norm() > 1e-9 ||
        (unified.velocity - manual.velocity).norm() > 1e-9 ||
        (unified.angular_velocity - manual.angular_velocity).norm() > 1e-9 ||
        unified.orientation.angularDistance(manual.orientation) > 1e-9)
    {
        std::cerr << "Unified simulation equivalence test failed\n";
        return 1;
    }

    // Motor states must also match.
    for (std::size_t i = 0; i < 4; ++i)
    {
        if (!Near(
            unified_motors.thrusts[i],
            manual_motors.thrusts[i]))
        {
            std::cerr << "Motor state equivalence test failed\n";
            return 1;
        }
    }

    // A zero timestep must not change the state.
    const DroneState previous_state = unified;
    const MotorState previous_motors = unified_motors;

    StepSimulation(
        unified,
        unified_motors,
        parameters,
        commands,
        0.0
    );

    if ((unified.position - previous_state.position).norm() > 1e-9 ||
        (unified.velocity - previous_state.velocity).norm() > 1e-9 ||
        (unified.angular_velocity - previous_state.angular_velocity).norm() > 1e-9 ||
        unified.orientation.angularDistance(previous_state.orientation) > 1e-9)
    {
        std::cerr << "Zero timestep state test failed\n";
        return 1;
    }

    for (std::size_t i = 0; i < 4; ++i)
    {
        if (!Near(unified_motors.thrusts[i], previous_motors.thrusts[i]))
        {
            std::cerr << "Zero timestep motor test failed\n";
            return 1;
        }
    }

    std::cout << "Unified simulation tests passed\n";
    return 0;
}

