// /tests/simulation_environment_test.cpp

#include <cmath>
#include <iostream>

#include "aerial/physics/simulation_environment.hpp"
#include "aerial/physics/integrator.hpp"

namespace
{
    bool Near(double a, double b, double tolerance = 1e-9)
    {
        return std::abs(a - b) < tolerance;
    }

    bool SameState(
        const aerial::physics::DroneState& a,
        const aerial::physics::DroneState& b
    )
    {
        return (a.position - b.position).norm() < 1e-9 &&
               (a.velocity - b.velocity).norm() < 1e-9 &&
               (a.angular_velocity - b.angular_velocity).norm() < 1e-9 &&
               a.orientation.angularDistance(b.orientation) < 1e-9;
    }
}

int main()
{
    using namespace aerial::physics;

    Simulation simulation;

    // Test 1: Default reset state.
    simulation.Reset();

    if (!Near(simulation.GetDroneState().position.z(), 1.0) ||
        !Near(simulation.GetDroneState().velocity.norm(), 0.0))
    {
        std::cerr << "Default reset test failed\n";
        return 1;
    }

    for (double thrust : simulation.GetMotorState().thrusts)
    {
        if (!Near(thrust, 0.0))
        {
            std::cerr << "Motor reset test failed\n";
            return 1;
        }
    }

    // Test 2: Simulation advances without raylib.
    const MotorCommands zero_commands{};

    simulation.Step(zero_commands, 0.1);

    if (!(simulation.GetDroneState().position.z() < 1.0))
    {
        std::cerr << "Headless simulation step failed\n";
        return 1;
    }

    // Test 3: Custom reset state.
    DroneState custom_state;
    custom_state.position = {2.0, -1.0, 5.0};
    custom_state.velocity = {1.0, 0.0, 0.0};

    simulation.Reset(custom_state);

    if (!SameState(simulation.GetDroneState(), custom_state))
    {
        std::cerr << "Custom reset test failed\n";
        return 1;
    }

    // Test 4: Identical resets produce identical trajectories.
    const double hover_command =
        simulation.GetParameters().mass * gravity /
        (4.0 * simulation.GetParameters().max_motor_thrust);

    const MotorCommands commands = {
        hover_command + 0.002,
        hover_command + 0.002,
        hover_command - 0.002,
        hover_command - 0.002
    };

    constexpr double dt = 1.0 / 120.0;

    simulation.Reset();

    for (int i = 0; i < 120; ++i)
    {
        simulation.Step(commands, dt);
    }

    const DroneState first_run = simulation.GetDroneState();
    const MotorState first_motors = simulation.GetMotorState();

    simulation.Reset();

    for (int i = 0; i < 120; ++i)
    {
        simulation.Step(commands, dt);
    }

    if (!SameState(first_run, simulation.GetDroneState()))
    {
        std::cerr << "Deterministic reset test failed\n";
        return 1;
    }

    for (std::size_t i = 0; i < first_motors.thrusts.size(); ++i)
    {
        if (!Near(
            first_motors.thrusts[i],
            simulation.GetMotorState().thrusts[i]))
        {
            std::cerr << "Deterministic motor state test failed\n";
            return 1;
        }
    }

    // Test 5: Invalid timestep must not advance simulation.
    const DroneState before_invalid_step = simulation.GetDroneState();
    const MotorState motors_before_invalid_step = simulation.GetMotorState();

    simulation.Step(commands, 0.0);

    if (!SameState(before_invalid_step, simulation.GetDroneState()))
    {
        std::cerr << "Invalid timestep test failed\n";
        return 1;
    }

    for (std::size_t i = 0; i < motors_before_invalid_step.thrusts.size(); ++i)
    {
        if (!Near(
            motors_before_invalid_step.thrusts[i],
            simulation.GetMotorState().thrusts[i]))
        {
            std::cerr << "Invalid timestep motor test failed\n";
            return 1;
        }
    }

    
    // Test simulation time and step counting.
    simulation.Reset();

    if (!Near(simulation.GetSimulationTime(), 0.0) ||
        simulation.GetStepCount() != 0)
    {
        std::cerr << "Simulation clock reset test failed\n";
        return 1;
    }

    constexpr double test_dt = 1.0 / 120.0;

    for (int i = 0; i < 120; ++i)
    {
        simulation.Step(zero_commands, test_dt);
    }

    if (!Near(simulation.GetSimulationTime(), 1.0) ||
        simulation.GetStepCount() != 120)
    {
        std::cerr << "Simulation clock advancement test failed\n";
        return 1;
    }

    // Invalid timesteps must not advance the clock.
    simulation.Step(zero_commands, 0.0);
    simulation.Step(zero_commands, -1.0);

    if (!Near(simulation.GetSimulationTime(), 1.0) ||
        simulation.GetStepCount() != 120)
    {
        std::cerr << "Invalid timestep clock test failed\n";
        return 1;
    }

    // Reset must clear the clock and step counter.
    simulation.Reset();

    if (!Near(simulation.GetSimulationTime(), 0.0) ||
        simulation.GetStepCount() != 0)
    {
        std::cerr << "Simulation clock reset failed\n";
        return 1;
    }


    std::cout << "Simulation environment tests passed\n";
    return 0;
}
