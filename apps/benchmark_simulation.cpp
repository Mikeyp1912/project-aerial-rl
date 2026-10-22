// /apps/benchmark_simulation.cpp

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "aerial/physics/integrator.hpp"
#include "aerial/physics/simulation_environment.hpp"

int main(int argc, char* argv[]) {
    using namespace aerial::physics;

    std::uint64_t num_steps = 1'000'000;

    if (argc > 2) {
        std::cerr << "Usage: benchmark_simulation [steps]\n";
        return 1;
    }

    if (argc == 2) {
        try {
            const std::string argument = argv[1];
            std::size_t parsed = 0;

            if (argument.empty() || argument[0] == '-') {
                throw std::invalid_argument("Invalid step count");
            }

            num_steps = std::stoull(argument, &parsed);

            if (parsed != argument.size() || num_steps == 0) {
                throw std::invalid_argument("Invalid step count");
            }
        }
        catch (const std::exception&) {
            std::cerr << "Invalid step count\n";
            return 1;
        }
    }

    constexpr double physics_dt = 1.0 / 120.0;

    Simulation simulation;

    const auto& parameters = simulation.GetParameters();

    const double hover_command = parameters.mass * gravity / (4.0 * parameters.max_motor_thrust);

    // Symmetric thrust commands: no intended roll,
    // pitch or yaw torque
    const MotorCommands commands = {
        hover_command,
        hover_command,
        hover_command,
        hover_command
    };

    // Warm up the benchmark workload
    constexpr std::uint64_t warmup_steps = 10'000;

    for (std::uint64_t i = 0; i < warmup_steps; ++i) {
        simulation.Step(commands, physics_dt);
    }

    // Ensure every measured run starts from the same state
    simulation.Reset();

    using Clock = std::chrono::steady_clock;

    const auto start = Clock::now();

    for (std::uint64_t i = 0; i < num_steps; ++i) {
        simulation.Step(commands, physics_dt);
    }

    const auto end = Clock::now();

    const double elapsed_seconds = std::chrono::duration<double>(end - start).count();

    const double simulated_seconds = simulation.GetSimulationTime();

    const double steps_per_second = static_cast<double>(num_steps) / elapsed_seconds;

    const double realtime_factor = simulated_seconds / elapsed_seconds;

    const auto& state = simulation.GetDroneState();

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\nProject Aerial RL - Physics Benchmark\n";
    std::cout << "--------------------------------------\n";

    std::cout << "Physics timestep:     "
              << physics_dt * 1000.0 << " ms\n";

    std::cout << "Physics steps:        "
              << simulation.GetStepCount() << '\n';

    std::cout << "Simulated time:       "
              << simulated_seconds << " s\n";

    std::cout << "Execution time:       "
              << elapsed_seconds << " s\n";

    std::cout << "Throughput:           "
              << std::setprecision(0)
              << steps_per_second << " steps/s\n";

    std::cout << "Real-time factor:     "
              << std::setprecision(2)
              << realtime_factor << "x\n";

    std::cout << std::setprecision(3);

    std::cout << "Final position:       ("
              << state.position.x() << ", "
              << state.position.y() << ", "
              << state.position.z() << ")\n";

    std::cout << "Final angular speed:  "
              << state.angular_velocity.norm() << " rad/s\n";

    if (!std::isfinite(elapsed_seconds) ||
        elapsed_seconds <= 0.0 ||
        !state.position.allFinite() ||
        !state.velocity.allFinite() ||
        !state.angular_velocity.allFinite() ||
        !state.orientation.coeffs().allFinite())
    {
        std::cerr << "Benchmark produced invalid results\n";
        return 1;
    }

    return 0;
}
