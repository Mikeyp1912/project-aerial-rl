// /tests/flight_environment_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/rl/flight_environment.hpp"
#include "aerial/rl/observation.hpp"

int main()
{
    using namespace aerial;

    rl::FlightEnvironmentParameters parameters;
    parameters.time_step = 1.0 / 120.0;
    parameters.maximum_episode_time = 0.1;

    rl::FlightEnvironment environment(parameters);

    // RL observation must match the structured sensor measurements.
    {
        const rl::Observation observation =
            environment.GetRLObservation();

        const rl::Observation expected =
            rl::MakeObservation(environment.GetObservation());

        for (std::size_t i = 0; i < rl::kObservationSize; ++i) {
            assert(std::abs(observation[i] - expected[i]) < 1e-12);
        }
    }

    assert(environment.GetSimulationTime() == 0.0);
    assert(std::abs(environment.GetTrueState().position.z() - 1.0) < 1e-12);

    const physics::MotorCommands commands{0.0, 0.0, 0.0, 0.0};

    auto result = environment.Step(commands);

    {
        const rl::Observation observation =
            environment.GetRLObservation();

        const rl::Observation expected =
            rl::MakeObservation(result.observation);

        for (std::size_t i = 0; i < rl::kObservationSize; ++i) {
            assert(std::abs(observation[i] - expected[i]) < 1e-12);
        }
    }

    assert(!result.terminated);
    assert(!result.truncated);
    assert(result.reward == 0.0);
    assert(std::abs(
        environment.GetSimulationTime() - parameters.time_step
    ) < 1e-12);

    for (int i = 1; i < 12; ++i) {
        result = environment.Step(commands);
    }

    assert(result.truncated);
    assert(!result.terminated);

    bool rejected = false;

    try {
        environment.Step(commands);
    }
    catch (const std::logic_error&) {
        rejected = true;
    }

    assert(rejected);

    environment.Reset(123);

    assert(environment.GetSimulationTime() == 0.0);

    result = environment.Step(commands);
    assert(!result.truncated);

    // Normalized RL actions must advance the environment.
    {
        rl::FlightEnvironment environment;

        environment.Reset(42);

        const rl::Action action{-1.0, -1.0, -1.0, -1.0};

        const auto result = environment.StepAction(action);

        assert(std::abs(
            environment.GetSimulationTime() - (1.0 / 120.0)
        ) < 1e-12);

        assert(!result.terminated);
        assert(!result.truncated);

        const auto observation = environment.GetRLObservation();

        for (double value : observation)
        {
            assert(std::isfinite(value));
        }
    }

    // RL actions must use the same motor commands as the physics API.
    {
        rl::FlightEnvironment rl_environment;
        rl::FlightEnvironment motor_environment;

        rl_environment.Reset(123);
        motor_environment.Reset(123);

        const rl::Action action{-1.0, -0.5, 0.5, 1.0};

        const physics::MotorCommands commands{
            0.0, 0.25, 0.75, 1.0
        };

        rl_environment.StepAction(action);
        motor_environment.Step(commands);

        const auto& rl_state = rl_environment.GetTrueState();
        const auto& motor_state = motor_environment.GetTrueState();

        assert(rl_state.position.isApprox(
            motor_state.position, 1e-12
        ));

        assert(rl_state.velocity.isApprox(
            motor_state.velocity, 1e-12
        ));
    }

    std::cout << "Flight environment tests PASSED\n";
    return 0;
}

