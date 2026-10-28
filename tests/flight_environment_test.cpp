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

    std::cout << "Flight environment tests PASSED\n";
    return 0;
}

