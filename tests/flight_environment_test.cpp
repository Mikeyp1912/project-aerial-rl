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

    // Default action mode must be direct motor control.
    {
        rl::FlightEnvironment environment;

        assert(environment.GetActionMode() ==
               rl::ActionMode::DirectMotor);
    }

    // Explicit direct motor mode must execute normally.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::DirectMotor;

        rl::FlightEnvironment environment(parameters);

        const rl::Action action{-1.0, -1.0, -1.0, -1.0};

        const auto result = environment.StepAction(action);

        assert(!result.terminated);
        assert(!result.truncated);

        assert(environment.GetSimulationTime() > 0.0);
    }

    // Body-rate actions must advance the simulation.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::BodyRate;

        rl::FlightEnvironment environment(parameters);

        const rl::Action action{0.0, 0.0, 0.0, 0.0};

        const auto result = environment.StepAction(action);

        assert(!result.terminated);
        assert(!result.truncated);

        assert(std::abs(
            environment.GetSimulationTime() - parameters.time_step
        ) < 1e-12);

        for (double value : environment.GetRLObservation())
        {
            assert(std::isfinite(value));
        }
    }

    // Reset must restore deterministic body-rate behaviour.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::BodyRate;

        rl::FlightEnvironment environment(parameters);

        const rl::Action action{0.2, -0.1, 0.15, 0.0};

        environment.Reset(123);
        environment.StepAction(action);

        const auto first_position =
            environment.GetTrueState().position.eval();

        const auto first_velocity =
            environment.GetTrueState().velocity.eval();

        environment.Reset(123);
        environment.StepAction(action);

        assert(environment.GetTrueState().position.isApprox(
            first_position, 1e-12
        ));

        assert(environment.GetTrueState().velocity.isApprox(
            first_velocity, 1e-12
        ));
    }

    // PID-target actions must advance the environment.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::PIDTargets;

        rl::FlightEnvironment environment(parameters);

        const rl::Action action{
            0.0,  // X target
            0.0,  // Y target
            0.0,  // Altitude target
            0.0   // Yaw target
        };

        const auto result = environment.StepAction(action);

        assert(!result.terminated);
        assert(!result.truncated);

        assert(std::abs(
            environment.GetSimulationTime() - parameters.time_step
        ) < 1e-12);

        for (double value : environment.GetRLObservation())
        {
            assert(std::isfinite(value));
        }
    }

    // Invalid action modes must be rejected.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = static_cast<rl::ActionMode>(255);

        rl::FlightEnvironment environment(parameters);

        bool threw = false;

        try
        {
            environment.StepAction(rl::Action{});
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
        assert(environment.GetSimulationTime() == 0.0);
    }

    // Resetting with the same seed must reproduce the same trajectory.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::PIDTargets;

        rl::FlightEnvironment environment(parameters);

        const rl::Action action{0.2, -0.1, -0.4, 0.15};

        environment.Reset(123);

        for (int i = 0; i < 60; ++i)
        {
            environment.StepAction(action);
        }

        const Eigen::Vector3d first_position =
            environment.GetTrueState().position;

        const Eigen::Vector3d first_velocity =
            environment.GetTrueState().velocity;

        const Eigen::Quaterniond first_orientation =
            environment.GetTrueState().orientation;

        environment.Reset(123);

        for (int i = 0; i < 60; ++i)
        {
            environment.StepAction(action);
        }

        const auto& state = environment.GetTrueState();

        assert(state.position.isApprox(first_position, 1e-12));
        assert(state.velocity.isApprox(first_velocity, 1e-12));

        assert(state.orientation.coeffs().isApprox(
            first_orientation.coeffs(), 1e-12
        ));
    }

    // Opposite horizontal targets should produce different motion.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::PIDTargets;

        rl::FlightEnvironment positive(parameters);
        rl::FlightEnvironment negative(parameters);

        positive.Reset(42);
        negative.Reset(42);

        const rl::Action positive_action{0.5, 0.0, -0.4, 0.0};
        const rl::Action negative_action{-0.5, 0.0, -0.4, 0.0};

        for (int i = 0; i < 120; ++i)
        {
            positive.StepAction(positive_action);
            negative.StepAction(negative_action);
        }

        const double positive_x =
            positive.GetTrueState().position.x();

        const double negative_x =
            negative.GetTrueState().position.x();

        assert(std::isfinite(positive_x));
        assert(std::isfinite(negative_x));

        assert(std::abs(positive_x - negative_x) > 1e-4);
    }

    // PID-target mode must respect episode truncation.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::PIDTargets;
        parameters.maximum_episode_time = 0.1;

        rl::FlightEnvironment environment(parameters);

        const rl::Action action{0.0, 0.0, -0.5, 0.0};

        rl::FlightStepResult result;

        for (int i = 0; i < 12; ++i)
        {
            result = environment.StepAction(action);
        }

        assert(result.truncated);
        assert(!result.terminated);

        bool threw = false;

        try
        {
            environment.StepAction(action);
        }
        catch (const std::logic_error&)
        {
            threw = true;
        }

        assert(threw);
    }

    std::cout << "Flight environment tests PASSED\n";
    return 0;
}

