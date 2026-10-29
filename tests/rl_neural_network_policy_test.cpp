// /tests/rl_neural_network_policy_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/rl/neural_network_policy.hpp"
#include "aerial/rl/policy_runner.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-12;

    // Verify the network architecture.
    {
        rl::NeuralNetworkPolicy policy;

        assert(policy.GetParameterCount() == 5508);
    }

    // Two networks initialized with the same seed
    // must produce identical predictions.
    {
        rl::NeuralNetworkPolicy first(123);
        rl::NeuralNetworkPolicy second(123);

        rl::Observation observation{};

        for (std::size_t i = 0; i < observation.size(); ++i)
        {
            observation[i] =
                static_cast<double>(i) / observation.size();
        }

        const auto action1 = first.Predict(observation);
        const auto action2 = second.Predict(observation);

        for (std::size_t i = 0; i < action1.size(); ++i)
        {
            assert(std::abs(action1[i] - action2[i]) < tolerance);
        }
    }

    // Every output must be finite and normalized.
    {
        rl::NeuralNetworkPolicy policy(42);

        rl::Observation observation{};

        for (std::size_t i = 0; i < observation.size(); ++i)
        {
            observation[i] =
                std::sin(static_cast<double>(i));
        }

        const auto action = policy.Predict(observation);

        for (double value : action)
        {
            assert(std::isfinite(value));
            assert(value >= -1.0);
            assert(value <= 1.0);
        }
    }

    // Reinitializing with the same seed restores
    // the original network parameters.
    {
        rl::NeuralNetworkPolicy policy(77);

        rl::Observation observation{};
        observation[0] = 0.5;
        observation[6] = 1.0;

        const auto original = policy.Predict(observation);

        policy.Initialize(100);
        policy.Initialize(77);

        const auto restored = policy.Predict(observation);

        for (std::size_t i = 0; i < original.size(); ++i)
        {
            assert(std::abs(original[i] - restored[i]) < tolerance);
        }
    }

    // Non-finite observations must be rejected.
    {
        rl::NeuralNetworkPolicy policy;

        rl::Observation observation{};
        observation[3] =
            std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try
        {
            policy.Predict(observation);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Verify integration with the flight environment.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::PIDTargets;

        rl::FlightEnvironment environment(parameters);
        rl::NeuralNetworkPolicy policy(42);

        rl::PolicyRunner runner(environment, policy);

        runner.Reset(123);

        const auto result = runner.Step();

        assert(std::isfinite(result.reward));

        for (double value : runner.GetLastAction())
        {
            assert(std::isfinite(value));
            assert(value >= -1.0);
            assert(value <= 1.0);
        }

        assert(std::abs(
            environment.GetSimulationTime() - parameters.time_step
        ) < tolerance);
    }

    std::cout << "RL neural network policy tests PASSED\n";

    return 0;
}

