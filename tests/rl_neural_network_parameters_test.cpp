// /tests/rl_neural_network_parameters_test.cpp


#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "aerial/rl/neural_network_policy.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-12;

    // Parameter export must contain all network parameters.
    {
        rl::NeuralNetworkPolicy policy(42);

        const auto parameters = policy.GetParameters();

        assert(parameters.size() == 5508);

        for (const double value : parameters)
        {
            assert(std::isfinite(value));
        }
    }

    // Export/import must reproduce identical inference.
    {
        rl::NeuralNetworkPolicy original(123);
        rl::NeuralNetworkPolicy restored(456);

        rl::Observation observation{};
        observation[0] = 0.3;
        observation[2] = 0.2;
        observation[6] = 1.0;

        restored.SetParameters(original.GetParameters());

        const auto expected = original.Predict(observation);
        const auto actual = restored.Predict(observation);

        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            assert(std::abs(expected[i] - actual[i]) < tolerance);
        }
    }

    // Updating a parameter should change network storage.
    {
        rl::NeuralNetworkPolicy policy(42);

        auto parameters = policy.GetParameters();
        parameters[0] += 0.25;

        policy.SetParameters(parameters);

        const auto updated = policy.GetParameters();

        assert(std::abs(updated[0] - parameters[0]) < tolerance);
    }

    // Incorrect parameter count must be rejected.
    {
        rl::NeuralNetworkPolicy policy;

        bool threw = false;

        try
        {
            policy.SetParameters(std::vector<double>(100));
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Non-finite parameters must be rejected without
    // modifying the original network.
    {
        rl::NeuralNetworkPolicy policy;

        const auto original = policy.GetParameters();

        auto invalid = original;
        invalid[10] = std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try
        {
            policy.SetParameters(invalid);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
        assert(policy.GetParameters() == original);
    }

    // Binary checkpoint round-trip.
    {
        const char* path = "rl_nn_parameters_test.bin";

        rl::NeuralNetworkPolicy original(123);
        rl::NeuralNetworkPolicy restored(456);

        original.SaveParameters(path);
        restored.LoadParameters(path);

        assert(
            original.GetParameters() ==
            restored.GetParameters()
        );

        std::remove(path);
    }

    // Loading a missing checkpoint must fail.
    {
        rl::NeuralNetworkPolicy policy;

        bool threw = false;

        try
        {
            policy.LoadParameters(
                "nonexistent_rl_nn_checkpoint.bin"
            );
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }

        assert(threw);
    }

    std::cout
        << "RL neural network parameter tests PASSED\n";

    return 0;
}

