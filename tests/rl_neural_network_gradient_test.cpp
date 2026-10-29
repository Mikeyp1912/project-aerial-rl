// /tests/rl_neural_network_gradient_test.cpp


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "aerial/rl/neural_network_policy.hpp"

int main()
{
    using namespace aerial;

    // Fixed input and target for reproducible tests.
    rl::Observation observation{};
    observation[0] = 0.25;
    observation[1] = -0.15;
    observation[2] = 0.35;
    observation[6] = 1.0;
    observation[10] = 0.1;

    const rl::Action target{
        0.2, -0.3, 0.4, -0.1
    };

    // Analytical gradients should agree with numerical gradients.
    {
        rl::NeuralNetworkPolicy policy(42);

        const auto original = policy.GetParameters();
        const auto analytical =
            policy.CalculateMSEGradient(observation, target);

        assert(analytical.size() == original.size());

        constexpr double epsilon = 1e-5;
        constexpr double absolute_tolerance = 1e-6;
        constexpr double relative_tolerance = 1e-4;

        // Check a reproducible spread of parameters, including
        // weights and biases from all three layers.
        const std::vector<std::size_t> indices{
            0, 17, 511, 1023,
            1024, 2048, 4095, 5119,
            5120, 5200, 5375,
            5376, 5400, 5439,
            5440, 5480, 5503,
            5504, 5505, 5506, 5507
        };

        for (const std::size_t index : indices)
        {
            auto modified = original;

            modified[index] += epsilon;
            policy.SetParameters(modified);

            const double loss_plus =
                policy.CalculateMSE(observation, target);

            modified[index] -= 2.0 * epsilon;
            policy.SetParameters(modified);

            const double loss_minus =
                policy.CalculateMSE(observation, target);

            const double numerical =
                (loss_plus - loss_minus) / (2.0 * epsilon);

            const double difference =
                std::abs(numerical - analytical[index]);

            const double tolerance =
                absolute_tolerance +
                relative_tolerance *
                std::max(
                    std::abs(numerical),
                    std::abs(analytical[index])
                );

            if (difference > tolerance)
            {
                std::cerr
                    << "Gradient mismatch at index " << index
                    << "\n  Numerical: " << numerical
                    << "\n  Analytical: " << analytical[index]
                    << "\n  Difference: " << difference
                    << '\n';
            }

            assert(difference <= tolerance);
        }

        policy.SetParameters(original);
    }

    // A small gradient-descent step should reduce the loss.
    {
        rl::NeuralNetworkPolicy policy(42);

        const double before =
            policy.CalculateMSE(observation, target);

        policy.TrainMSE(observation, target, 0.001);

        const double after =
            policy.CalculateMSE(observation, target);

        assert(after < before);
    }

    // Repeated training should substantially reduce the loss.
    {
        rl::NeuralNetworkPolicy policy(42);

        const double initial_loss =
            policy.CalculateMSE(observation, target);

        for (int step = 0; step < 500; ++step)
        {
            policy.TrainMSE(observation, target, 0.01);
        }

        const double final_loss =
            policy.CalculateMSE(observation, target);

        std::cout
            << "Initial loss: " << initial_loss << '\n'
            << "Final loss:   " << final_loss << '\n';

        assert(final_loss < initial_loss * 0.1);
    }

    // Reject invalid learning rates.
    {
        rl::NeuralNetworkPolicy policy;

        bool threw = false;

        try
        {
            policy.TrainMSE(observation, target, -0.01);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Reject non-finite target actions.
    {
        rl::NeuralNetworkPolicy policy;

        auto invalid_target = target;
        invalid_target[0] =
            std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try
        {
            policy.CalculateMSEGradient(
                observation,
                invalid_target
            );
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    std::cout
        << "RL neural network gradient tests PASSED\n";

    return 0;
}

