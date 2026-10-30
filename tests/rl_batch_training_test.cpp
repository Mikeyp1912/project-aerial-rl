// /tests/rl_batch_training_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "aerial/rl/adam_optimizer.hpp"
#include "aerial/rl/batch_trainer.hpp"

int main()
{
    using namespace aerial::rl;

    // Test 1: Adam produces the expected first update.
    {
        AdamOptimizer optimizer(2);

        std::vector<double> parameters{1.0, -1.0};
        const std::vector<double> gradients{0.5, -0.5};

        optimizer.Step(parameters, gradients);

        assert(std::abs(parameters[0] - 0.999) < 1e-8);
        assert(std::abs(parameters[1] + 0.999) < 1e-8);
        assert(optimizer.GetStepCount() == 1);

        optimizer.Reset();
        assert(optimizer.GetStepCount() == 0);
    }

    // Test 2: Invalid gradients must not change parameters.
    {
        AdamOptimizer optimizer(2);

        std::vector<double> parameters{1.0, 2.0};
        const auto original = parameters;

        const std::vector<double> gradients{
            0.1,
            std::numeric_limits<double>::quiet_NaN()
        };

        bool threw = false;

        try
        {
            optimizer.Step(parameters, gradients);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
        assert(parameters == original);
        assert(optimizer.GetStepCount() == 0);
    }

    // Test 3: Mini-batch training should reduce MSE.
    {
        NeuralNetworkPolicy policy(42);
        AdamOptimizer optimizer(policy.GetParameterCount());

        std::vector<TrainingSample> samples;

        for (int i = 0; i < 16; ++i)
        {
            TrainingSample sample;

            sample.observation[0] = 0.05 * i;
            sample.observation[1] = -0.03 * i;
            sample.observation[2] = 0.2;
            sample.observation[6] = 1.0;

            sample.target = {
                0.2,
                -0.1,
                0.3,
                0.0
            };

            samples.push_back(sample);
        }

        double initial_loss = 0.0;

        for (const auto& sample : samples)
        {
            initial_loss += policy.CalculateMSE(
                sample.observation,
                sample.target
            );
        }

        initial_loss /= static_cast<double>(samples.size());

        BatchTrainingResult result;

        for (int epoch = 0; epoch < 200; ++epoch)
        {
            result = TrainBatchMSE(
                policy,
                optimizer,
                samples
            );
        }

        std::cout
            << "Initial batch loss: " << initial_loss << '\n'
            << "Final batch loss:   "
            << result.mean_loss_after << '\n';

        assert(result.sample_count == samples.size());
        assert(result.mean_loss_after < initial_loss * 0.1);
        assert(optimizer.GetStepCount() == 200);
    }

    // Test 4: Empty batches must be rejected.
    {
        NeuralNetworkPolicy policy;
        AdamOptimizer optimizer(policy.GetParameterCount());

        bool threw = false;

        try
        {
            TrainBatchMSE(policy, optimizer, {});
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
        assert(optimizer.GetStepCount() == 0);
    }

    std::cout << "RL batch training tests PASSED\n";
    return 0;
}

