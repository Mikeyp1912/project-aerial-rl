// /tests/rl_training_dataset_test.cpp


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "aerial/rl/epoch_trainer.hpp"

int main()
{
    using namespace aerial::rl;

    TrainingDataset dataset;

    for (int i = 0; i < 100; ++i)
    {
        TrainingSample sample;

        sample.observation[0] = i / 100.0;
        sample.observation[1] = -i / 200.0;
        sample.observation[2] = 0.2;
        sample.observation[6] = 1.0;

        sample.target = {
            0.2,
            -0.1,
            0.3,
            0.0
        };

        dataset.Add(sample);
    }

    assert(dataset.Size() == 100);
    assert(!dataset.Empty());

    // Deterministic splitting.
    const auto split1 = SplitDataset(dataset, 0.2, 42);
    const auto split2 = SplitDataset(dataset, 0.2, 42);

    assert(split1.training_indices == split2.training_indices);
    assert(split1.validation_indices == split2.validation_indices);

    assert(split1.training_indices.size() == 80);
    assert(split1.validation_indices.size() == 20);

    // Every sample appears in exactly one subset.
    {
        std::unordered_set<std::size_t> indices;

        for (const auto index : split1.training_indices)
        {
            assert(indices.insert(index).second);
        }

        for (const auto index : split1.validation_indices)
        {
            assert(indices.insert(index).second);
        }

        assert(indices.size() == dataset.Size());
    }

    // Deterministic mini-batch sampling.
    {
        const auto batches1 = CreateMiniBatches(
            split1.training_indices, 16, 123
        );

        const auto batches2 = CreateMiniBatches(
            split1.training_indices, 16, 123
        );

        assert(batches1 == batches2);
        assert(batches1.size() == 5);

        std::unordered_set<std::size_t> seen;

        for (const auto& batch : batches1)
        {
            assert(batch.size() == 16);

            for (const auto index : batch)
            {
                assert(seen.insert(index).second);
            }
        }

        assert(seen.size() == 80);
    }

    // A non-divisible batch size retains the remainder.
    {
        const auto batches = CreateMiniBatches(
            split1.training_indices, 30, 42
        );

        assert(batches.size() == 3);
        assert(batches[0].size() == 30);
        assert(batches[1].size() == 30);
        assert(batches[2].size() == 20);
    }

    // Training should improve both training and validation MSE
    // for this simple, shared target mapping.
    {
        NeuralNetworkPolicy policy(42);
        AdamOptimizer optimizer(policy.GetParameterCount());

        const double initial_training_loss =
            EvaluateDatasetMSE(
                policy, dataset, split1.training_indices
            );

        const double initial_validation_loss =
            EvaluateDatasetMSE(
                policy, dataset, split1.validation_indices
            );

        EpochTrainingResult result;

        for (std::uint32_t epoch = 0; epoch < 50; ++epoch)
        {
            result = TrainEpoch(
                policy,
                optimizer,
                dataset,
                split1,
                16,
                1000 + epoch
            );
        }

        std::cout
            << "Initial training MSE: "
            << initial_training_loss << '\n'
            << "Final training MSE:   "
            << result.training_loss << '\n'
            << "Initial validation MSE: "
            << initial_validation_loss << '\n'
            << "Final validation MSE:   "
            << result.validation_loss << '\n';

        assert(result.training_samples == 80);
        assert(result.validation_samples == 20);
        assert(result.batch_count == 5);

        assert(result.training_loss <
               initial_training_loss * 0.1);

        assert(result.validation_loss <
               initial_validation_loss * 0.1);

        assert(optimizer.GetStepCount() == 250);
    }

    // Reject invalid dataset values.
    {
        TrainingSample invalid;

        invalid.observation[0] =
            std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try
        {
            dataset.Add(invalid);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
        assert(dataset.Size() == 100);
    }

    // Reject invalid batch sizes.
    {
        bool threw = false;

        try
        {
            CreateMiniBatches(
                split1.training_indices, 0, 42
            );
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Reject overlapping training and validation subsets.
    {
        NeuralNetworkPolicy policy;
        AdamOptimizer optimizer(policy.GetParameterCount());

        DatasetSplit invalid_split;
        invalid_split.training_indices = {0, 1};
        invalid_split.validation_indices = {1, 2};

        bool threw = false;

        try
        {
            TrainEpoch(
                policy,
                optimizer,
                dataset,
                invalid_split,
                2,
                42
            );
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
        assert(optimizer.GetStepCount() == 0);
    }

    std::cout << "RL training dataset tests PASSED\n";

    return 0;
}

