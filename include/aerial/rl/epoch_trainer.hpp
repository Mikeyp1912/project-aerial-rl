// /include/aerial/rl/epoch_trainer.hpp

#pragma once

#include <cstddef>
#include <cstdint>

#include "aerial/rl/training_dataset.hpp"

namespace aerial::rl
{
    struct EpochTrainingResult
    {
        double training_loss = 0.0;
        double validation_loss = 0.0;

        std::size_t training_samples = 0;
        std::size_t validation_samples = 0;
        std::size_t batch_count = 0;
    };

    double EvaluateDatasetMSE(
            const NeuralNetworkPolicy& policy,
            const TrainingDataset& dataset,
            const std::vector<std::size_t>& indices
            );

    EpochTrainingResult TrainEpoch(
            NeuralNetworkPolicy& policy,
            AdamOptimizer& optimizer,
            const TrainingDataset& dataset,
            const DatasetSplit& split,
            std::size_t batch_size,
            std::uint32_t seed
            );
}
