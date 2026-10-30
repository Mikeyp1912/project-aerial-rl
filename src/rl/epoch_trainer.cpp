// /src/rl/epoch_trainer.cpp

#include "aerial/rl/epoch_trainer.hpp"

#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace aerial::rl
{
    double EvaluateDatasetMSE(
            const NeuralNetworkPolicy& policy,
            const TrainingDataset& dataset,
            const std::vector<std::size_t>& indices
            )
    {
        if (indices.empty()) {
            throw std::invalid_argument("Cannot evaluate an empty subset");
        }

        double total_loss = 0.0;

        for (const std::size_t index : indices) {
            const auto& sample = dataset.Get(index);

            total_loss += policy.CalculateMSE(
                    sample.observation,
                    sample.target
                    );
        }

        return total_loss / static_cast<double>(indices.size());
    }

    EpochTrainingResult TrainEpoch(
            NeuralNetworkPolicy& policy,
            AdamOptimizer& optimizer,
            const TrainingDataset& dataset,
            const DatasetSplit& split,
            std::size_t batch_size,
            std::uint32_t seed
            )
    {
        if (split.training_indices.empty() ||
            split.validation_indices.empty()) {
            throw std::invalid_argument("Training and validation subsets must be non-empty");
        }

        // Validate the split before modifying the network
        std::unordered_set<std::size_t> seen;

        for (const auto index : split.training_indices) {
            dataset.Get(index);

            if (!seen.insert(index).second) {
                throw std::invalid_argument("Duplicate training index");
            }
        }

        for (const auto index : split.validation_indices) {
            dataset.Get(index);

            if (!seen.insert(index).second) {
                throw std::invalid_argument("Training/validation overlap or duplicate");
            }
        }

        const auto batches = CreateMiniBatches(
                split.training_indices,
                batch_size,
                seed,
                true
                );

        EpochTrainingResult result;

        result.training_samples = split.training_indices.size();

        result.validation_samples = split.validation_indices.size();

        result.batch_count = batches.size();

        for (const auto& batch_indices : batches) {
            std::vector<TrainingSample> batch;
            batch.reserve(batch_indices.size());

            for (const auto index : batch_indices) {
                batch.push_back(dataset.Get(index));
            }

            TrainBatchMSE(policy, optimizer, batch);
        }

        // Both metrics are evaluated after the epoch using
        // the same final network parameters
        result.training_loss = EvaluateDatasetMSE(
                policy,
                dataset,
                split.training_indices
                );

        result.validation_loss = EvaluateDatasetMSE(
                policy,
                dataset,
                split.validation_indices
                );

        return result;
    }
}
