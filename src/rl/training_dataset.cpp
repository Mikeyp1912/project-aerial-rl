// src/rl/training_dataset.cpp

#include "aerial/rl/training_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace aerial::rl
{
    void TrainingDataset::Add(const TrainingSample& sample)
    {
        for (const double value : sample.observation) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Dataset observation must be finite");
            }
        }

        for (const double value : sample.target) {
            if (!std::isfinite(value) ||
                    value < -1.0 || value > 1.0) {
                throw std::invalid_argument("Dataset target must be in [-1, 1]");
            }
        }

        samples_.push_back(sample);
    }

    std::size_t TrainingDataset::Size() const
    {
        return samples_.size();
    }

    bool TrainingDataset::Empty() const
    {
        return samples_.empty();
    }

    const std::vector<TrainingSample>& TrainingDataset::GetSamples() const
    {
        return samples_;
    }

    const TrainingSample& TrainingDataset::Get(std::size_t index) const
    {
        return samples_.at(index);
    }

    DatasetSplit SplitDataset(
        const TrainingDataset& dataset,
        double validation_fraction,
        std::uint32_t seed
    )
    {
        if (dataset.Size() < 2)
        {
            throw std::invalid_argument(
                "Dataset must contain at least two samples"
            );
        }

        if (!std::isfinite(validation_fraction) ||
            validation_fraction <= 0.0 ||
            validation_fraction >= 1.0)
        {
            throw std::invalid_argument(
                "Validation fraction must be in (0, 1)"
            );
        }

        std::vector<std::size_t> indices(dataset.Size());

        std::iota(indices.begin(), indices.end(), 0);

        // Fresh generator initialized with the supplied seed.
        std::mt19937 generator(seed);

        std::shuffle(
            indices.begin(),
            indices.end(),
            generator
        );

        const std::size_t validation_count =
            std::clamp(
                static_cast<std::size_t>(
                    dataset.Size() * validation_fraction
                ),
                std::size_t{1},
                dataset.Size() - 1
            );

        DatasetSplit split;

        split.validation_indices.assign(
            indices.begin(),
            indices.begin() + validation_count
        );

        split.training_indices.assign(
            indices.begin() + validation_count,
            indices.end()
        );

        return split;
    }
    std::vector<std::vector<std::size_t>> CreateMiniBatches(
            const std::vector<std::size_t>& indices,
            std::size_t batch_size,
            std::uint32_t seed,
            bool shuffle
            )
    {
        if (batch_size == 0) {
            throw std::invalid_argument(
                "Batch size must be positive"
            );
        }

        // A repeated index would train on a sample more than once.
        const std::unordered_set<std::size_t> unique(
            indices.begin(), indices.end()
        );

        if (unique.size() != indices.size()) {
            throw std::invalid_argument(
                "Mini-batch indices must be unique"
            );
        }

        auto ordered = indices;

        if (shuffle) {
            std::mt19937 generator(seed);
            std::shuffle(
                    ordered.begin(),
                    ordered.end(),
                    generator
                    );
        }

        std::vector<std::vector<std::size_t>> batches;

        for (std::size_t start = 0; start < ordered.size(); start += std::min(batch_size, ordered.size() - start)) {
            const std::size_t end =
                start + std::min(
                        batch_size,
                        ordered.size() - start
                        );

            batches.emplace_back(
                    ordered.begin() + start,
                    ordered.begin() + end
                    );
        }
        
        return batches;
    }
}
