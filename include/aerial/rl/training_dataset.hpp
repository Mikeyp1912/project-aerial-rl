// /include/aerial/rl/training_dataset.hpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aerial/rl/batch_trainer.hpp"

namespace aerial::rl
{
    class TrainingDataset
    {
        public:
            void Add(const TrainingSample& sample);

            std::size_t Size() const;
            bool Empty() const;

            const TrainingSample& Get(std::size_t index) const;

            const std::vector<TrainingSample>& GetSamples() const;

        private:
            std::vector<TrainingSample> samples_;
    };

    struct DatasetSplit
    {
        std::vector<std::size_t> training_indices;
        std::vector<std::size_t> validation_indices;
    };

    DatasetSplit SplitDataset(
            const TrainingDataset& dataset,
            double validation_fraction,
            std::uint32_t seed
            );

    std::vector<std::vector<std::size_t>> CreateMiniBatches(
            const std::vector<std::size_t>& indices,
            std::size_t batch_size,
            std::uint32_t seed,
            bool shuffle = true
            );
}
