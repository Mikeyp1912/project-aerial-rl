// /include/aerial/rl/imitation_dataset_collector.hpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aerial/rl/flight_environment.hpp"
#include "aerial/rl/policy.hpp"
#include "aerial/rl/training_dataset.hpp"

namespace aerial::rl
{
    struct ImitationCollectionParameters
    {
        std::size_t episode_count = 20;
        std::uint32_t base_seed = 42;
    };

    struct CollectedEpisode
    {
        std::size_t first_sample = 0;
        std::size_t sample_count = 0;

        double total_reward = 0.0;

        bool terminated = false;
        bool truncated = false;
    };

    struct ImitationCollectionResult
    {
        TrainingDataset dataset;
        std::vector<CollectedEpisode> episodes;
    };

    ImitationCollectionResult CollectImitationDataset(
            FlightEnvironment& environment,
            Policy& teacher,
            const ImitationCollectionParameters& parameters = {}
            );

    DatasetSplit SplitImitationEpisodes(
            const ImitationCollectionResult& collection,
            double validation_fraction,
            std::uint32_t seed
            );
}
