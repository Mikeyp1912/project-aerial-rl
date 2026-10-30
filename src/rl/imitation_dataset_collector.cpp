// /src/rl/imitation_dataset_collector.cpp


#include "aerial/rl/imitation_dataset_collector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace aerial::rl
{
    ImitationCollectionResult CollectImitationDataset(
        FlightEnvironment& environment,
        Policy& teacher,
        const ImitationCollectionParameters& parameters
    )
    {
        if (parameters.episode_count == 0)
        {
            throw std::invalid_argument(
                "Episode count must be positive"
            );
        }

        if (environment.GetActionMode() !=
            ActionMode::PIDTargets)
        {
            throw std::invalid_argument(
                "Teacher collection requires PIDTargets mode"
            );
        }

        if (parameters.episode_count >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max() -
                parameters.base_seed
            ) + 1)
        {
            throw std::invalid_argument(
                "Episode seeds would overflow"
            );
        }

        ImitationCollectionResult result;

        result.episodes.reserve(parameters.episode_count);

        for (std::size_t episode = 0;
             episode < parameters.episode_count;
             ++episode)
        {
            environment.Reset(
                parameters.base_seed +
                static_cast<std::uint32_t>(episode)
            );

            teacher.Reset();

            CollectedEpisode record;
            record.first_sample = result.dataset.Size();

            while (true)
            {
                // Observe before taking the action.
                const Observation observation =
                    environment.GetNormalizedRLObservation();

                const Action action =
                    teacher.Predict(observation);

                TrainingSample sample;
                sample.observation = observation;
                sample.target = action;

                result.dataset.Add(sample);

                const auto step =
                    environment.StepAction(action);

                ++record.sample_count;
                record.total_reward += step.reward;

                if (step.terminated || step.truncated)
                {
                    record.terminated = step.terminated;
                    record.truncated = step.truncated;
                    break;
                }
            }

            result.episodes.push_back(record);
        }

        return result;
    }

    DatasetSplit SplitImitationEpisodes(
        const ImitationCollectionResult& collection,
        double validation_fraction,
        std::uint32_t seed
    )
    {
        if (collection.episodes.size() < 2)
        {
            throw std::invalid_argument(
                "At least two episodes are required"
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

        std::vector<std::size_t> episode_indices(
            collection.episodes.size()
        );

        std::iota(
            episode_indices.begin(),
            episode_indices.end(),
            0
        );

        std::mt19937 generator(seed);

        std::shuffle(
            episode_indices.begin(),
            episode_indices.end(),
            generator
        );

        const std::size_t validation_count =
            std::clamp(
                static_cast<std::size_t>(
                    episode_indices.size() *
                    validation_fraction
                ),
                std::size_t{1},
                episode_indices.size() - 1
            );

        DatasetSplit split;

        for (std::size_t i = 0;
             i < episode_indices.size();
             ++i)
        {
            const auto& episode =
                collection.episodes[episode_indices[i]];

            auto& destination =
                i < validation_count
                    ? split.validation_indices
                    : split.training_indices;

            for (std::size_t sample = 0;
                 sample < episode.sample_count;
                 ++sample)
            {
                destination.push_back(
                    episode.first_sample + sample
                );
            }
        }

        return split;
    }
}

