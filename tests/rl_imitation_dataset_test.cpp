// /tests/rl_imitation_dataset_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_set>

#include "aerial/rl/imitation_dataset_collector.hpp"

int main()
{
    using namespace aerial::rl;

    FlightEnvironmentParameters parameters;

    parameters.action_mode = ActionMode::PIDTargets;
    parameters.maximum_episode_time = 2.0;

    FlightEnvironment environment(parameters);
    HoverFeedbackPolicy teacher;

    ImitationCollectionParameters collection_parameters;

    collection_parameters.episode_count = 10;
    collection_parameters.base_seed = 42;

    const auto collection = CollectImitationDataset(
        environment,
        teacher,
        collection_parameters
    );

    assert(collection.episodes.size() == 10);
    assert(!collection.dataset.Empty());

    std::size_t total_samples = 0;

    for (const auto& episode : collection.episodes)
    {
        assert(episode.sample_count > 0);
        assert(episode.first_sample == total_samples);
        assert(episode.terminated || episode.truncated);
        assert(std::isfinite(episode.total_reward));

        total_samples += episode.sample_count;
    }

    assert(total_samples == collection.dataset.Size());

    for (const auto& sample :
         collection.dataset.GetSamples())
    {
        for (const double value : sample.observation)
        {
            assert(std::isfinite(value));
        }

        for (const double value : sample.target)
        {
            assert(std::isfinite(value));
            assert(value >= -1.0);
            assert(value <= 1.0);
        }
    }

    const auto split1 = SplitImitationEpisodes(
        collection, 0.2, 123
    );

    const auto split2 = SplitImitationEpisodes(
        collection, 0.2, 123
    );

    assert(split1.training_indices ==
           split2.training_indices);

    assert(split1.validation_indices ==
           split2.validation_indices);

    assert(!split1.training_indices.empty());
    assert(!split1.validation_indices.empty());

    std::unordered_set<std::size_t> training_indices(
        split1.training_indices.begin(),
        split1.training_indices.end()
    );

    std::unordered_set<std::size_t> validation_indices(
        split1.validation_indices.begin(),
        split1.validation_indices.end()
    );

    assert(training_indices.size() ==
           split1.training_indices.size());

    assert(validation_indices.size() ==
           split1.validation_indices.size());

    for (const auto index : validation_indices)
    {
        assert(training_indices.count(index) == 0);
    }

    assert(
        training_indices.size() +
        validation_indices.size() ==
        collection.dataset.Size()
    );

    // Verify that each complete episode belongs to one split.
    for (const auto& episode : collection.episodes)
    {
        const bool in_training =
            training_indices.count(episode.first_sample) != 0;

        const bool in_validation =
            validation_indices.count(episode.first_sample) != 0;

        assert(in_training != in_validation);

        for (std::size_t i = 0;
             i < episode.sample_count;
             ++i)
        {
            const auto index = episode.first_sample + i;

            assert(
                (training_indices.count(index) != 0) ==
                in_training
            );

            assert(
                (validation_indices.count(index) != 0) ==
                in_validation
            );
        }
    }

    std::cout
        << "Collected episodes: "
        << collection.episodes.size() << '\n'
        << "Total samples: "
        << collection.dataset.Size() << '\n'
        << "Training samples: "
        << split1.training_indices.size() << '\n'
        << "Validation samples: "
        << split1.validation_indices.size() << '\n';

    std::cout
        << "RL imitation dataset tests PASSED\n";

    return 0;
}

