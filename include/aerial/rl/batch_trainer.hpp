// /include/aerial/rl/batch_trainer.hpp

#pragma once

#include <vector>

#include "aerial/rl/adam_optimizer.hpp"
#include "aerial/rl/neural_network_policy.hpp"

namespace aerial::rl
{
    struct TrainingSample
    {
        Observation observation{};
        Action target{};
    };

    struct BatchTrainingResult
    {
        double mean_loss_before = 0.0;
        double mean_loss_after = 0.0;
        std::size_t sample_count = 0;
    };

    BatchTrainingResult TrainBatchMSE(
        NeuralNetworkPolicy& policy,
        AdamOptimizer& optimizer,
        const std::vector<TrainingSample>& samples
    );
}

