// /src/rl/batch_trainer.cpp


#include "aerial/rl/batch_trainer.hpp"

#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    BatchTrainingResult TrainBatchMSE(
        NeuralNetworkPolicy& policy,
        AdamOptimizer& optimizer,
        const std::vector<TrainingSample>& samples
    )
    {
        if (samples.empty())
        {
            throw std::invalid_argument(
                "Training batch must not be empty"
            );
        }

        auto parameters = policy.GetParameters();

        std::vector<double> mean_gradient(
            parameters.size(),
            0.0
        );

        BatchTrainingResult result;
        result.sample_count = samples.size();

        // Calculate gradients at the same parameter values.
        for (const auto& sample : samples)
        {
            result.mean_loss_before += policy.CalculateMSE(
                sample.observation,
                sample.target
            );

            const auto gradient = policy.CalculateMSEGradient(
                sample.observation,
                sample.target
            );

            for (std::size_t i = 0; i < gradient.size(); ++i)
            {
                mean_gradient[i] += gradient[i];
            }
        }

        const double inverse_batch_size =
            1.0 / static_cast<double>(samples.size());

        result.mean_loss_before *= inverse_batch_size;

        for (double& gradient : mean_gradient)
        {
            gradient *= inverse_batch_size;

            if (!std::isfinite(gradient))
            {
                throw std::runtime_error(
                    "Non-finite batch gradient"
                );
            }
        }

        optimizer.Step(parameters, mean_gradient);
        policy.SetParameters(parameters);

        for (const auto& sample : samples)
        {
            result.mean_loss_after += policy.CalculateMSE(
                sample.observation,
                sample.target
            );
        }

        result.mean_loss_after *= inverse_batch_size;

        return result;
    }
}
