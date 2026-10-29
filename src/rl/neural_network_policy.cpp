// /src/rl/neural_network_policy.cpp

#include "aerial/rl/neural_network_policy.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace aerial::rl
{
    static_assert(
            NeuralNetworkPolicy::kInputSize ==
            static_cast<int>(kObservationSize)
            );

    static_assert(
            NeuralNetworkPolicy::kOuputSize ==
            static_cast<int>(kActionSize)
            );

    NeuralNetworkPolicy::NeuralNetworkPolicy(std::uint32_t seed)
    {
        Initialize(seed);
    }

    void NeuralNetworkPolicy::Initialize(std::uint32_t seed)
    {
        std::mt19937 generator(seed);
        std::normal_distribution<double> distribution(0.0, 1.0);

        // He initialization for ReLU layers
        const double scale1 = std::sqrt(2.0 / static_cast<double>(kInputSize));

        const double scale2 = std::sqrt(2.0 / static_cast<double>(kHiddenSize));

        // Smaller initial output weights help avoid tanh saturation
        const double scale3 = std::sqrt(1.0 / static_cast<double>(kHiddenSize));

        for (int row = 0; row < weights1_.rows(); ++row) {
            for (int col = 0; col < weights1_.cols(); ++col) {
                weights1_(row, col) = distribution(generator) * scale1;
            }
        }

        for (int row = 0; row < weights2_.rows(); ++row) {
            for (int col = 0; col < weights2_.cols(); ++col) {
                weights2_(row, col) = distribution(generator) * scale2;
            }
        }

        for (int row = 0; row < weights3_.rows(); ++row) {
            for (int col = 0; col < weights3_.cols(); ++col) {
                weights3_(row, col) = distribution(generator) * scale3;
            }
        }

        bias1_.setZero();
        bias2_.setZero();
        bias3_.setZero();
    }

    NeuralNetworkPolicy::HiddenVector NeuralNetworkPolicy::ReLU(const HiddenVector& input)
    {
        return input.cwiseMax(0.0);
    }

    Action NeuralNetworkPolicy::Predict(
            const Observation& observation
            )
    {
        InputVector input;

        for (int i = 0; i < kInputSize; ++i) {
            const double value = observation[static_cast<std::size_t>(i)];

            if (!std::isfinite(value)) {
                throw std::invalid_argument("Neural network observation must be finite");
            }

            input(i) = value;
        }

        const HiddenVector hidden1 = ReLU((weights1_ * input + bias1_).eval());

        const HiddenVector hidden2 = ReLU((weights2_ * hidden1 + bias2_).eval());

        const OutputVector output =
            (weights3_ * hidden2 + bias3_)
            .array()
            .tanh()
            .matrix();

        Action action{};

        for (int i = 0; i < kOuputSize; ++i) {
            action[static_cast<std::size_t>(i)] = output(i);
        }

        return ValidatePolicyAction(action);
    }

    std::size_t NeuralNetworkPolicy::GetParameterCount() const
    {
        return static_cast<std::size_t>(
                weights1_.size() +
                weights2_.size() +
                weights3_.size() +
                bias1_.size() +
                bias2_.size() +
                bias3_.size()
                );
    }
}
