// /include/aerial/rl/neural_network_policy.hpp

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "aerial/rl/policy.hpp"

namespace aerial::rl
{
    class NeuralNetworkPolicy final : public Policy
    {
        public:
            static constexpr int kInputSize = 16;
            static constexpr int kHiddenSize = 64;
            static constexpr int kOutputSize = 4;

            using InputVector = Eigen::Matrix<double, kInputSize, 1>;

            using HiddenVector = Eigen::Matrix<double, kHiddenSize, 1>;

            using OutputVector = Eigen::Matrix<double, kOutputSize, 1>;

            using InputWeights = Eigen::Matrix<double, kHiddenSize, kInputSize>;

            using HiddenWeights = Eigen::Matrix<double, kHiddenSize, kHiddenSize>;

            using OutputWeights = Eigen::Matrix<double, kOutputSize, kHiddenSize>;

            explicit NeuralNetworkPolicy(std::uint32_t seed = 42);

            Action Predict(const Observation& observation) override;

            void Initialize(std::uint32_t seed);

            std::size_t GetParameterCount() const;

            std::vector<double> GetParameters() const;

            void SetParameters(const std::vector<double>& parameters);

            void SaveParameters(const std::string& path) const;

            void LoadParameters(const std::string& path);

            // Mean squared error across the four output actions
            double CalculateMSE(
                    const Observation& observation,
                    const Action& target
                    ) const;

            // Returns gradients in the same order as GetParameters()
            std::vector<double> CalculateMSEGradient(
                    const Observation& observation,
                    const Action& target
                    ) const;

            // Applies one gradient-descent update
            void TrainMSE(
                    const Observation& observation,
                    const Action& target,
                    double learning_rate
                    );

        private:
            static HiddenVector ReLU(const HiddenVector& input);

            InputWeights weights1_;
            HiddenWeights weights2_;
            OutputWeights weights3_;

            HiddenVector bias1_;
            HiddenVector bias2_;
            OutputVector bias3_;
    };
}
