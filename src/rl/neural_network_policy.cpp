// /src/rl/neural_network_policy.cpp

#include "aerial/rl/neural_network_policy.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>

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

    std::vector<double> NeuralNetworkPolicy::GetParameters() const
    {
        std::vector<double> parameters;
        parameters.reserve(GetParameterCount());

        const auto append = [&parameters](const auto& matrix)
        {
            for (Eigen::Index i = 0; i < matrix.size(); ++i) {
                parameters.push_back(matrix.data()[i]);
            }
        };

        append(weights1_);
        append(weights2_);
        append(weights3_);

        append(bias1_);
        append(bias2_);
        append(bias3_);

        return parameters;
    }

    void NeuralNetworkPolicy::SetParameters(
            const std::vector<double>& parameters
            )
    {
        if (parameters.size() != GetParameterCount()) {
            throw std::invalid_argument("Incorrect neural network parameter count");
        }

        for (const double value : parameters) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Neural network parameters must be finite");
            }
        }

        std::size_t offset = 0;

        const auto read = [&parameters, &offset](auto& matrix)
        {
            for (Eigen::Index i = 0; i < matrix.size(); ++i) {
                matrix.data()[i] = parameters[offset++];
            }
        };

        read(weights1_);
        read(weights2_);
        read(weights3_);

        read(bias1_);
        read(bias2_);
        read(bias3_);
    }

    void NeuralNetworkPolicy::SaveParameters(
            const std::string& path
            ) const
    {
        const std::vector<double> parameters = GetParameters();

        std::ofstream file(path, std::ios::binary | std::ios::trunc);

        if (!file) {
            throw std::runtime_error("Failed to open neural network checkpoint for writing");
        }

        // Format v1:
        // 8-byte magic, uint32 version, uint32 count,
        // followed by count native-endian IEEE-754 doubles.
        constexpr std::array<char, 8> magic{
            'A', 'E', 'R', 'I', 'A', 'L', 'N', 'N'
        };

        constexpr std::uint32_t version = 1;

        const auto count = static_cast<std::uint32_t>(parameters.size());

        file.write(magic.data(), magic.size());

        file.write(
                reinterpret_cast<const char*>(&version),
                sizeof(count)
                );

        file.write(
                reinterpret_cast<const char*>(&count),
                sizeof(count)
                );

        file.write(
                reinterpret_cast<const char*>(parameters.data()),
                static_cast<std::streamsize>(
                    parameters.size() * sizeof(double)
                    )
                );

        if (!file) {
            throw std::runtime_error("Failed to write neural network checkpoint");
        }
    }

    void NeuralNetworkPolicy::LoadParameters(
            const std::string& path
            )
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);

        if (!file) {
            throw std::runtime_error("Failed to open neural network checkpoint");
        }

        const std::streamoff file_size = file.tellg();

        constexpr std::array<char, 8> expected_magic{
            'A', 'E', 'R', 'I', 'A', 'L', 'N', 'N'
        };

        constexpr std::uint32_t expected_version = 1;

        const std::streamoff expected_size =
            static_cast<std::streamoff>(
                    expected_magic.size() +
                    2 * sizeof(std::uint32_t) +
                    GetParameterCount() * sizeof(double)
                    );

        if (file_size != expected_size) {
            throw std::runtime_error("Invalid neural network checkpoint size");
        }

        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic{};
        std::uint32_t version = 0;
        std::uint32_t count = 0;

        file.read(magic.data(), magic.size());

        file.read(
                reinterpret_cast<char*>(&version),
                sizeof(version)
                );

        file.read(
                reinterpret_cast<char*>(&count),
                sizeof(count)
                );

        if (!file ||
            magic != expected_magic ||
            version != expected_version ||
            count != GetParameterCount()) {
            throw std::runtime_error("Invalid or incompatible neural network checkpoint");
        }

        std::vector<double> parameters(count);

        file.read(
                reinterpret_cast<char*>(parameters.data()),
                static_cast<std::streamsize>(
                    parameters.size() * sizeof(double)
                    )
                );

        if (!file) {
            throw std::runtime_error("Failed to read neural network checkpoint");
        }

        // SetParameters validates everything before modifying the model
        SetParameters(parameters);
    }
}
