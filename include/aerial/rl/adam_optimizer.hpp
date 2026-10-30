// /include/aerial/rl/adam_optimizer.hpp

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aerial::rl
{
    struct AdamParameters
    {
        double learning_rate = 0.001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
    };

    class AdamOptimizer
    {
        public:
            explicit AdamOptimizer(
                    std::size_t parameter_count,
                    const AdamParameters& parameters = {}
                    );

            void Reset();

            void Step(
                    std::vector<double>& parameters,
                    const std::vector<double>& gradients
                    );

            std::uint64_t GetStepCount() const;

        private:
            AdamParameters settings_;
            std::vector<double> first_moment_;
            std::vector<double> second_moment_;
            std::uint64_t step_count_ = 0;
    };
}
