// /src/rl/adam_optimizer.cpp

#include "aerial/rl/adam_optimizer.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace aerial::rl
{
    AdamOptimizer::AdamOptimizer(
            std::size_t parameter_count,
            const AdamParameters& parameters
            )
        :
            settings_(parameters),
            first_moment_(parameter_count, 0.0),
            second_moment_(parameter_count, 0.0)
    {
        if (parameter_count == 0 ||
            !std::isfinite(settings_.learning_rate) ||
            settings_.learning_rate <= 0.0 ||
            !std::isfinite(settings_.beta1) ||
            settings_.beta1 < 0.0 ||
            settings_.beta1 >= 1.0 ||
            !std::isfinite(settings_.beta2) ||
            settings_.beta2 < 0.0 ||
            settings_.beta2 >= 1.0 ||
            !std::isfinite(settings_.epsilon) ||
            settings_.epsilon <= 0.0)
        {
            throw std::invalid_argument(
                "Invalid Adam optimizer configuration"
            );
        }
    }

    void AdamOptimizer::Reset()
    {
        std::fill(
                first_moment_.begin(),
                first_moment_.end(),
                0.0
                );

        std::fill(
                second_moment_.begin(),
                second_moment_.end(),
                0.0
                );

        step_count_ = 0;
    }

    void AdamOptimizer::Step(
            std::vector<double>& parameters,
            const std::vector<double>& gradients
            )
    {
        if (parameters.size() != first_moment_.size() ||
            gradients.size() != first_moment_.size())
        {
            throw std::invalid_argument(
                "Adam parameter/gradient size mismatch"
            );
        }

        if (step_count_ ==
            std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error(
                "Adam step counter overflow"
            );
        }

        for (std::size_t i = 0; i < parameters.size(); ++i)
        {
            if (!std::isfinite(parameters[i]) ||
                !std::isfinite(gradients[i]))
            {
                throw std::invalid_argument(
                    "Adam parameters and gradients must be finite"
                );
            }
        }

        const std::uint64_t next_step = step_count_ + 1;

        const double correction1 =
            1.0 - std::pow(
                    settings_.beta1,
                    static_cast<double>(next_step)
                    );

        const double correction2 =
            1.0 - std::pow(
                    settings_.beta2,
                    static_cast<double>(next_step)
                    );

        // Compute candidates first, so an invalid update
        // cannot partially modify the optimizer
        auto next_parameters = parameters;
        auto next_first = first_moment_;
        auto next_second = second_moment_;

        for (std::size_t i = 0; i < parameters.size(); ++i) {
            const double gradient = gradients[i];

            next_first[i] =
                settings_.beta1 * first_moment_[i] +
                (1.0 - settings_.beta1) * gradient;

            next_second[i] =
                settings_.beta2 * second_moment_[i] +
                (1.0 - settings_.beta2) * gradient * gradient;

            const double corrected_first =
                next_first[i] / correction1;

            const double corrected_second =
                next_second[i] / correction2;

            next_parameters[i] -=
                settings_.learning_rate *
                corrected_first /
                (std::sqrt(corrected_second) +
                 settings_.epsilon);

            if (!std::isfinite(next_parameters[i]) ||
                !std::isfinite(next_first[i]) ||
                !std::isfinite(next_second[i]))
            {
                throw std::runtime_error(
                    "Adam update produced non-finite values"
                );
            }
        }

        parameters.swap(next_parameters);
        first_moment_.swap(next_first);
        second_moment_.swap(next_second);
        step_count_ = next_step;
    }

    std::uint64_t AdamOptimizer::GetStepCount() const
    {
        return step_count_;
    }
}
