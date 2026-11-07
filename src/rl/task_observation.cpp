// /src/rl/task_observation.cpp

#include "aerial/rl/task_observation.hpp"

#include <algorithm>
#include <stdexcept>

namespace aerial::rl
{
    std::size_t GetObservationSize(ObservationMode mode)
    {
        switch (mode)
        {
            case ObservationMode::Base:
                return kObservationSize;

            case ObservationMode::HoverAugmented:
                return kHoverObservationSize;
        }

        throw std::invalid_argument("Unknown observation mode");
    }

    HoverObservation MakeHoverObservation(
        const Observation& base_observation,
        const Eigen::Vector3d& target_position,
        const ObservationNormalizationParameters& normalization
    )
    {
        ValidateObservationNormalizationParameters(normalization);

        if (!target_position.allFinite())
        {
            throw std::invalid_argument(
                "Hover target must be finite"
            );
        }

        const Observation normalized_base =
            NormalizeObservation(base_observation, normalization);

        HoverObservation result{};

        std::copy(
            normalized_base.begin(),
            normalized_base.end(),
            result.begin()
        );

        for (int axis = 0; axis < 3; ++axis)
        {
            // Use the original position measurement, not the
            // already normalized or clipped position.
            double error =
                (base_observation[axis] - target_position[axis]) /
                normalization.target_error_scale;

            if (normalization.clip)
            {
                error = std::clamp(error, -1.0, 1.0);
            }

            result[kObservationSize + axis] = error;
        }

        return result;
    }
}
