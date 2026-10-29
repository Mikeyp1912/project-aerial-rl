// /src/rl/observation_normalizer.cpp

#include "aerial/rl/observation_normalizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    void ValidateObservationNormalizationParameters(
            const ObservationNormalizationParameters& parameters
            )
    {
        const double scales[] = {
            parameters.position_scale,
            parameters.velocity_scale,
            parameters.angular_velocity_scale,
            parameters.acceleration_scale
        };

        for (const double scale : scales) {
            if (!std::isfinite(scale) || scale <= 0.0) {
                throw std::invalid_argument("Observation normalization scales must be finite and positive");
            }
        }
    }

    Observation NormalizeObservation(
            const Observation& observation,
            const ObservationNormalizationParameters& parameters
            )
    {
        ValidateObservationNormalizationParameters(parameters);

        Observation normalized{};

        for (std::size_t i = 0; i < observation.size(); ++i) {
            if (!std::isfinite(observation[i])) {
                throw std::invalid_argument("Observation contains a non-finite value");
            }

            double scale = 1.0;

            if (i < 3) {
                scale = parameters.position_scale;
            }
            else if (i < 6) {
                scale = parameters.velocity_scale;
            }
            else if (i < 10) {
                scale = 1.0;
            }
            else if (i < 13) {
                scale = parameters.angular_velocity_scale;
            }
            else {
                scale = parameters.acceleration_scale;
            }

            double value = observation[i] / scale;

            if (parameters.clip) {
                value = std::clamp(value, -1.0, 1.0);
            }

            normalized[i] = value;
        }

        return normalized;
    }
}
