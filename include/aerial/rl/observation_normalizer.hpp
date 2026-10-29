// /include/aerial/rl/observation_normalizer.hpp

#pragma once

#include "aerial/rl/rl_types.hpp"

namespace aerial::rl
{
    struct ObservationNormalizationParameters
    {
        double position_scale = 10.0;
        double velocity_scale = 10.0;
        double angular_velocity_scale = 10.0;
        double acceleration_scale = 20.0;

        bool clip = true;
    };

    void ValidateObservationNormalizationParameters(
        const ObservationNormalizationParameters& parameters
        );

    Observation NormalizeObservation(
            const Observation& observation,
            const ObservationNormalizationParameters& parameters = {}
            );
}
