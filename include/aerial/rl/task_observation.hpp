// /include/aerial/rl/task_observation.hpp

#pragma once

#include <array>
#include <cstddef>

#include <Eigen/Core>

#include "aerial/rl/rl_types.hpp"
#include "aerial/rl/observation_normalizer.hpp"

namespace aerial::rl
{
    enum class ObservationMode
    {
        Base,
        HoverAugmented
    };

    inline constexpr std::size_t kHoverObservationSize =
        kObservationSize + 3;

    using HoverObservation =
        std::array<double, kHoverObservationSize>;

    std::size_t GetObservationSize(ObservationMode mode);

    HoverObservation MakeHoverObservation(
        const Observation& base_observation,
        const Eigen::Vector3d& target_position,
        const ObservationNormalizationParameters& normalization
    );
}
