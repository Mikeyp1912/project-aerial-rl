

#pragma once

#include <cstdint>

#include <Eigen/Core>

namespace aerial::rl
{
    enum class FlightTask : std::uint8_t
    {
        Hover,
        TargetTracking
    };

    struct TargetTrackingParameters
    {
       
        // Randomize the target once at the beginning of each episode.
        bool randomize_target = false;

        // Maximum offset from the configured reference position.
        Eigen::Vector3d maximum_target_offset{
            2.0, 2.0, 1.0
        };
    };
}
