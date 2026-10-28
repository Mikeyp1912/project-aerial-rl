// /include/aerial/rl/rl_types.hpp

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace aerial::rl
{
    enum class ActionMode : std::uint8_t
    {
        DirectMotor,
        BodyRate,
        PIDTargets
    };

    // Initial observation:
    // position (3), velocity (3), orientation quaternion (4),
    // angular velocity (3), accelerometer (3)
    inline constexpr std::size_t kObservationSize = 16;

    using Observation = std::array<double, kObservationSize>;

    // Four policy outputs for all three initial control modes
    // Their meanings depend on ActionMode
    inline constexpr std::size_t kActionSize = 4;

    using Action = std::array<double, kActionSize>;

    struct ActionSpace
    {
        ActionMode mode = ActionMode::DirectMotor;

        // Policy outputs are normalized to [-1, 1]
        // The adapter maps these to physical control commands
        Action minimum{-1.0, -1.0, -1.0, -1.0};
        Action maximum{1.0, 1.0, 1.0, 1.0};
    };
}
