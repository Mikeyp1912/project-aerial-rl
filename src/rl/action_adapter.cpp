// /src/rl/action_adapter.cpp

#include "aerial/rl/action_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    physics::MotorCommands DirectMotorActionAdapter::Convert(
            const Action& action) const
    {
        physics::MotorCommands commands{};

        for (std::size_t i = 0; i < kActionSize; ++i) {
            const double value = action[i];

            if (!std::isfinite(value)) {
                throw std::invalid_argument("RL action values must be finite");
            }

            const double normalized = std::clamp(value, -1.0, 1.0);

            commands[i] = 0.5 * (normalized + 1.0);
        }

        return commands;
    }
}
