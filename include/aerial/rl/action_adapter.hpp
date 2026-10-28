// /include/aerial/rl/action_adapter.hpp

#pragma once

#include "aerial/physics/motor_commands.hpp"
#include "aerial/rl/rl_types.hpp"

namespace aerial::rl
{
    class DirectMotorActionAdapter
    {
        public:
            physics::MotorCommands Convert(
                    const Action& action
                    ) const;
    };
}
