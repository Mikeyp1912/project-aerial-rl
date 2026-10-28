// /tests/rl_action_adapter_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/rl/action_adapter.hpp"

int main()
{
    using namespace aerial;

    rl::DirectMotorActionAdapter adapter;

    // Minimum action maps to stopped motors.
    {
        const rl::Action action{-1.0, -1.0, -1.0, -1.0};
        const auto commands = adapter.Convert(action);

        for (double command : commands)
        {
            assert(std::abs(command) < 1e-12);
        }
    }

    // Zero action maps to half command.
    {
        const rl::Action action{0.0, 0.0, 0.0, 0.0};
        const auto commands = adapter.Convert(action);

        for (double command : commands)
        {
            assert(std::abs(command - 0.5) < 1e-12);
        }
    }

    // Maximum action maps to full command.
    {
        const rl::Action action{1.0, 1.0, 1.0, 1.0};
        const auto commands = adapter.Convert(action);

        for (double command : commands)
        {
            assert(std::abs(command - 1.0) < 1e-12);
        }
    }

    // Each motor is controlled independently.
    {
        const rl::Action action{-1.0, -0.5, 0.5, 1.0};
        const auto commands = adapter.Convert(action);

        assert(std::abs(commands[0] - 0.0) < 1e-12);
        assert(std::abs(commands[1] - 0.25) < 1e-12);
        assert(std::abs(commands[2] - 0.75) < 1e-12);
        assert(std::abs(commands[3] - 1.0) < 1e-12);
    }

    // Values outside [-1, 1] are clamped.
    {
        const rl::Action action{-2.0, 2.0, -5.0, 5.0};
        const auto commands = adapter.Convert(action);

        assert(commands[0] == 0.0);
        assert(commands[1] == 1.0);
        assert(commands[2] == 0.0);
        assert(commands[3] == 1.0);
    }

    // Non-finite actions are rejected.
    {
        const rl::Action action{
            0.0,
            std::numeric_limits<double>::quiet_NaN(),
            0.0,
            0.0
        };

        bool threw = false;

        try
        {
            adapter.Convert(action);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    std::cout << "RL action adapter tests PASSED\n";

    return 0;
}

