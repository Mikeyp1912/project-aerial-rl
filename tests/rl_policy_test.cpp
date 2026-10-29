// /tests/rl_policy_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/rl/policy.hpp"
#include "aerial/rl/policy_runner.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-12;

    // Constant policy must reproduce its configured action.
    {
        const rl::Action expected{0.1, -0.2, 0.3, -0.4};

        rl::ConstantPolicy policy(expected);

        const rl::Action actual = policy.Predict(
            rl::Observation{}
        );

        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            assert(std::abs(actual[i] - expected[i]) < tolerance);
        }
    }

    // Policy actions must be clipped to [-1, 1].
    {
        const rl::Action input{2.0, -3.0, 0.5, 0.0};

        const auto output = rl::ValidatePolicyAction(input);

        assert(output[0] == 1.0);
        assert(output[1] == -1.0);
        assert(output[2] == 0.5);
        assert(output[3] == 0.0);
    }

    // Non-finite policy actions must be rejected.
    {
        rl::Action action{};

        action[0] = std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try
        {
            rl::ValidatePolicyAction(action);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Hover feedback should counteract horizontal position error.
    {
        rl::HoverFeedbackPolicy policy;

        rl::Observation observation{};

        observation[0] = 0.2;   // X = +2 metres
        observation[1] = -0.1;  // Y = -1 metre
        observation[2] = 0.1;   // Z = +1 metre

        const auto action = policy.Predict(observation);

        assert(action[0] < 0.0);
        assert(action[1] > 0.0);
        const double expected_altitude_action = -1.0 / 9.0;

        assert(std::abs(
            action[2] - expected_altitude_action
        ) < tolerance);
        assert(action[3] == 0.0);
    }

    // Runner must advance the simulation and record its action.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::DirectMotor;

        rl::FlightEnvironment environment(parameters);

        rl::ConstantPolicy policy({
            0.0, 0.0, 0.0, 0.0
        });

        rl::PolicyRunner runner(environment, policy);

        runner.Reset(42);

        const auto result = runner.Step();

        assert(!result.terminated);
        assert(!result.truncated);

        assert(std::abs(
            environment.GetSimulationTime() - parameters.time_step
        ) < tolerance);

        for (double value : runner.GetLastAction())
        {
            assert(value == 0.0);
        }
    }

    // Policy runner must be deterministic for a fixed seed.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.action_mode = rl::ActionMode::PIDTargets;

        rl::FlightEnvironment environment(parameters);
        rl::HoverFeedbackPolicy policy;

        rl::PolicyRunner runner(environment, policy);

        runner.Reset(123);

        const auto first = runner.Step();
        const auto first_position =
            environment.GetTrueState().position.eval();

        runner.Reset(123);

        const auto second = runner.Step();

        assert(std::abs(first.reward - second.reward) < tolerance);

        assert(environment.GetTrueState().position.isApprox(
            first_position, tolerance
        ));
    }

    // Runner must respect the environment's episode limit.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.maximum_episode_time = parameters.time_step;

        rl::FlightEnvironment environment(parameters);
        rl::ConstantPolicy policy;

        rl::PolicyRunner runner(environment, policy);

        runner.Reset();

        const auto result = runner.Step();

        assert(result.truncated);

        bool threw = false;

        try
        {
            runner.Step();
        }
        catch (const std::logic_error&)
        {
            threw = true;
        }

        assert(threw);
    }

    std::cout << "RL policy tests PASSED\n";

    return 0;
}

