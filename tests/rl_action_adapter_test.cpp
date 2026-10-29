// /tests/rl_action_adapter_test.cpp



#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/rl/action_adapter.hpp"
#include "aerial/physics/motor_mixer.hpp"

namespace
{
    constexpr double kTolerance = 1e-12;

    bool ApproximatelyEqual(double a, double b)
    {
        return std::abs(a - b) < kTolerance;
    }

    void AssertCommandsEqual(
        const aerial::physics::MotorCommands& actual,
        const aerial::physics::MotorCommands& expected)
    {
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            assert(ApproximatelyEqual(actual[i], expected[i]));
        }
    }

    void AssertCommandsValid(
        const aerial::physics::MotorCommands& commands)
    {
        for (double command : commands)
        {
            assert(std::isfinite(command));
            assert(command >= 0.0);
            assert(command <= 1.0);
        }
    }
}

int main()
{
    using namespace aerial;

    constexpr double dt = 1.0 / 120.0;

    physics::DroneParameters drone_parameters;
    sensors::SensorMeasurements measurements;

    measurements.angular_velocity = Eigen::Vector3d::Zero();

    // ---------------------------------------------------------
    // Direct motor action tests
    // ---------------------------------------------------------

    rl::DirectMotorActionAdapter direct_adapter;

    {
        const rl::Action action{-1.0, -1.0, -1.0, -1.0};
        const auto commands = direct_adapter.Convert(action);

        for (double command : commands)
            assert(ApproximatelyEqual(command, 0.0));
    }

    {
        const rl::Action action{0.0, 0.0, 0.0, 0.0};
        const auto commands = direct_adapter.Convert(action);

        for (double command : commands)
            assert(ApproximatelyEqual(command, 0.5));
    }

    {
        const rl::Action action{1.0, 1.0, 1.0, 1.0};
        const auto commands = direct_adapter.Convert(action);

        for (double command : commands)
            assert(ApproximatelyEqual(command, 1.0));
    }

    {
        const rl::Action action{-1.0, -0.5, 0.5, 1.0};
        const auto commands = direct_adapter.Convert(action);

        assert(ApproximatelyEqual(commands[0], 0.0));
        assert(ApproximatelyEqual(commands[1], 0.25));
        assert(ApproximatelyEqual(commands[2], 0.75));
        assert(ApproximatelyEqual(commands[3], 1.0));
    }

    {
        const rl::Action action{-2.0, 2.0, -5.0, 5.0};
        const auto commands = direct_adapter.Convert(action);

        assert(commands[0] == 0.0);
        assert(commands[1] == 1.0);
        assert(commands[2] == 0.0);
        assert(commands[3] == 1.0);
    }

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
            direct_adapter.Convert(action);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // ---------------------------------------------------------
    // Body-rate action tests
    // ---------------------------------------------------------

    // Zero desired rates and zero measured rates should produce
    // equal motor commands at 50% collective thrust.
    {
        rl::BodyRateActionAdapter adapter;

        const rl::Action action{0.0, 0.0, 0.0, 0.0};

        const auto commands = adapter.Convert(
            action, measurements, drone_parameters, dt
        );

        AssertCommandsValid(commands);

        for (double command : commands)
            assert(ApproximatelyEqual(command, 0.5));
    }

    // Minimum collective thrust with zero rates stops all motors.
    {
        rl::BodyRateActionAdapter adapter;

        const rl::Action action{0.0, 0.0, 0.0, -1.0};

        const auto commands = adapter.Convert(
            action, measurements, drone_parameters, dt
        );

        for (double command : commands)
            assert(ApproximatelyEqual(command, 0.0));
    }

    // Maximum collective thrust with zero rates drives all
    // motors to their maximum normalized command.
    {
        rl::BodyRateActionAdapter adapter;

        const rl::Action action{0.0, 0.0, 0.0, 1.0};

        const auto commands = adapter.Convert(
            action, measurements, drone_parameters, dt
        );

        for (double command : commands)
            assert(ApproximatelyEqual(command, 1.0));
    }

    // Compare normalized action scaling against the existing
    // rate controller and motor mixer.
    {
        rl::BodyRateActionParameters parameters;
        parameters.maximum_rates = Eigen::Vector3d(2.0, 4.0, 1.0);

        rl::BodyRateActionAdapter adapter(parameters);

        const rl::Action action{0.5, -0.25, 0.75, 0.0};

        const Eigen::Vector3d expected_rates{
            1.0, -1.0, 0.75
        };

        control::RateControllerState reference_state;

        const Eigen::Vector3d expected_torque =
            control::CalculateBodyTorque(
                expected_rates,
                measurements.angular_velocity,
                dt,
                parameters.rate_controller,
                reference_state
            );

        const auto expected_commands = physics::MixMotors(
            drone_parameters,
            2.0 * drone_parameters.max_motor_thrust,
            expected_torque
        ).commands;

        const auto actual_commands = adapter.Convert(
            action, measurements, drone_parameters, dt
        );

        AssertCommandsEqual(actual_commands, expected_commands);
    }

    // Out-of-range actions must be clamped.
    {
        rl::BodyRateActionAdapter adapter_a;
        rl::BodyRateActionAdapter adapter_b;

        const rl::Action excessive{3.0, -2.0, 5.0, 2.0};
        const rl::Action bounded{1.0, -1.0, 1.0, 1.0};

        const auto a = adapter_a.Convert(
            excessive, measurements, drone_parameters, dt
        );

        const auto b = adapter_b.Convert(
            bounded, measurements, drone_parameters, dt
        );

        AssertCommandsEqual(a, b);
        AssertCommandsValid(a);
    }

    // Invalid action values must be rejected.
    {
        rl::BodyRateActionAdapter adapter;

        const rl::Action action{
            0.0,
            std::numeric_limits<double>::infinity(),
            0.0,
            0.0
        };

        bool threw = false;

        try
        {
            adapter.Convert(
                action, measurements, drone_parameters, dt
            );
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Invalid time steps must be rejected.
    {
        rl::BodyRateActionAdapter adapter;

        bool threw = false;

        try
        {
            adapter.Convert(
                rl::Action{}, measurements, drone_parameters, 0.0
            );
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Invalid configuration must be rejected.
    {
        rl::BodyRateActionParameters parameters;
        parameters.maximum_rates.x() = -1.0;

        bool threw = false;

        try
        {
            rl::BodyRateActionAdapter adapter(parameters);
            (void)adapter;
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Motor outputs must stay within [0, 1] even under
    // large rate requests and controller saturation.
    {
        rl::BodyRateActionAdapter adapter;

        const rl::Action action{1.0, -1.0, 1.0, 0.8};

        for (int i = 0; i < 100; ++i)
        {
            const auto commands = adapter.Convert(
                action, measurements, drone_parameters, dt
            );

            AssertCommandsValid(commands);
        }
    }

    // Reset must clear PID state. Use nonzero integral gains
    // so accumulated state actually influences the commands.
    {
        rl::BodyRateActionParameters parameters;
        parameters.rate_controller.ki =
            Eigen::Vector3d(0.02, 0.02, 0.002);

        rl::BodyRateActionAdapter adapter(parameters);
        rl::BodyRateActionAdapter fresh_adapter(parameters);

        const rl::Action action{0.2, -0.1, 0.1, 0.0};

        const auto initial = fresh_adapter.Convert(
            action, measurements, drone_parameters, dt
        );

        for (int i = 0; i < 50; ++i)
        {
            adapter.Convert(
                action, measurements, drone_parameters, dt
            );
        }

        adapter.Reset();

        const auto after_reset = adapter.Convert(
            action, measurements, drone_parameters, dt
        );

        AssertCommandsEqual(after_reset, initial);
    }

    std::cout << "RL action adapter tests PASSED\n";

    return 0;
}

