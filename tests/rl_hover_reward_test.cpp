// /tests/rl_hover_reward_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Geometry>

#include "aerial/rl/hover_reward.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-10;
    constexpr double pi = 3.14159265358979323846;

    rl::HoverRewardParameters parameters;

    physics::DroneState state;
    state.position = parameters.target_position;

    // Perfect hover receives the maximum reward.
    {
        const auto reward =
            rl::CalculateHoverReward(state, parameters);

        assert(std::abs(reward.total - 1.0) < tolerance);
        assert(std::abs(reward.position) < tolerance);
        assert(std::abs(reward.velocity) < tolerance);
        assert(std::abs(reward.tilt) < tolerance);
        assert(std::abs(reward.angular_velocity) < tolerance);
    }

    // Position error produces the expected quadratic penalty.
    {
        auto displaced = state;
        displaced.position.x() += 2.0;

        const auto reward =
            rl::CalculateHoverReward(displaced, parameters);

        assert(std::abs(reward.position + 4.0) < tolerance);
        assert(std::abs(reward.total + 3.0) < tolerance);
    }

    // Velocity error produces the expected penalty.
    {
        auto moving = state;
        moving.velocity = Eigen::Vector3d(3.0, 0.0, 0.0);

        const auto reward =
            rl::CalculateHoverReward(moving, parameters);

        assert(std::abs(reward.velocity + 1.8) < tolerance);
    }

    // A 90-degree tilt gives a penalty of -0.5.
    {
        auto tilted = state;
        tilted.orientation =
            Eigen::AngleAxisd(pi / 2.0, Eigen::Vector3d::UnitX());

        const auto reward =
            rl::CalculateHoverReward(tilted, parameters);

        assert(std::abs(reward.tilt + 0.5) < tolerance);
    }

    // Yaw alone does not reduce the hover reward.
    {
        auto yawed = state;
        yawed.orientation =
            Eigen::AngleAxisd(pi / 2.0, Eigen::Vector3d::UnitZ());

        const auto reward =
            rl::CalculateHoverReward(yawed, parameters);

        assert(std::abs(reward.tilt) < tolerance);
        assert(std::abs(reward.total - 1.0) < tolerance);
    }

    // Angular velocity produces a quadratic penalty.
    {
        auto rotating = state;
        rotating.angular_velocity = Eigen::Vector3d(0.0, 0.0, 2.0);

        const auto reward =
            rl::CalculateHoverReward(rotating, parameters);

        assert(std::abs(reward.angular_velocity + 0.4) < tolerance);
    }

    // Negative weights are invalid.
    {
        auto invalid_parameters = parameters;
        invalid_parameters.position_weight = -1.0;

        bool threw = false;

        try
        {
            rl::CalculateHoverReward(state, invalid_parameters);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Non-finite state values are invalid.
    {
        auto invalid_state = state;
        invalid_state.position.z() =
            std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try
        {
            rl::CalculateHoverReward(invalid_state, parameters);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    std::cout << "RL hover reward tests PASSED\n";

    return 0;
}
