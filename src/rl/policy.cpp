// /src/rl/policy.cpp

#include "aerial/rl/policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    Action ValidatePolicyAction(const Action& action)
    {
        Action validated = action;

        for (double& value : validated) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Policy action must contain finite values");
            }

            value = std::clamp(value, -1.0, 1.0);
        }

        return validated;
    }

    ConstantPolicy::ConstantPolicy(const Action& action)
        :
            action_(ValidatePolicyAction(action))
    {
    }

    Action ConstantPolicy::Predict(const Observation& observation)
    {
            for (const double value : observation) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("Policy observation must be finite");
                }
            }

        return action_;
    }

    HoverFeedbackPolicy::HoverFeedbackPolicy(
            const HoverFeedbackPolicyParameters& parameters
            )
        :
            parameters_(parameters)
    {
        const double values[] = {
            parameters_.target_altitude,
            parameters_.horizontal_gain,
            parameters_.altitude_gain,
            parameters_.maximum_horizontal_offset,
            parameters_.minimum_altitude,
            parameters_.maximum_altitude,
            parameters_.observation_position_scale
        };

        for (const double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Hover feedback parameters must be finite");
            }
        }

        if (parameters_.horizontal_gain < 0.0 ||
            parameters_.altitude_gain < 0.0 ||
            parameters_.maximum_horizontal_offset <= 0.0 ||
            parameters_.maximum_altitude <= parameters_.minimum_altitude ||
            parameters_.observation_position_scale <= 0.0 ||
            parameters_.target_altitude < parameters_.minimum_altitude ||
            parameters_.target_altitude > parameters_.maximum_altitude)
        {
            throw std::invalid_argument(
                "Invalid hover feedback policy parameters"
            );
        }
    }

    Action HoverFeedbackPolicy::Predict(
            const Observation& observation
            )
    {
        for (const double value : observation) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Policy observation must be finite");
            }
        }

        // Normalized observation indices 0, 1, 2 are XYZ position
        const double x = observation[0] * parameters_.observation_position_scale;

        const double y = observation[1] * parameters_.observation_position_scale;

        const double z = observation[2] * parameters_.observation_position_scale;

        // PIDTargets uses normalized horizontal target offsets.
        // Counteract position error with a small target correction
        const double target_x_offset = -parameters_.horizontal_gain * x;

        const double target_y_offset = -parameters_.horizontal_gain * y;

        const double x_action = target_x_offset / parameters_.maximum_horizontal_offset;

        const double y_action = target_y_offset / parameters_.maximum_horizontal_offset;

        // Map desired altitude to the PIDTargets action range
        const double altitude_command =
            parameters_.target_altitude +
            parameters_.altitude_gain *
            (parameters_.target_altitude - z);

        const double altitude_action =
            2.0 *
            (altitude_command - parameters_.minimum_altitude) /
            (parameters_.maximum_altitude - parameters_.minimum_altitude)
            - 1.0;

        // Yaw target is zero
        return ValidatePolicyAction({
                x_action,
                y_action,
                altitude_action,
                0.0
                });
    }
}
