// /include/aerial/rl/policy.hpp

#pragma once

#include "aerial/rl/rl_types.hpp"

namespace aerial::rl
{
    class Policy
    {
        public:
            virtual ~Policy() = default;

            virtual void Reset() {}

            virtual Action Predict(
                    const Observation& observation
                    ) = 0;
    };

    // Always outputs the same normalized action
    class ConstantPolicy final : public Policy
    {
        public:
            explicit ConstantPolicy(const Action& action = {});

            Action Predict(
                    const Observation& observation
                    ) override;

        private:
            Action action_;
    };

    // Simple proportional feedback policy for PIDTargets mode.
    // This is a deterministic baseline, not a trained network
    struct HoverFeedbackPolicyParameters
    {
        double target_altitude = 2.0;

        double horizontal_gain = 0.2;
        double altitude_gain = 0.5;

        double maximum_horizontal_offset = 5.0;
        double minimum_altitude = 0.5;
        double maximum_altitude = 5.0;

        double observation_position_scale = 10.0;
    };

    class HoverFeedbackPolicy final : public Policy
    {
        public:
            explicit HoverFeedbackPolicy(
                    const HoverFeedbackPolicyParameters& parameters = {}
                    );

            Action Predict(
                    const Observation& observation
                    ) override;

        private:
            HoverFeedbackPolicyParameters parameters_;
    };

    // Enforce the common policy output contract
    Action ValidatePolicyAction(const Action& action);
}
