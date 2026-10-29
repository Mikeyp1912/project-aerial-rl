// /src/rl/policy_runner.cpp

#include "aerial/rl/policy_runner.hpp"

namespace aerial::rl
{
    PolicyRunner::PolicyRunner(
            FlightEnvironment& environment,
            Policy& policy
            )
        :
            environment_(environment),
            policy_(policy)
    {
    }

    void PolicyRunner::Reset(std::uint32_t seed)
    {
        environment_.Reset(seed);
        policy_.Reset();
        last_action_ = {};
    }

    FlightStepResult PolicyRunner::Step()
    {
        const Observation observation = environment_.GetNormalizedRLObservation();

        last_action_ = ValidatePolicyAction(
                policy_.Predict(observation)
                );

        return environment_.StepAction(last_action_);
    }

    const Action& PolicyRunner::GetLastAction() const
    {
        return last_action_;
    }
}
