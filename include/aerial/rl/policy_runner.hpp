// /include/aerial/rl/policy_runner.hpp

#pragma once

#include "aerial/rl/flight_environment.hpp"
#include "aerial/rl/policy.hpp"

namespace aerial::rl
{
    class PolicyRunner
    {
        public:
            PolicyRunner(
                    FlightEnvironment& environment,
                    Policy& policy
                    );

            void Reset(std::uint32_t seed = 42);

            FlightStepResult Step();

            const Action& GetLastAction() const;

        private:
            FlightEnvironment& environment_;
            Policy& policy_;

            Action last_action_{};
    };
}
