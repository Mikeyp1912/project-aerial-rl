// /tests/rl_observation_normalizer_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "aerial/rl/observation_normalizer.hpp"
#include "aerial/rl/flight_environment.hpp"

int main()
{
    using namespace aerial;

    constexpr double tolerance = 1e-12;

    // Zero observations remain zero.
    {
        rl::Observation observation{};

        const auto normalized =
            rl::NormalizeObservation(observation);

        for (double value : normalized)
        {
            assert(value == 0.0);
        }
    }

    // Verify the scaling of every observation group.
    {
        rl::Observation observation{};

        observation[0] = 5.0;
        observation[1] = -10.0;
        observation[2] = 2.0;

        observation[3] = 5.0;
        observation[4] = -2.5;
        observation[5] = 10.0;

        observation[6] = 1.0;
        observation[7] = 0.0;
        observation[8] = 0.0;
        observation[9] = 0.0;

        observation[10] = 5.0;
        observation[11] = -10.0;
        observation[12] = 2.0;

        observation[13] = 10.0;
        observation[14] = -20.0;
        observation[15] = 5.0;

        const auto normalized =
            rl::NormalizeObservation(observation);

        assert(std::abs(normalized[0] - 0.5) < tolerance);
        assert(std::abs(normalized[1] + 1.0) < tolerance);
        assert(std::abs(normalized[2] - 0.2) < tolerance);

        assert(std::abs(normalized[3] - 0.5) < tolerance);
        assert(std::abs(normalized[4] + 0.25) < tolerance);
        assert(std::abs(normalized[5] - 1.0) < tolerance);

        assert(std::abs(normalized[6] - 1.0) < tolerance);

        assert(std::abs(normalized[10] - 0.5) < tolerance);
        assert(std::abs(normalized[11] + 1.0) < tolerance);
        assert(std::abs(normalized[12] - 0.2) < tolerance);

        assert(std::abs(normalized[13] - 0.5) < tolerance);
        assert(std::abs(normalized[14] + 1.0) < tolerance);
        assert(std::abs(normalized[15] - 0.25) < tolerance);
    }

    // Values outside the configured range must be clipped.
    {
        rl::Observation observation{};

        observation[0] = 100.0;
        observation[3] = -100.0;
        observation[10] = 50.0;
        observation[13] = -100.0;

        const auto normalized =
            rl::NormalizeObservation(observation);

        assert(normalized[0] == 1.0);
        assert(normalized[3] == -1.0);
        assert(normalized[10] == 1.0);
        assert(normalized[13] == -1.0);
    }

    // Clipping can be disabled.
    {
        rl::ObservationNormalizationParameters parameters;
        parameters.clip = false;

        rl::Observation observation{};
        observation[0] = 25.0;

        const auto normalized =
            rl::NormalizeObservation(observation, parameters);

        assert(std::abs(normalized[0] - 2.5) < tolerance);
    }

    // Invalid normalization scales must be rejected.
    {
        rl::ObservationNormalizationParameters parameters;
        parameters.velocity_scale = 0.0;

        bool threw = false;

        try
        {
            rl::NormalizeObservation(rl::Observation{}, parameters);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Non-finite observations must be rejected.
    {
        rl::Observation observation{};

        observation[5] =
            std::numeric_limits<double>::quiet_NaN();

        bool threw = false;

        try
        {
            rl::NormalizeObservation(observation);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }

        assert(threw);
    }

    // Environment integration: normalized observations must
    // match normalization of the raw observation.
    {
        rl::FlightEnvironmentParameters parameters;
        parameters.observation_normalization.position_scale = 5.0;

        rl::FlightEnvironment environment(parameters);

        const auto raw = environment.GetRLObservation();

        const auto expected = rl::NormalizeObservation(
            raw,
            parameters.observation_normalization
        );

        const auto actual =
            environment.GetNormalizedRLObservation();

        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            assert(std::abs(actual[i] - expected[i]) < tolerance);
        }

        const physics::MotorCommands commands{
            0.0, 0.0, 0.0, 0.0
        };

        environment.Step(commands);

        const auto updated =
            environment.GetNormalizedRLObservation();

        for (double value : updated)
        {
            assert(std::isfinite(value));
            assert(value >= -1.0);
            assert(value <= 1.0);
        }
    }

    std::cout << "RL observation normalization tests PASSED\n";

    return 0;
}

