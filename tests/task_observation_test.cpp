#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/rl/task_observation.hpp"
#include "aerial/rl/flight_environment.hpp"

int main()
{
    using namespace aerial::rl;

    static_assert(kObservationSize == 16);
    static_assert(kHoverObservationSize == 19);

    assert(GetObservationSize(ObservationMode::Base) == 16);
    assert(GetObservationSize(ObservationMode::HoverAugmented) == 19);

    Observation base{};
    base[0] = 1.0;
    base[1] = -2.0;
    base[2] = 2.5;
    base[6] = 1.0;

    const Eigen::Vector3d target(0.0, 0.0, 2.0);

    ObservationNormalizationParameters normalization;

    const auto augmented =
        MakeHoverObservation(base, target, normalization);

    assert(std::abs(augmented[16] - 0.1) < 1e-12);
    assert(std::abs(augmented[17] + 0.2) < 1e-12);
    assert(std::abs(augmented[18] - 0.05) < 1e-12);

    // The original observations must remain unchanged.
    const auto expected_base =
        NormalizeObservation(base, normalization);

    for (std::size_t i = 0; i < kObservationSize; ++i)
    {
        assert(std::abs(augmented[i] - expected_base[i]) < 1e-12);
    }

    // At the target, all three error features are zero.
    base[0] = target.x();
    base[1] = target.y();
    base[2] = target.z();

    const auto at_target =
        MakeHoverObservation(base, target, normalization);

    for (std::size_t i = 16; i < 19; ++i)
    {
        assert(std::abs(at_target[i]) < 1e-12);
    }

    // Environment-level observation selection.
    FlightEnvironmentParameters parameters;
    parameters.observation_mode = ObservationMode::HoverAugmented;

    FlightEnvironment environment(parameters);

    const auto observation = environment.GetPolicyObservation();

    assert(observation.size() == 19);
    assert(environment.GetPolicyObservationSize() == 19);

    const auto& measured_position =
        environment.GetObservation().position;

    for (int axis = 0; axis < 3; ++axis)
    {
        const double expected =
            (measured_position[axis] -
             parameters.hover_reward.target_position[axis]) /
            parameters.observation_normalization.position_scale;

        assert(std::abs(observation[16 + axis] - expected) < 1e-12);
    }

    parameters.observation_mode = ObservationMode::Base;

    FlightEnvironment base_environment(parameters);

    assert(base_environment.GetPolicyObservation().size() == 16);

    std::cout << "Task observation tests PASSED\n";
}
