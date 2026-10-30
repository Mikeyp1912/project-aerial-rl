// /tests/rl_initial_state_randomization_test.cpp



#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "aerial/rl/flight_environment.hpp"

namespace
{
    bool SameVector(
        const Eigen::Vector3d& a,
        const Eigen::Vector3d& b
    )
    {
        return (a.array() == b.array()).all();
    }

    void TestDefaultReset()
    {
        aerial::rl::FlightEnvironment environment;

        environment.Reset(42);

        const auto& state = environment.GetTrueState();

        assert(state.position.x() == 0.0);
        assert(state.position.y() == 0.0);
        assert(state.position.z() == 1.0);

        assert(state.velocity.isZero());

        std::cout << "Default reset: PASS\n";
    }

    void TestDeterministicReset()
    {
        aerial::rl::FlightEnvironmentParameters parameters;

        parameters.initial_state_randomization.enabled = true;

        aerial::rl::FlightEnvironment environment(parameters);

        environment.Reset(123);

        const auto first = environment.GetTrueState();

        environment.Reset(123);

        const auto second = environment.GetTrueState();

        assert(SameVector(first.position, second.position));
        assert(SameVector(first.velocity, second.velocity));

        std::cout << "Deterministic reset: PASS\n";
    }

    void TestDifferentSeeds()
    {
        aerial::rl::FlightEnvironmentParameters parameters;

        parameters.initial_state_randomization.enabled = true;

        aerial::rl::FlightEnvironment environment(parameters);

        environment.Reset(123);
        const auto first = environment.GetTrueState();

        environment.Reset(456);
        const auto second = environment.GetTrueState();

        const bool position_differs =
            !SameVector(first.position, second.position);

        const bool velocity_differs =
            !SameVector(first.velocity, second.velocity);

        assert(position_differs || velocity_differs);

        std::cout << "Different seeds: PASS\n";
    }

    void TestRandomizationBounds()
    {
        aerial::rl::FlightEnvironmentParameters parameters;

        parameters.initial_state_randomization.enabled = true;

        aerial::rl::FlightEnvironment environment(parameters);

        const auto& randomization =
            parameters.initial_state_randomization;

        for (std::uint32_t seed = 0; seed < 1000; ++seed) {
            environment.Reset(seed);

            const auto& state = environment.GetTrueState();

            const Eigen::Vector3d offset =
                state.position -
                Eigen::Vector3d(0.0, 0.0, 1.0);

            for (int axis = 0; axis < 3; ++axis) {
                assert(
                    std::abs(offset[axis]) <=
                    randomization.maximum_position_offset[axis]
                    + 1e-12
                );

                assert(
                    std::abs(state.velocity[axis]) <=
                    randomization.maximum_velocity[axis]
                    + 1e-12
                );
            }

            assert(state.position.z() > 0.0);
            assert(state.orientation.isApprox(
                Eigen::Quaterniond::Identity()
            ));
        }

        std::cout << "Randomization bounds: PASS\n";
    }

    void TestInvalidParameters()
    {
        aerial::rl::FlightEnvironmentParameters parameters;

        parameters.initial_state_randomization
            .maximum_position_offset.x() = -0.1;

        bool rejected = false;

        try {
            aerial::rl::FlightEnvironment environment(parameters);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }

        assert(rejected);

        std::cout << "Invalid parameters: PASS\n";
    }
}

int main()
{
    TestDefaultReset();
    TestDeterministicReset();
    TestDifferentSeeds();
    TestRandomizationBounds();
    TestInvalidParameters();

    std::cout
        << "Initial-state randomization tests PASSED\n";

    return 0;
}
