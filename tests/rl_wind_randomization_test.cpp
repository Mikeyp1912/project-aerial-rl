

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <Eigen/Core>

#include "aerial/rl/flight_environment.hpp"

namespace
{
    using aerial::rl::FlightEnvironment;
    using aerial::rl::FlightEnvironmentParameters;

    void TestWindDisabled()
    {
        FlightEnvironment environment;

        environment.Reset(42);

        assert(environment.GetWindVelocity().isZero(1e-12));

        environment.Reset(123);

        assert(environment.GetWindVelocity().isZero(1e-12));
    }

    void TestDeterministicWind()
    {
        FlightEnvironmentParameters parameters;

        parameters.wind_randomization.enabled = true;
        parameters.wind_randomization.maximum_horizontal_speed = 5.0;

        FlightEnvironment environment(parameters);

        environment.Reset(42);

        const Eigen::Vector3d first =
            environment.GetWindVelocity();

        environment.Reset(42);

        const Eigen::Vector3d second =
            environment.GetWindVelocity();

        assert((first - second).norm() < 1e-12);
        assert(first.allFinite());
        assert(std::abs(first.z()) < 1e-12);
        assert(first.head<2>().norm() <= 5.0 + 1e-12);
    }

    void TestDifferentSeeds()
    {
        FlightEnvironmentParameters parameters;

        parameters.wind_randomization.enabled = true;
        parameters.wind_randomization.maximum_horizontal_speed = 5.0;

        FlightEnvironment environment(parameters);

        environment.Reset(42);

        const Eigen::Vector3d first =
            environment.GetWindVelocity();

        environment.Reset(43);

        const Eigen::Vector3d second =
            environment.GetWindVelocity();

        assert((first - second).norm() > 1e-9);
    }

    void TestInitialStateIndependence()
    {
        FlightEnvironmentParameters parameters;

        parameters.initial_state_randomization.enabled = true;

        FlightEnvironment without_wind(parameters);

        parameters.wind_randomization.enabled = true;
        parameters.wind_randomization.maximum_horizontal_speed = 5.0;

        FlightEnvironment with_wind(parameters);

        without_wind.Reset(123);
        with_wind.Reset(123);

        assert(
            (without_wind.GetTrueState().position -
             with_wind.GetTrueState().position).norm() < 1e-12
        );

        assert(
            (without_wind.GetTrueState().velocity -
             with_wind.GetTrueState().velocity).norm() < 1e-12
        );
    }

    void TestZeroMaximumWind()
    {
        FlightEnvironmentParameters parameters;

        parameters.wind_randomization.enabled = true;
        parameters.wind_randomization.maximum_horizontal_speed = 0.0;

        FlightEnvironment environment(parameters);

        environment.Reset(42);

        assert(environment.GetWindVelocity().isZero(1e-12));
    }

    void TestInvalidConfiguration()
    {
        FlightEnvironmentParameters parameters;

        parameters.wind_randomization.enabled = true;
        parameters.wind_randomization.maximum_horizontal_speed = -1.0;

        bool threw = false;

        try {
            FlightEnvironment environment(parameters);
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }
}

int main()
{
    TestWindDisabled();
    TestDeterministicWind();
    TestDifferentSeeds();
    TestInitialStateIndependence();
    TestZeroMaximumWind();
    TestInvalidConfiguration();

    std::cout << "RL wind randomization tests PASSED\n";

    return 0;
}
