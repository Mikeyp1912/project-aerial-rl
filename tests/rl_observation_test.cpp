// /tests/rl_observation_test.cpp


#include <cassert>
#include <cmath>
#include <iostream>

#include "aerial/rl/observation.hpp"

int main()
{
    using namespace aerial;

    sensors::SensorMeasurements measurements;

    measurements.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    measurements.velocity = Eigen::Vector3d(4.0, 5.0, 6.0);

    measurements.orientation =
        Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);

    measurements.angular_velocity =
        Eigen::Vector3d(7.0, 8.0, 9.0);

    measurements.acceleration =
        Eigen::Vector3d(10.0, 11.0, 12.0);

    const rl::Observation observation =
        rl::MakeObservation(measurements);

    static_assert(rl::kObservationSize == 16);

    const rl::Observation expected{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        1.0, 0.0, 0.0, 0.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0
    };

    for (std::size_t i = 0; i < rl::kObservationSize; ++i) {
        assert(std::abs(observation[i] - expected[i]) < 1e-12);
    }

    std::cout << "RL observation tests PASSED\n";
    return 0;
}

