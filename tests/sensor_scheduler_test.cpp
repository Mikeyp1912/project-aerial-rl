// /tests/sensor_scheduler_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/sensors/sensor_scheduler.hpp"

int main()
{
    using namespace aerial;

    constexpr double dt = 1.0 / 120.0;

    sensors::SensorScheduler scheduler;

    physics::DroneState state;

    const Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();

    // Test 1: Both sensors sample at time zero.
    scheduler.Update(state, acceleration, 0.0);

    assert(scheduler.GetImuSampleCount() == 1);
    assert(scheduler.GetNavigationSampleCount() == 1);

    // Test 2: Navigation data is held between updates.
    state.position.x() = 5.0;

    scheduler.Update(state, acceleration, dt);

    assert(scheduler.GetImuSampleCount() == 2);
    assert(scheduler.GetNavigationSampleCount() == 1);

    assert(
        std::abs(scheduler.GetMeasurements().position.x()) < 1e-12
    );

    // Test 3: Navigation updates at 30 Hz.
    scheduler.Update(state, acceleration, 4.0 * dt);

    assert(scheduler.GetNavigationSampleCount() == 2);

    assert(
        std::abs(
            scheduler.GetMeasurements().position.x() - 5.0
        ) < 1e-12
    );

    // Test 4: Reset clears counters and measurements.
    scheduler.Reset(42);

    assert(scheduler.GetImuSampleCount() == 0);
    assert(scheduler.GetNavigationSampleCount() == 0);
    assert(scheduler.GetMeasurements().position.isZero());

    // Test 5: Run a full second at 120 Hz.
    scheduler.Update(state, acceleration, 0.0);

    for (int i = 1; i < 120; ++i) {
        scheduler.Update(state, acceleration, i * dt);
    }

    assert(scheduler.GetImuSampleCount() == 120);
    assert(scheduler.GetNavigationSampleCount() == 30);

    // Test 6: Invalid frequency is rejected.
    sensors::SensorSchedulerParameters invalid;
    invalid.imu_frequency = 0.0;

    bool rejected = false;

    try {
        sensors::SensorScheduler bad(
            sensors::SensorParameters{},
            invalid
        );
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }

    assert(rejected);

    std::cout << "Sensor scheduler tests PASSED\n";

    return 0;
}
