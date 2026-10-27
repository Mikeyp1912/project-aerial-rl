// /tests/sensor_scheduler_test.cpp


#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "aerial/sensors/sensor_scheduler.hpp"

namespace
{
    constexpr double tolerance = 1e-12;

    void AssertMeasurementsEqual(
        const aerial::sensors::SensorMeasurements& a,
        const aerial::sensors::SensorMeasurements& b)
    {
        assert(a.position.isApprox(b.position, tolerance));
        assert(a.velocity.isApprox(b.velocity, tolerance));
        assert(a.angular_velocity.isApprox(b.angular_velocity, tolerance));
        assert(a.acceleration.isApprox(b.acceleration, tolerance));

        // Quaternion coefficients include x, y, z and w.
        assert(a.orientation.coeffs().isApprox(
            b.orientation.coeffs(), tolerance));
    }
}

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
    assert(std::abs(scheduler.GetMeasurements().position.x()) < tolerance);

    // Test 3: Navigation updates at 30 Hz.
    scheduler.Update(state, acceleration, 4.0 * dt);

    assert(scheduler.GetNavigationSampleCount() == 2);
    assert(std::abs(
        scheduler.GetMeasurements().position.x() - 5.0
    ) < tolerance);

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

    // Test 7: Resetting with the same seed reproduces measurements.
    {
        sensors::SensorParameters noisy_parameters;
        noisy_parameters.position_noise =
            Eigen::Vector3d::Constant(0.05);

        noisy_parameters.velocity_noise =
            Eigen::Vector3d::Constant(0.02);

        noisy_parameters.angular_velocity_noise =
            Eigen::Vector3d::Constant(0.01);

        constexpr std::uint32_t seed = 12345;

        sensors::SensorScheduler deterministic(
            noisy_parameters,
            sensors::SensorSchedulerParameters{},
            seed
        );

        physics::DroneState test_state;
        test_state.position = Eigen::Vector3d(1.0, 2.0, 3.0);
        test_state.velocity = Eigen::Vector3d(0.5, -0.2, 0.1);

        const Eigen::Vector3d test_acceleration(0.1, 0.2, 0.3);

        deterministic.Update(test_state, test_acceleration, 0.0);
        const auto first = deterministic.GetMeasurements();

        deterministic.Update(test_state, test_acceleration, dt);
        const auto second = deterministic.GetMeasurements();

        deterministic.Reset(seed);

        deterministic.Update(test_state, test_acceleration, 0.0);
        AssertMeasurementsEqual(first, deterministic.GetMeasurements());

        deterministic.Update(test_state, test_acceleration, dt);
        AssertMeasurementsEqual(second, deterministic.GetMeasurements());

        assert(deterministic.GetImuSampleCount() == 2);
        assert(deterministic.GetNavigationSampleCount() == 1);
    }

    // Test 8: A large time jump does not fabricate past samples.
    {
        sensors::SensorScheduler skipped;

        physics::DroneState test_state;
        skipped.Update(test_state, acceleration, 0.0);

        assert(skipped.GetImuSampleCount() == 1);
        assert(skipped.GetNavigationSampleCount() == 1);

        test_state.position.x() = 10.0;

        skipped.Update(test_state, acceleration, 1.0);

        // One new sample per due sensor, not 120/30 backfilled samples.
        assert(skipped.GetImuSampleCount() == 2);
        assert(skipped.GetNavigationSampleCount() == 2);

        assert(std::abs(
            skipped.GetMeasurements().position.x() - 10.0
        ) < tolerance);

        // The scheduler has advanced beyond the skipped intervals.
        skipped.Update(test_state, acceleration, 1.0);

        assert(skipped.GetImuSampleCount() == 2);
        assert(skipped.GetNavigationSampleCount() == 2);

        skipped.Update(test_state, acceleration, 1.0 + dt);

        assert(skipped.GetImuSampleCount() == 3);
        assert(skipped.GetNavigationSampleCount() == 2);
    }

    // Test 9: Reset restores the initial scheduling state.
    {
        sensors::SensorScheduler reset_scheduler;

        physics::DroneState test_state;
        test_state.position = Eigen::Vector3d(4.0, 5.0, 6.0);

        reset_scheduler.Update(test_state, acceleration, 0.0);
        reset_scheduler.Update(test_state, acceleration, dt);
        reset_scheduler.Update(test_state, acceleration, 4.0 * dt);

        reset_scheduler.Reset(42);

        assert(reset_scheduler.GetImuSampleCount() == 0);
        assert(reset_scheduler.GetNavigationSampleCount() == 0);

        const auto& measurements = reset_scheduler.GetMeasurements();

        assert(measurements.position.isZero());
        assert(measurements.velocity.isZero());
        assert(measurements.angular_velocity.isZero());
        assert(measurements.acceleration.isZero());

        // Compare against a newly constructed scheduler's initial state.
        sensors::SensorScheduler fresh;

        AssertMeasurementsEqual(
            measurements,
            fresh.GetMeasurements()
        );

        // Both sensor groups must be eligible to sample at time zero.
        reset_scheduler.Update(test_state, acceleration, 0.0);

        assert(reset_scheduler.GetImuSampleCount() == 1);
        assert(reset_scheduler.GetNavigationSampleCount() == 1);

        assert(std::abs(
            reset_scheduler.GetMeasurements().position.x() - 4.0
        ) < tolerance);
    }

    std::cout << "Sensor scheduler tests PASSED\n";

    return 0;
}

