// /tests/sensor_model_test.cpp

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "aerial/sensors/sensor_model.hpp"

namespace
{
    bool Near(double a, double b, double tolerance = 1e-12)
    {
        return std::abs(a - b) < tolerance;
    }
}

int main()
{
    using namespace aerial;

    physics::DroneState state;

    state.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    state.velocity = Eigen::Vector3d(-1.0, 0.5, 2.0);
    state.angular_velocity = Eigen::Vector3d(0.1, -0.2, 0.3);

    // Test 1: Default sensor returns exact measurements.
    sensors::SensorModel ideal_sensor;

    const auto ideal = ideal_sensor.Measure(state);

    assert(ideal.position.isApprox(state.position));
    assert(ideal.velocity.isApprox(state.velocity));
    assert(ideal.orientation.isApprox(state.orientation));
    assert(ideal.angular_velocity.isApprox(state.angular_velocity));

    // Test 2: Constant biases are applied correctly.
    sensors::SensorParameters biased_parameters;

    biased_parameters.position_bias =
        Eigen::Vector3d(0.1, -0.2, 0.3);

    biased_parameters.velocity_bias =
        Eigen::Vector3d(0.05, 0.0, -0.05);

    biased_parameters.angular_velocity_bias =
        Eigen::Vector3d(0.01, 0.02, -0.01);

    sensors::SensorModel biased_sensor(biased_parameters);

    const auto biased = biased_sensor.Measure(state);

    assert(
        (biased.position -
         state.position -
         biased_parameters.position_bias).norm() < 1e-12
    );

    assert(
        (biased.velocity -
         state.velocity -
         biased_parameters.velocity_bias).norm() < 1e-12
    );

    assert(
        (biased.angular_velocity -
         state.angular_velocity -
         biased_parameters.angular_velocity_bias).norm() < 1e-12
    );

    // Test 3: Identical seeds generate identical noise.
    sensors::SensorParameters noisy_parameters;

    noisy_parameters.position_noise =
        Eigen::Vector3d(0.1, 0.1, 0.1);

    noisy_parameters.velocity_noise =
        Eigen::Vector3d(0.05, 0.05, 0.05);

    noisy_parameters.angular_velocity_noise =
        Eigen::Vector3d(0.01, 0.01, 0.01);

    sensors::SensorModel first(noisy_parameters, 1234);
    sensors::SensorModel second(noisy_parameters, 1234);

    for (int i = 0; i < 100; ++i) {
        const auto a = first.Measure(state);
        const auto b = second.Measure(state);

        assert(a.position.isApprox(b.position));
        assert(a.velocity.isApprox(b.velocity));
        assert(a.angular_velocity.isApprox(b.angular_velocity));
    }

    // Test 4: Resetting the seed reproduces the sequence.
    first.Reset(1234);
    second.Reset(1234);

    const auto reset_a = first.Measure(state);
    const auto reset_b = second.Measure(state);

    assert(reset_a.position.isApprox(reset_b.position));
    assert(reset_a.velocity.isApprox(reset_b.velocity));

    // Test 5: Noise actually changes measurements.
    assert(
        (reset_a.position - state.position).norm() > 1e-9
    );

    // Test 6: Invalid noise parameters are rejected.
    sensors::SensorParameters invalid_parameters;
    invalid_parameters.position_noise.x() = -0.1;

    bool rejected = false;

    try {
        sensors::SensorModel invalid_sensor(invalid_parameters);
    }
    catch (const std::invalid_argument&) {
        rejected = true;
    }

    assert(rejected);

    // Test 7: Constant orientation bias is applied correctly.
    {
        sensors::SensorParameters parameters;

        constexpr double bias_angle = 0.1;

        parameters.orientation_bias =
            Eigen::Vector3d(0.0, 0.0, bias_angle);

        sensors::SensorModel sensor(parameters);

        const auto measurement = sensor.Measure(state);

        const Eigen::Quaterniond expected =
            state.orientation *
            Eigen::Quaterniond(
                Eigen::AngleAxisd(
                    bias_angle,
                    Eigen::Vector3d::UnitZ()
                )
            );

        assert(measurement.orientation.isApprox(expected, 1e-12));
        assert(Near(measurement.orientation.norm(), 1.0));
    }

    // Test 8: Noisy orientation remains normalized.
    {
        sensors::SensorParameters parameters;

        parameters.orientation_noise =
            Eigen::Vector3d(0.02, 0.02, 0.04);

        sensors::SensorModel sensor(parameters, 5678);

        bool observed_noise = false;

        for (int i = 0; i < 1000; ++i) {
            const auto measurement = sensor.Measure(state);

            assert(measurement.orientation.coeffs().allFinite());
            assert(Near(measurement.orientation.norm(), 1.0));

            if (!measurement.orientation.isApprox(
                    state.orientation, 1e-9)) {
                observed_noise = true;
            }
        }

        assert(observed_noise);
    }

    // Test 9: Identical seeds reproduce orientation noise.
    {
        sensors::SensorParameters parameters;

        parameters.orientation_noise =
            Eigen::Vector3d(0.01, 0.02, 0.03);

        sensors::SensorModel first_sensor(parameters, 100);
        sensors::SensorModel second_sensor(parameters, 100);

        for (int i = 0; i < 100; ++i) {
            const auto a = first_sensor.Measure(state);
            const auto b = second_sensor.Measure(state);

            assert(a.orientation.isApprox(b.orientation, 1e-12));
        }

        first_sensor.Reset(100);
        second_sensor.Reset(100);

        const auto a = first_sensor.Measure(state);
        const auto b = second_sensor.Measure(state);

        assert(a.orientation.isApprox(b.orientation, 1e-12));
    }

    // Test 10: Invalid orientation noise is rejected.
    {
        sensors::SensorParameters parameters;

        parameters.orientation_noise.z() = -0.01;

        bool rejected = false;

        try {
            sensors::SensorModel sensor(parameters);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }

        assert(rejected);
    }

    std::cout << "Sensor model tests PASSED\n";

    return 0;
}
