// /tests/accelerometer_test.cpp
#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "aerial/physics/integrator.hpp"
#include "aerial/sensors/sensor_model.hpp"

namespace
{
    bool Near(double a, double b, double tolerance = 1e-9)
    {
        return std::abs(a - b) < tolerance;
    }
}

int main()
{
    using namespace aerial;

    constexpr double g = physics::gravity;

    physics::DroneState state;
    sensors::SensorModel ideal_sensor;

    // Test 1: Upright stationary drone measures +g on body Z.
    {
        const auto measurement = ideal_sensor.Measure(
            state,
            Eigen::Vector3d::Zero()
        );

        assert(Near(measurement.acceleration.x(), 0.0));
        assert(Near(measurement.acceleration.y(), 0.0));
        assert(Near(measurement.acceleration.z(), g));
    }

    // Test 2: Free fall produces zero specific force.
    {
        const auto measurement = ideal_sensor.Measure(
            state,
            Eigen::Vector3d(0.0, 0.0, -g)
        );

        assert(measurement.acceleration.norm() < 1e-9);
    }

    // Test 3: Horizontal acceleration is measured correctly.
    {
        const auto measurement = ideal_sensor.Measure(
            state,
            Eigen::Vector3d(2.0, -3.0, 0.0)
        );

        assert(Near(measurement.acceleration.x(), 2.0));
        assert(Near(measurement.acceleration.y(), -3.0));
        assert(Near(measurement.acceleration.z(), g));
    }

    // Test 4: Orientation rotates the accelerometer axes.
    {
        state.orientation = Eigen::Quaterniond(
            Eigen::AngleAxisd(
                std::acos(-1.0) / 2.0,
                Eigen::Vector3d::UnitY()
            )
        );

        const auto measurement = ideal_sensor.Measure(
            state,
            Eigen::Vector3d::Zero()
        );

        assert(Near(measurement.acceleration.x(), -g));
        assert(Near(measurement.acceleration.y(), 0.0));
        assert(Near(measurement.acceleration.z(), 0.0));

        state.orientation = Eigen::Quaterniond::Identity();
    }

    // Test 5: Constant accelerometer bias.
    {
        sensors::SensorParameters parameters;

        parameters.accelerometer_bias =
            Eigen::Vector3d(0.1, -0.2, 0.3);

        sensors::SensorModel sensor(parameters);

        const auto measurement = sensor.Measure(state);

        assert(Near(measurement.acceleration.x(), 0.1));
        assert(Near(measurement.acceleration.y(), -0.2));
        assert(Near(measurement.acceleration.z(), g + 0.3));
    }

    // Test 6: Noise is deterministic for identical seeds.
    {
        sensors::SensorParameters parameters;

        parameters.accelerometer_noise =
            Eigen::Vector3d(0.1, 0.1, 0.1);

        sensors::SensorModel first(parameters, 1234);
        sensors::SensorModel second(parameters, 1234);

        for (int i = 0; i < 100; ++i) {
            const auto a = first.Measure(state);
            const auto b = second.Measure(state);

            assert(a.acceleration.isApprox(b.acceleration, 1e-12));
        }

        first.Reset(1234);
        second.Reset(1234);

        assert(
            first.Measure(state).acceleration.isApprox(
                second.Measure(state).acceleration,
                1e-12
            )
        );
    }

    std::cout << "Accelerometer tests PASSED\n";

    return 0;
}
