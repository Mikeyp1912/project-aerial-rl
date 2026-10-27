// /tests/physics_integrated_imu_test.cpp
#include <cassert>
#include <cmath>
#include <iostream>

#include "aerial/physics/integrator.hpp"
#include "aerial/physics/simulation_environment.hpp"
#include "aerial/sensors/sensor_model.hpp"

int main()
{
    using namespace aerial;

    constexpr double dt = 1.0 / 120.0;
    constexpr double tolerance = 1e-9;

    physics::Simulation simulation;
    sensors::SensorModel sensor;

    const physics::MotorCommands commands{};

    // Test 1: No thrust -> free fall.
    simulation.Step(commands, dt);

    assert(std::abs(
        simulation.GetWorldAcceleration().z() +
        physics::gravity
    ) < tolerance);

    const auto free_fall = sensor.Measure(
        simulation.GetDroneState(),
        simulation.GetWorldAcceleration()
    );

    assert(free_fall.acceleration.norm() < tolerance);

    // Test 2: External horizontal force.
    simulation.Reset();

    simulation.SetExternalForce(Eigen::Vector3d(2.0, 0.0, 0.0));

    simulation.Step(commands, dt);

    const auto disturbed = sensor.Measure(
        simulation.GetDroneState(),
        simulation.GetWorldAcceleration()
    );

    const double expected_x =
        2.0 / simulation.GetParameters().mass;

    assert(std::abs(
        disturbed.acceleration.x() - expected_x
    ) < tolerance);

    assert(std::abs(disturbed.acceleration.z()) < tolerance);

    // Test 3: Reset clears the last acceleration.
    simulation.Reset();

    assert(simulation.GetWorldAcceleration().isZero());

    std::cout << "Physics-integrated IMU tests PASSED\n";

    return 0;
}
