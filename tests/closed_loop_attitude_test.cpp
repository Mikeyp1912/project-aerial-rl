// /tests/closed_loop_attitude_test.cpp


#include "aerial/control/flight_controller.hpp"
#include "aerial/physics/simulation.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace aerial::control;
using namespace aerial::physics;

namespace
{
    constexpr double pi = 3.14159265358979323846;
    constexpr double dt = 0.002;
    constexpr int steps = 2000;

    double OrientationError(
            const Eigen::Quaterniond& orientation,
            const Eigen::Quaterniond& target
            )
    {
        Eigen::Quaterniond error =
            orientation.normalized().conjugate() *
            target.normalized();

        return Eigen::AngleAxisd(error).angle();
    }
}

int main()
{
    DroneParameters drone_parameters;
    DroneState drone;
    MotorState motors;

    FlightControllerParameters controller_parameters;
    FlightControllerState controller_state;

    const Eigen::Quaterniond target =
        Eigen::Quaterniond::Identity();

    // Start with a 15-degree roll disturbance.
    drone.orientation = Eigen::Quaterniond(
            Eigen::AngleAxisd(
                15.0 * pi / 180.0,
                Eigen::Vector3d::UnitX()
                )
            );

    // Start the motors at their steady-state hover thrust.
    // This isolates attitude response from motor spin-up.
    const double hover_thrust =
        CalculateHoverThrust(drone_parameters);

    motors.thrusts.fill(hover_thrust / 4.0);

    const double initial_error =
        OrientationError(drone.orientation, target);

    double maximum_error = initial_error;
    int saturated_steps = 0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Time(s),Roll(deg),Error(deg),Rate(rad/s)\n";

    for (int step = 0; step < steps; ++step) {
        const auto output = UpdateFlightController(
                drone,
                drone_parameters,
                target,
                hover_thrust,
                dt,
                controller_parameters,
                controller_state
                );

        if (output.motor_saturated) {
            ++saturated_steps;
        }

        StepSimulation(
                drone,
                motors,
                drone_parameters,
                output.motor_commands,
                dt
                );

        assert(drone.orientation.coeffs().allFinite());
        assert(drone.angular_velocity.allFinite());

        const double error =
            OrientationError(drone.orientation, target);

        maximum_error = std::max(maximum_error, error);

        if (step % 100 == 0 || step == steps - 1) {
            // atan2 provides a signed roll angle.
            const Eigen::Matrix3d rotation =
                drone.orientation.toRotationMatrix();

            const double roll = std::atan2(
                    rotation(2, 1),
                    rotation(2, 2)
                    );

            std::cout
                << (step + 1) * dt << ","
                << roll * 180.0 / pi << ","
                << error * 180.0 / pi << ","
                << drone.angular_velocity.norm()
                << "\n";
        }
    }

    const double final_error =
        OrientationError(drone.orientation, target);

    const double final_rate =
        drone.angular_velocity.norm();

    std::cout << "\nFinal attitude error: "
              << final_error * 180.0 / pi << " deg\n";

    std::cout << "Final angular rate: "
              << final_rate << " rad/s\n";

    std::cout << "Maximum attitude error: "
              << maximum_error * 180.0 / pi << " deg\n";

    std::cout << "Saturated steps: "
              << saturated_steps << "/" << steps << "\n";

    // Require substantial recovery from the initial disturbance.
    assert(final_error < 1.0 * pi / 180.0);

    // Require the drone to be nearly stationary in rotation.
    assert(final_rate < 0.05);

    std::cout << "Closed-loop attitude stabilization PASSED\n";

    return 0;
}
