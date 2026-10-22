// /tests/flight_controller_test.cpp

#include "aerial/control/flight_controller.hpp"

#include "aerial/physics/motor_commands.hpp"
#include "aerial/physics/motor_model.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace aerial::control;
using namespace aerial::physics;

namespace
{
    constexpr double tolerance = 1e-9;
    constexpr double dt = 0.01;

    bool NearlyEqual(double a, double b)
    {
        return std::abs(a - b) < tolerance;
    }

    Eigen::Quaterniond Rotation(
            const Eigen::Vector3d& axis,
            double angle
            )
    {
        return Eigen::Quaterniond(
                Eigen::AngleAxisd(angle, axis)
                );
    }

    MotorOutput GetMotorOutput(
            const DroneParameters& parameters,
            const FlightControllerOutput& output
            )
    {
        const auto thrusts = CommandsToThrust(
                parameters,
                output.motor_commands
                );

        return CalculateMotorOutput(parameters, thrusts);
    }

    FlightControllerParameters TestControllerParameters()
    {
        FlightControllerParameters parameters;

        parameters.attitude.kp = 4.0;
        parameters.attitude.max_angular_rate = 4.0;

        parameters.rate.kp =
            Eigen::Vector3d(0.1, 0.1, 0.05);

        parameters.rate.ki.setZero();
        parameters.rate.kd.setZero();

        parameters.rate.torque_limit =
            Eigen::Vector3d(0.5, 0.5, 0.1);

        return parameters;
    }
}

int main()
{
    DroneParameters drone_parameters;
    FlightControllerParameters controller_parameters =
        TestControllerParameters();

    const Eigen::Quaterniond identity =
        Eigen::Quaterniond::Identity();

    const double hover_thrust =
        CalculateHoverThrust(drone_parameters);

    // 1. Hover thrust equals mass * gravity.
    {
        assert(NearlyEqual(hover_thrust, 9.81));
    }

    // 2. Level hover produces equal motor commands.
    {
        DroneState drone;
        FlightControllerState state;

        const auto output = UpdateFlightController(
                drone,
                drone_parameters,
                identity,
                hover_thrust,
                dt,
                controller_parameters,
                state
                );

        assert(!output.motor_saturated);

        for (double command : output.motor_commands) {
            assert(NearlyEqual(command, 0.4905));
        }

        assert(output.desired_angular_rate.norm() < tolerance);
        assert(output.requested_body_torque.norm() < tolerance);
    }

    // 3. Positive roll target produces positive roll torque.
    {
        DroneState drone;
        FlightControllerState state;

        const auto output = UpdateFlightController(
                drone,
                drone_parameters,
                Rotation(Eigen::Vector3d::UnitX(), 0.1),
                hover_thrust,
                dt,
                controller_parameters,
                state
                );

        const auto motors = GetMotorOutput(
                drone_parameters, output
                );

        assert(output.desired_angular_rate.x() > 0.0);
        assert(output.requested_body_torque.x() > 0.0);
        assert(motors.torque.x() > 0.0);
        assert(!output.motor_saturated);
    }

    // 4. Positive pitch target produces positive pitch torque.
    {
        DroneState drone;
        FlightControllerState state;

        const auto output = UpdateFlightController(
                drone,
                drone_parameters,
                Rotation(Eigen::Vector3d::UnitY(), 0.1),
                hover_thrust,
                dt,
                controller_parameters,
                state
                );

        const auto motors = GetMotorOutput(
                drone_parameters, output
                );

        assert(output.requested_body_torque.y() > 0.0);
        assert(motors.torque.y() > 0.0);
    }

    // 5. Positive yaw target produces positive yaw torque.
    {
        DroneState drone;
        FlightControllerState state;

        const auto output = UpdateFlightController(
                drone,
                drone_parameters,
                Rotation(Eigen::Vector3d::UnitZ(), 0.1),
                hover_thrust,
                dt,
                controller_parameters,
                state
                );

        const auto motors = GetMotorOutput(
                drone_parameters, output
                );

        assert(output.requested_body_torque.z() > 0.0);
        assert(motors.torque.z() > 0.0);
    }

    // 6. Existing angular velocity produces opposing torque.
    {
        DroneState drone;
        FlightControllerState state;

        drone.angular_velocity =
            Eigen::Vector3d(0.5, 0.0, 0.0);

        const auto output = UpdateFlightController(
                drone,
                drone_parameters,
                identity,
                hover_thrust,
                dt,
                controller_parameters,
                state
                );

        assert(output.requested_body_torque.x() < 0.0);
    }

    // 7. Motor saturation is reported.
    {
        DroneState drone;
        FlightControllerState state;

        const auto output = UpdateFlightController(
                drone,
                drone_parameters,
                identity,
                30.0,
                dt,
                controller_parameters,
                state
                );

        assert(output.motor_saturated);

        for (double command : output.motor_commands) {
            assert(command >= 0.0);
            assert(command <= 1.0);
        }
    }

    // 8. Reset clears internal controller state.
    {
        DroneState drone;
        FlightControllerState state;

        UpdateFlightController(
                drone,
                drone_parameters,
                identity,
                hover_thrust,
                dt,
                controller_parameters,
                state
                );

        assert(state.rate.initialized);

        ResetFlightController(state);

        assert(!state.rate.initialized);
        assert(state.rate.integral_error.norm() < tolerance);
    }

    // 9. Invalid time step is rejected.
    {
        DroneState drone;
        FlightControllerState state;

        bool threw = false;

        try {
            UpdateFlightController(
                    drone,
                    drone_parameters,
                    identity,
                    hover_thrust,
                    0.0,
                    controller_parameters,
                    state
                    );
        }
        catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    return 0;
}
