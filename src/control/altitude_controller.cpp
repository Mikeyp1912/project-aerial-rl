// /src/control/altitude_controller.cpp

#include "aerial/control/altitude_controller.hpp"

#include <algorithm>

#include "aerial/physics/integrator.hpp"

namespace aerial::control
{
    AltitudeControllerOutput UpdateAltitudeController(
            const physics::DroneState& drone_state,
            const physics::DroneParameters& drone_parameters,
            double target_altitude,
            const AltitudeControllerParameters& parameters
            )
    {
        AltitudeControllerOutput output;

        const double altitude_error = target_altitude - drone_state.position.z();

        output.desired_vertical_velocity = std::clamp(
                parameters.position_kp * altitude_error,
                -parameters.max_vertical_velocity,
                parameters.max_vertical_velocity
                );

        const double velocity_error = output.desired_vertical_velocity - drone_state.velocity.z();

        output.desired_vertical_acceleration = std::clamp(
                parameters.velocity_kp * velocity_error,
                -parameters.max_vertical_acceleration,
                parameters.max_vertical_acceleration
                );

        // Body +Z axis expressed in world coordinates
        const Eigen::Vector3d thrust_direction = drone_state.orientation * Eigen::Vector3d::UnitZ();

        const double vertical_fraction = thrust_direction.z();

        // Do not attempt full tilt compensation near horizontal
        // or when the drone is inverted
        const double safe_vertical_fraction = std::max(
                vertical_fraction,
                parameters.minimum_vertical_thrust_fraction
                );

        const double required_vertical_force = drone_parameters.mass * (physics::gravity + output.desired_vertical_acceleration);

        const double requested_thrust = required_vertical_force / safe_vertical_fraction;

        const double maximum_collective_thrust = 4.0 * drone_parameters.max_motor_thrust;

        output.collective_thrust = std::clamp(
                requested_thrust,
                0.0,
                maximum_collective_thrust
                );

        output.saturated = 
            vertical_fraction < parameters.minimum_vertical_thrust_fraction ||
            requested_thrust > maximum_collective_thrust ||
            requested_thrust < 0.0;

        return output;
    }
}
