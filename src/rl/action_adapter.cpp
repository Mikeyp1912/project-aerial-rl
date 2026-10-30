// /src/rl/action_adapter.cpp

#include "aerial/rl/action_adapter.hpp"
#include "aerial/physics/motor_mixer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aerial::rl
{
    physics::MotorCommands DirectMotorActionAdapter::Convert(
            const Action& action) const
    {
        physics::MotorCommands commands{};

        for (std::size_t i = 0; i < kActionSize; ++i) {
            const double value = action[i];

            if (!std::isfinite(value)) {
                throw std::invalid_argument("RL action values must be finite");
            }

            const double normalized = std::clamp(value, -1.0, 1.0);

            commands[i] = 0.5 * (normalized + 1.0);
        }

        return commands;
    }

    BodyRateActionAdapter::BodyRateActionAdapter(
            const BodyRateActionParameters& parameters
            )
        :
            parameters_(parameters)
    {
        if (!parameters_.maximum_rates.allFinite() ||
            (parameters_.maximum_rates.array() <= 0.0).any() ||
            !std::isfinite(parameters_.maximum_thrust_fraction) ||
            parameters_.maximum_thrust_fraction <= 0.0 ||
            parameters_.maximum_thrust_fraction > 1.0)
        {
            throw std::invalid_argument("Invalid body-rate action parameters");
        }

        Reset();
    }

    void BodyRateActionAdapter::Reset()
    {
        control::ResetRateController(controller_state_);
    }

    physics::MotorCommands BodyRateActionAdapter::Convert(
            const Action& action,
            const sensors::SensorMeasurements& measurements,
            const physics::DroneParameters& drone_parameters,
            double dt
            )
    {
        for (double value : action) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("RL action values must be finite");
            }
        }

        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::invalid_argument("Invalid controller time step");
        }

        const Eigen::Vector3d desired_rates{
            std::clamp(action[0], -1.0, 1.0) * parameters_.maximum_rates.x(),
            std::clamp(action[1], -1.0, 1.0) * parameters_.maximum_rates.y(),
            std::clamp(action[2], -1.0, 1.0) * parameters_.maximum_rates.z()
        };

        const double thrust_fraction =
            0.5 * (std::clamp(action[3], -1.0, 1.0) + 1.0) *
                    parameters_.maximum_thrust_fraction;

        const double collective_thrust =
            thrust_fraction *
            4.0 * drone_parameters.max_motor_thrust;

        const Eigen::Vector3d body_torque =
            control::CalculateBodyTorque(
                    desired_rates,
                    measurements.angular_velocity,
                    dt,
                    parameters_.rate_controller,
                    controller_state_
                    );

        return physics::MixMotors(
                drone_parameters,
                collective_thrust,
                body_torque
                ).commands;
    }

    PIDTargetActionAdapter::PIDTargetActionAdapter(
            const PIDTargetActionParameters& parameters
            )
        :
            parameters_(parameters)
    {
        if (!std::isfinite(parameters_.maximum_horizontal_offset) ||
            parameters_.maximum_horizontal_offset <= 0.0 ||
            !std::isfinite(parameters_.minimum_altitude) ||
            !std::isfinite(parameters_.maximum_altitude) ||
            parameters_.minimum_altitude < 0.0 ||
            parameters_.maximum_altitude <= parameters_.minimum_altitude ||
            !parameters_.world_target_position.allFinite() ||
            (parameters_.reference != PIDTargetReference::EpisodeRelative &&
             parameters_.reference != PIDTargetReference::WorldRelative))
        {
            throw std::invalid_argument(
                "Invalid PID-target action parameters"
            );
        }
    }

    void PIDTargetActionAdapter::Reset(
            const sensors::SensorMeasurements& measurements
            )
    {
        if (!measurements.position.allFinite())
        {
            throw std::invalid_argument("Invalid intial position measurement");
        }

        episode_origin_ = measurements.position.head<2>();

        control::ResetFlightController(controller_state_);

        initialized_ = true;
    }

    physics::MotorCommands PIDTargetActionAdapter::Convert(
            const Action& action,
            const sensors::SensorMeasurements& measurements,
            const physics::DroneParameters& drone_parameters,
            double dt
            )
    {
        if (!initialized_) {
            throw std::logic_error("PID-target adapter must be reset before use");
        }

        for (double value : action) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("RL action values must be finite");
            }
        }

        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::invalid_argument("Invalid controller time step");
        }

        if (!measurements.position.allFinite() ||
            !measurements.velocity.allFinite() ||
            !measurements.angular_velocity.allFinite() ||
            !measurements.orientation.coeffs().allFinite()) {
            throw std::invalid_argument(
                "Invalid sensor measurements"
            );
        }

        const Eigen::Vector2d target_position =
            CalculatePIDHorizontalTarget(
                parameters_,
                episode_origin_,
                action
            );

        const double altitude_fraction =
            0.5 * (std::clamp(action[2], -1.0, 1.0) + 1.0);

        const double target_altitude =
            parameters_.minimum_altitude +
            altitude_fraction *
            (parameters_.maximum_altitude -
             parameters_.minimum_altitude);

        constexpr double pi = 3.14159265358979323846;

        const double target_yaw =
            std::clamp(action[3], -1.0, 1.0) * pi;

        // contruct a state estimate exclusively from sensors
        physics::DroneState estimated_state;

        estimated_state.position = measurements.position;
        estimated_state.velocity = measurements.velocity;
        estimated_state.orientation = measurements.orientation;
        estimated_state.angular_velocity = measurements.angular_velocity;

        // Position -> desired horizontal velocity
        const auto position_output =
            control::UpdateHorizontalPositionController(
                    estimated_state,
                    target_position,
                    parameters_.position
                    );

        // Horizontal velocity -> desired roll and pitch
        const auto velocity_output =
            control::UpdateHorizontalVelocityController(
                    estimated_state,
                    position_output.desired_velocity,
                    target_yaw,
                    parameters_.velocity
                    );

        // Altitude -> collective thrust
        const auto altitude_output =
            control::UpdateAltitudeController(
                    estimated_state,
                    drone_parameters,
                    target_altitude,
                    parameters_.altitude
                    );

        // Desired roll, pitch and yaw -> target orientation
        const Eigen::Quaterniond target_orientation =
            Eigen::AngleAxisd(
                    target_yaw, Eigen::Vector3d::UnitZ()
                    ) *
            Eigen::AngleAxisd(
                    velocity_output.target_pitch,
                    Eigen::Vector3d::UnitY()
                    ) *
            Eigen::AngleAxisd(
                    velocity_output.target_roll,
                    Eigen::Vector3d::UnitX()
                    );

        // Attitude -> body rates -> body torque -> motors
        const auto flight_output =
            control::UpdateFlightController(
                    estimated_state,
                    drone_parameters,
                    target_orientation,
                    altitude_output.collective_thrust,
                    dt,
                    parameters_.flight,
                    controller_state_
                    );

        return flight_output.motor_commands;
    }

    Eigen::Vector2d CalculatePIDHorizontalTarget(
        const PIDTargetActionParameters& parameters,
        const Eigen::Vector2d& episode_origin,
        const Action& action
    )
    {
        if (!episode_origin.allFinite() ||
            !parameters.world_target_position.allFinite() ||
            !std::isfinite(parameters.maximum_horizontal_offset) ||
            parameters.maximum_horizontal_offset <= 0.0 ||
            (parameters.reference != PIDTargetReference::EpisodeRelative &&
             parameters.reference != PIDTargetReference::WorldRelative)) {
            throw std::invalid_argument("Invalid PID target configuration");
        }

        for (double value : action) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Invalid PID action");
            }
        }

        const Eigen::Vector2d reference_position =
            parameters.reference == PIDTargetReference::WorldRelative
                ? parameters.world_target_position
                : episode_origin;

        return reference_position +
            parameters.maximum_horizontal_offset *
            Eigen::Vector2d(
                std::clamp(action[0], -1.0, 1.0),
                std::clamp(action[1], -1.0, 1.0)
            );
    }
}
