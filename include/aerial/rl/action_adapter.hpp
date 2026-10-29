// /include/aerial/rl/action_adapter.hpp

#pragma once

#include "aerial/physics/motor_commands.hpp"
#include "aerial/rl/rl_types.hpp"
#include "aerial/control/rate_controller.hpp"
#include "aerial/physics/drone_parameters.hpp"
#include "aerial/sensors/sensor_model.hpp"
#include "aerial/control/altitude_controller.hpp"
#include "aerial/control/horizontal_position_controller.hpp"
#include "aerial/control/horizontal_velocity_controller.hpp"
#include "aerial/control/flight_controller.hpp"

namespace aerial::rl
{
    class DirectMotorActionAdapter
    {
        public:
            physics::MotorCommands Convert(
                    const Action& action
                    ) const;
    };

    struct BodyRateActionParameters
    {
        // Maximum commanded angular rates (rad/s)
        Eigen::Vector3d maximum_rates{3.0, 3.0, 2.0};

        // Maximum collective thrust as a fraction of
        // the drone's total available thrust
        double maximum_thrust_fraction = 1.0;

        control::RateControllerParameters rate_controller;
    };

    class BodyRateActionAdapter
    {
        public:
            explicit BodyRateActionAdapter(
                    const BodyRateActionParameters& parameters = {}
                    );

            void Reset();

            physics::MotorCommands Convert(
                    const Action& action,
                    const sensors::SensorMeasurements& measurements,
                    const physics::DroneParameters& drone_parameters,
                    double dt
                    );

        private:
            BodyRateActionParameters parameters_;
            control::RateControllerState controller_state_;
    };

    struct PIDTargetActionParameters
    {
        // Maximum XY displacement relative to the episode origin (m)
        double maximum_horizontal_offset = 5.0;

        // Absolute altitude target range (m)
        double minimum_altitude = 0.5;
        double maximum_altitude = 5.0;

        control::HorizontalPositionControllerParameters position;
        control::HorizontalVelocityControllerParameters velocity;
        control::AltitudeControllerParameters altitude;
        control::FlightControllerParameters flight;
    };

    class PIDTargetActionAdapter
    {
        public:
            explicit PIDTargetActionAdapter(
                    const PIDTargetActionParameters& parameters = {}
                    );

            void Reset(const sensors::SensorMeasurements& measurements);

            physics::MotorCommands Convert(
                    const Action& action,
                    const sensors::SensorMeasurements& measurements,
                    const physics::DroneParameters& drone_parameters,
                    double dt
                    );

        private:
            PIDTargetActionParameters parameters_;
            control::FlightControllerState controller_state_;

            Eigen::Vector2d episode_origin_ = Eigen::Vector2d::Zero();

            bool initialized_ = false;
    };
}
