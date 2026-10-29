// /include/aerial/rl/action_adapter.hpp

#pragma once

#include "aerial/physics/motor_commands.hpp"
#include "aerial/rl/rl_types.hpp"
#include "aerial/control/rate_controller.hpp"
#include "aerial/physics/drone_parameters.hpp"
#include "aerial/sensors/sensor_model.hpp"

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
}
