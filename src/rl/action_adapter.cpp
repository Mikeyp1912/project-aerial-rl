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
}
