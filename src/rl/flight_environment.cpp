// /src/rl/flight_environment.cpp

#include "aerial/rl/flight_environment.hpp"

#include <cmath>
#include <stdexcept>

#include "aerial/rl/observation.hpp"

namespace aerial::rl 
{
    FlightEnvironment::FlightEnvironment(
            const FlightEnvironmentParameters& parameters
            )
        :
            parameters_(parameters),
            body_rate_adapter_(parameters.body_rate),
            pid_target_adapter_(parameters.pid_targets)
    {
        if (!std::isfinite(parameters_.time_step) ||
            parameters_.time_step <= 0.0 ||
            !std::isfinite(parameters_.maximum_episode_time) ||
            parameters_.maximum_episode_time <= 0.0) {
            throw std::invalid_argument(
                    "Invalid flight environment parameters"
                    );
        }

        Reset();
    }

    void FlightEnvironment::Reset(std::uint32_t seed)
    {
        simulation_.Reset();
        sensors_.Reset(seed);

        body_rate_adapter_.Reset();

        episode_finished_ = false;

        sensors_.Update(
                simulation_.GetDroneState(),
                simulation_.GetWorldAcceleration(),
                simulation_.GetSimulationTime()
                );

        pid_target_adapter_.Reset(sensors_.GetMeasurements());
    }

    FlightStepResult FlightEnvironment::Step(
            const physics::MotorCommands& commands
            )
    {
        if (episode_finished_) {
            throw std::logic_error("Cannot step a finished episode; call Reset()");
        }

        for (const double command : commands) {
            if (!std::isfinite(command)) {
                throw std::invalid_argument("Motor commands must be finite");
            }
        }

        simulation_.Step(commands, parameters_.time_step);

        sensors_.Update(
                simulation_.GetDroneState(),
                simulation_.GetWorldAcceleration(),
                simulation_.GetSimulationTime()
                );

        FlightStepResult result;
        result.observation = sensors_.GetMeasurements();

        // Reward and physical termination will be added later
        result.reward = 0.0;
        result.terminated = false;

        result.truncated =
            simulation_.GetSimulationTime() + 1e-9 >=
            parameters_.maximum_episode_time;

        episode_finished_ = result.terminated || result.truncated;

        return result;
    }

    FlightStepResult FlightEnvironment::StepAction(const Action& action)
    {
        if (episode_finished_) {
            throw std::logic_error("Cannot step a finished episode; call Reset()");
        }

        switch (parameters_.action_mode) {
            case ActionMode::DirectMotor:
                {
                    return Step(direct_motor_adapter_.Convert(action));
                }

            case ActionMode::BodyRate:
                {
                    return Step(body_rate_adapter_.Convert(
                                action,
                                sensors_.GetMeasurements(),
                                simulation_.GetParameters(),
                                parameters_.time_step
                                ));
                }
            case ActionMode::PIDTargets:
                {
                    return Step(pid_target_adapter_.Convert(
                                action,
                                sensors_.GetMeasurements(),
                                simulation_.GetParameters(),
                                parameters_.time_step
                                ));
                }
        }

        throw std::invalid_argument("Unknown RL action mode");
    }

    const sensors::SensorMeasurements& FlightEnvironment::GetObservation() const
    {
        return sensors_.GetMeasurements();
    }

    Observation FlightEnvironment::GetRLObservation() const
    {
        return MakeObservation(sensors_.GetMeasurements());
    }

    const physics::DroneState& FlightEnvironment::GetTrueState() const
    {
        return simulation_.GetDroneState();
    }

    double FlightEnvironment::GetSimulationTime() const
    {
        return simulation_.GetSimulationTime();
    }

    ActionMode FlightEnvironment::GetActionMode() const
    {
        return parameters_.action_mode;
    }
}
