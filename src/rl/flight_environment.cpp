// /src/rl/flight_environment.cpp

#include "aerial/rl/flight_environment.hpp"

#include <cmath>
#include <stdexcept>

namespace aerial::rl 
{
    FlightEnvironment::FlightEnvironment(
            const FlightEnvironmentParameters& parameters
            )
        :
            parameters_(parameters)
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

        episode_finished_ = false;

        sensors_.Update(
                simulation_.GetDroneState(),
                simulation_.GetWorldAcceleration(),
                simulation_.GetSimulationTime()
                );
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

    const sensors::SensorMeasurements& FlightEnvironment::GetObservation() const
    {
        return sensors_.GetMeasurements();
    }

    const physics::DroneState& FlightEnvironment::GetTrueState() const
    {
        return simulation_.GetDroneState();
    }

    double FlightEnvironment::GetSimulationTime() const
    {
        return simulation_.GetSimulationTime();
    }
}
