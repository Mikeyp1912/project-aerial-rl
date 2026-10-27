// /include/aerial/rl/flight_environment.hpp

#pragma once

#include <cstdint>

#include "aerial/physics/simulation_environment.hpp"
#include "aerial/sensors/sensor_scheduler.hpp"

namespace aerial::rl
{
    struct FlightEnvironmentParameters
    {
        double time_step = 1.0 / 120.0;
        double maximum_episode_time = 20.0;
    };

    struct FlightStepResult
    {
        sensors::SensorMeasurements observation;

        double reward = 0.0;

        bool terminated = false;
        bool truncated = false;
    };

    class FlightEnvironment
    {
        public:
            explicit FlightEnvironment(
                    const FlightEnvironmentParameters& parameters = {}
                    );

            void Reset(std::uint32_t seed = 42);

            FlightStepResult Step(
                    const physics::MotorCommands& commands
                    );

            const sensors::SensorMeasurements& GetObservation() const;

            const physics::DroneState& GetTrueState() const;

            double GetSimulationTime() const;

        private:
            FlightEnvironmentParameters parameters_;

            physics::Simulation simulation_;
            sensors::SensorScheduler sensors_;

            bool episode_finished_ = false;
    };
}
