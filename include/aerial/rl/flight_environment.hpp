// /include/aerial/rl/flight_environment.hpp

#pragma once

#include <cstdint>

#include "aerial/physics/simulation_environment.hpp"
#include "aerial/sensors/sensor_scheduler.hpp"
#include "aerial/rl/rl_types.hpp"
#include "aerial/rl/action_adapter.hpp"

namespace aerial::rl
{
    struct FlightEnvironmentParameters
    {
        double time_step = 1.0 / 120.0;
        double maximum_episode_time = 20.0;

        ActionMode action_mode = ActionMode::DirectMotor;
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

            // Neural-network action interface
            FlightStepResult StepAction(const Action& action);

            const sensors::SensorMeasurements& GetObservation() const;

            // Fixed-size observation for neural-network input
            Observation GetRLObservation() const;

            const physics::DroneState& GetTrueState() const;

            double GetSimulationTime() const;

            ActionMode GetActionMode() const;

        private:
            FlightEnvironmentParameters parameters_;

            physics::Simulation simulation_;
            sensors::SensorScheduler sensors_;

            DirectMotorActionAdapter direct_motor_adapter_;

            bool episode_finished_ = false;
    };
}
