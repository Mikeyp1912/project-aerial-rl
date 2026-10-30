// /include/aerial/rl/flight_environment.hpp

#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "aerial/physics/simulation_environment.hpp"
#include "aerial/sensors/sensor_scheduler.hpp"
#include "aerial/rl/rl_types.hpp"
#include "aerial/rl/action_adapter.hpp"
#include "aerial/rl/hover_reward.hpp"
#include "aerial/rl/episode_termination.hpp"
#include "aerial/rl/episode_statistics.hpp"
#include "aerial/rl/observation_normalizer.hpp"

namespace aerial::rl
{
    struct InitialStateRandomizationParameters
    {
        bool enabled = false;

        Eigen::Vector3d maximum_position_offset{
            0.5, 0.5, 0.2
        };

        Eigen::Vector3d maximum_velocity{
            0.2, 0.2, 0.0
        };
    };

    struct WindRandomizationParameters
    {
        bool enabled = false;

        // Maximum horizontal wind speed in m/s.
        double maximum_horizontal_speed = 5.0;
    };

    struct FlightEnvironmentParameters
    {
        double time_step = 1.0 / 120.0;
        double maximum_episode_time = 20.0;

        ActionMode action_mode = ActionMode::DirectMotor;

        BodyRateActionParameters body_rate;

        PIDTargetActionParameters pid_targets;

        HoverRewardParameters hover_reward;

        EpisodeTerminationParameters termination;

        ObservationNormalizationParameters observation_normalization;

        InitialStateRandomizationParameters initial_state_randomization;

        WindRandomizationParameters wind_randomization;
    };

    struct FlightStepResult
    {
        sensors::SensorMeasurements observation;

        double reward = 0.0;
        HoverRewardComponents reward_components;

        TerminationReason termination_reason = TerminationReason::None;

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

            const EpisodeStatistics& GetEpisodeStatistics() const;

            Observation GetNormalizedRLObservation() const;

            const Eigen::Vector3d& GetWindVelocity() const;

        private:
            FlightEnvironmentParameters parameters_;

            physics::Simulation simulation_;
            sensors::SensorScheduler sensors_;

            DirectMotorActionAdapter direct_motor_adapter_;

            BodyRateActionAdapter body_rate_adapter_;

            PIDTargetActionAdapter pid_target_adapter_;

            EpisodeStatisticsTracker statistics_tracker_;

            bool episode_finished_ = false;
    };
}
