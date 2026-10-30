// /src/rl/flight_environment.cpp

#include "aerial/rl/flight_environment.hpp"

#include <cmath>
#include <stdexcept>
#include <random>

#include "aerial/physics/drone_state.hpp"
#include "aerial/rl/observation.hpp"

namespace aerial::rl 
{
    FlightEnvironment::FlightEnvironment(
            const FlightEnvironmentParameters& parameters
            )
        :
            parameters_(parameters),
            body_rate_adapter_(parameters.body_rate),
            pid_target_adapter_([&parameters] {
                auto pid_parameters = parameters.pid_targets;

                if (pid_parameters.reference ==
                    PIDTargetReference::WorldRelative) {
                    pid_parameters.world_target_position =
                        parameters.hover_reward.target_position.head<2>();
                }

                if (pid_parameters.hover_centered_altitude) {
                    pid_parameters.hover_altitude =
                        parameters.hover_reward.target_position.z();
                }

                return pid_parameters;
            }())
    {
        if (!std::isfinite(parameters_.time_step) ||
            parameters_.time_step <= 0.0 ||
            !std::isfinite(parameters_.maximum_episode_time) ||
            parameters_.maximum_episode_time <= 0.0) {
            throw std::invalid_argument(
                    "Invalid flight environment parameters"
                    );
        }

        ValidateEpisodeTerminationParameters(parameters_.termination);

        ValidateObservationNormalizationParameters(parameters_.observation_normalization);

        const auto& randomization = parameters_.initial_state_randomization;

        if (!randomization.maximum_position_offset.allFinite() ||
            !randomization.maximum_velocity.allFinite() ||
            (randomization.maximum_position_offset.array() < 0.0).any() ||
            (randomization.maximum_velocity.array() < 0.0).any()) {
            throw std::invalid_argument(
                "Invalid initial-state randomization parameters"
            );
        }

        Reset();
    }

    

    void FlightEnvironment::Reset(std::uint32_t seed)
    {
        const auto& randomization =
            parameters_.initial_state_randomization;

        if (randomization.enabled) {
            std::mt19937 generator(seed);

            std::uniform_real_distribution<double> distribution(
                -1.0, 1.0
            );

            physics::DroneState initial_state;

            initial_state.position = Eigen::Vector3d(
                0.0, 0.0, 1.0
            );

            for (int axis = 0; axis < 3; ++axis) {
                initial_state.position[axis] +=
                    distribution(generator) *
                    randomization.maximum_position_offset[axis];

                initial_state.velocity[axis] =
                    distribution(generator) *
                    randomization.maximum_velocity[axis];
            }

            simulation_.Reset(initial_state);
        }
        else {
            simulation_.Reset();
        }

        sensors_.Reset(seed);
        body_rate_adapter_.Reset();

        episode_finished_ = false;
        statistics_tracker_.Reset();

        sensors_.Update(
            simulation_.GetDroneState(),
            simulation_.GetWorldAcceleration(),
            simulation_.GetSimulationTime()
        );

        pid_target_adapter_.Reset(
            sensors_.GetMeasurements()
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

        const auto& state = simulation_.GetDroneState();

        const bool state_is_finite =
            state.position.allFinite() &&
            state.velocity.allFinite() &&
            state.angular_velocity.allFinite() &&
            state.orientation.coeffs().allFinite() &&
            state.orientation.norm() >= 1e-12;

        if (state_is_finite) {
            sensors_.Update(
                state,
                simulation_.GetWorldAcceleration(),
                simulation_.GetSimulationTime()
            );
        }

        FlightStepResult result;
        result.observation = sensors_.GetMeasurements();

        const auto& true_state = simulation_.GetDroneState();

        const auto termination = CheckEpisodeTermination(
                true_state,
                parameters_.termination,
                parameters_.hover_reward.target_position
                );

        result.terminated = termination.terminated;
        result.termination_reason = termination.reason;

        if (termination.reason == TerminationReason::NonFiniteState) {
            // Do not evaluate the quadratic reward on invalid state data
            result.reward = parameters_.termination.failure_penalty;
        }
        else {
            result.reward_components = CalculateHoverReward(
                    true_state,
                    parameters_.hover_reward
                    );

            result.reward = result.reward_components.total;

            if (result.terminated) {
                result.reward += parameters_.termination.failure_penalty;
            }
        }
        
        result.truncated =
            simulation_.GetSimulationTime() + 1e-9 >=
            parameters_.maximum_episode_time;

        statistics_tracker_.RecordStep(
                true_state,
                parameters_.hover_reward.target_position,
                result.reward,
                simulation_.GetSimulationTime(),
                result.terminated,
                result.truncated,
                result.termination_reason
                );

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

    const EpisodeStatistics& FlightEnvironment::GetEpisodeStatistics() const
    {
        return statistics_tracker_.GetStatistics();
    }

    Observation FlightEnvironment::GetNormalizedRLObservation() const
    {
        return NormalizeObservation(
                GetRLObservation(),
                parameters_.observation_normalization
                );
    }
}
