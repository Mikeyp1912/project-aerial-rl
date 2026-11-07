
#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aerial/rl/flight_environment.hpp"

namespace py = pybind11;

namespace
{
    class PythonFlightEnvironment
    {
    public:
        explicit PythonFlightEnvironment(
            double time_step = 1.0 / 120.0,
            double maximum_episode_time = 20.0,
            aerial::rl::ActionMode action_mode =
                aerial::rl::ActionMode::PIDTargets,
            bool randomize_initial_state = false,
            std::array<double, 3> maximum_position_offset =
                {0.5, 0.5, 0.2},
            std::array<double, 3> maximum_velocity =
                {0.2, 0.2, 0.0},
            aerial::rl::PIDTargetReference pid_target_reference =
                aerial::rl::PIDTargetReference::EpisodeRelative,
            double maximum_horizontal_offset = 5.0,
            bool hover_centered_altitude = false,
            double maximum_altitude_offset = 1.0,
            bool randomize_wind = false,
            double maximum_wind_speed = 5.0,
            std::array<double, 3> linear_drag = {0.0, 0.0, 0.0},
            double horizontal_action_weight = 0.0,
            double altitude_action_weight = 0.0,
            double yaw_action_weight = 0.0,
            aerial::rl::ObservationMode observation_mode =
                aerial::rl::ObservationMode::Base,
            aerial::rl::FlightTask task =
                aerial::rl::FlightTask::Hover,
            bool randomize_target = false,
            std::array<double, 3> maximum_target_offset =
                {2.0, 2.0, 1.0}
        )
        {
            aerial::rl::FlightEnvironmentParameters parameters;

            parameters.time_step = time_step;
            parameters.maximum_episode_time =
                maximum_episode_time;
            parameters.action_mode = action_mode;

            parameters.pid_targets.reference = pid_target_reference;

            parameters.pid_targets.maximum_horizontal_offset =
                maximum_horizontal_offset;

            parameters.pid_targets.hover_centered_altitude =
                hover_centered_altitude;

            parameters.pid_targets.maximum_altitude_offset =
                maximum_altitude_offset;

            parameters.wind_randomization.enabled = randomize_wind;

            parameters.wind_randomization.maximum_horizontal_speed =
                maximum_wind_speed;

            parameters.linear_drag = Eigen::Vector3d(
                linear_drag[0],
                linear_drag[1],
                linear_drag[2]
            );

            parameters.action_regularization.horizontal_weight =
                horizontal_action_weight;

            parameters.action_regularization.altitude_weight =
                altitude_action_weight;

            parameters.action_regularization.yaw_weight =
                yaw_action_weight;

            parameters.observation_mode = observation_mode;

            parameters.task = task;

            parameters.target_tracking.randomize_target =
                randomize_target;

            parameters.target_tracking.maximum_target_offset =
                Eigen::Vector3d(
                    maximum_target_offset[0],
                    maximum_target_offset[1],
                    maximum_target_offset[2]
                );

            auto& randomization =
                parameters.initial_state_randomization;

            randomization.enabled = randomize_initial_state;

            randomization.maximum_position_offset =
                Eigen::Vector3d(
                    maximum_position_offset[0],
                    maximum_position_offset[1],
                    maximum_position_offset[2]
                );

            randomization.maximum_velocity =
                Eigen::Vector3d(
                    maximum_velocity[0],
                    maximum_velocity[1],
                    maximum_velocity[2]
                );

            environment_ =
                std::make_unique<aerial::rl::FlightEnvironment>(
                    parameters
                );
        }

        std::vector<double> Reset(
            std::uint32_t seed = 42
        )
        {
            environment_->Reset(seed);

            return environment_->GetPolicyObservation();
        }

        std::tuple<
            std::vector<double>,
            double,
            bool,
            bool
        > Step(const aerial::rl::Action& action)
        {
            const auto result = environment_->StepAction(action);

            return {
                environment_->GetPolicyObservation(),
                result.reward,
                result.terminated,
                result.truncated
            };
        }

        std::vector<double> GetObservation() const
        {
            return environment_->GetPolicyObservation();
        }

        std::size_t GetObservationSize() const
        {
            return environment_->GetPolicyObservationSize();
        }

        std::array<double, 3> GetPosition() const
        {
            const auto& position =
                environment_->GetTrueState().position;

            return {
                position.x(),
                position.y(),
                position.z()
            };
        }

        std::array<double, 3> GetTargetPosition() const
        {
            const auto& target =
                environment_->GetTargetPosition();

            return {
                target.x(),
                target.y(),
                target.z()
            };
        }

        std::array<double, 3> GetVelocity() const
        {
            const auto& velocity =
                environment_->GetTrueState().velocity;

            return {
                velocity.x(),
                velocity.y(),
                velocity.z()
            };
        }

        std::array<double, 3> GetWindVelocity() const
        {
            const auto& wind = environment_->GetWindVelocity();

            return {
                wind.x(),
                wind.y(),
                wind.z()
            };
        }

        double GetSimulationTime() const
        {
            return environment_->GetSimulationTime();
        }

        aerial::rl::ActionMode GetActionMode() const
        {
            return environment_->GetActionMode();
        }

    private:
        std::unique_ptr<aerial::rl::FlightEnvironment>
            environment_;
    };
}

PYBIND11_MODULE(aerial_sim, module)
{
    module.doc() =
        "Python bindings for Project Aerial RL";

    py::enum_<aerial::rl::ActionMode>(
        module, "ActionMode"
    )
        .value(
            "DIRECT_MOTOR",
            aerial::rl::ActionMode::DirectMotor
        )
        .value(
            "BODY_RATE",
            aerial::rl::ActionMode::BodyRate
        )
        .value(
            "PID_TARGETS",
            aerial::rl::ActionMode::PIDTargets
        )
        .export_values();

    py::enum_<aerial::rl::PIDTargetReference>(
        module, "PIDTargetReference"
    )
        .value(
            "EPISODE_RELATIVE",
            aerial::rl::PIDTargetReference::EpisodeRelative
        )
        .value(
            "WORLD_RELATIVE",
            aerial::rl::PIDTargetReference::WorldRelative
        )
        .export_values();

    py::enum_<aerial::rl::ObservationMode>(
        module, "ObservationMode"
    )
        .value(
            "BASE",
            aerial::rl::ObservationMode::Base
        )
        .value(
            "HOVER_AUGMENTED",
            aerial::rl::ObservationMode::HoverAugmented
        )
        .export_values();

    py::enum_<aerial::rl::FlightTask>(
        module, "FlightTask"
    )
        .value(
            "HOVER",
            aerial::rl::FlightTask::Hover
        )
        .value(
            "TARGET_TRACKING",
            aerial::rl::FlightTask::TargetTracking
        )
        .export_values();

    py::class_<PythonFlightEnvironment>(
        module, "FlightEnvironment"
    )
        .def(
            py::init<
                double,
                double,
                aerial::rl::ActionMode,
                bool,
                std::array<double, 3>,
                std::array<double, 3>,
                aerial::rl::PIDTargetReference,
                double,
                bool,
                double,
                bool,
                double,
                std::array<double, 3>,
                double,
                double,
                double,
                aerial::rl::ObservationMode,
                aerial::rl::FlightTask,
                bool,
                std::array<double, 3>
            >(),
            py::arg("time_step") = 1.0 / 120.0,
            py::arg("maximum_episode_time") = 20.0,
            py::arg("action_mode") =
                aerial::rl::ActionMode::PIDTargets,
            py::arg("randomize_initial_state") = false,
            py::arg("maximum_position_offset") =
                std::array<double, 3>{0.5, 0.5, 0.2},
            py::arg("maximum_velocity") =
                std::array<double, 3>{0.2, 0.2, 0.0},
            py::arg("pid_target_reference") =
                aerial::rl::PIDTargetReference::EpisodeRelative,
            py::arg("maximum_horizontal_offset") = 5.0,
            py::arg("hover_centered_altitude") = false,
            py::arg("maximum_altitude_offset") = 1.0,
            py::arg("randomize_wind") = false,
            py::arg("maximum_wind_speed") = 5.0,
            py::arg("linear_drag") = std::array<double, 3>{0.0, 0.0, 0.0},
            py::arg("horizontal_action_weight") = 0.0,
            py::arg("altitude_action_weight") = 0.0,
            py::arg("yaw_action_weight") = 0.0,
            py::arg("observation_mode") =
                aerial::rl::ObservationMode::Base,
            py::arg("task") =
                aerial::rl::FlightTask::Hover,
            py::arg("randomize_target") = false,
            py::arg("maximum_target_offset") =
                std::array<double, 3>{2.0, 2.0, 1.0}
        )
        .def(
            "reset",
            &PythonFlightEnvironment::Reset,
            py::arg("seed") = 42
        )
        .def(
            "step",
            &PythonFlightEnvironment::Step,
            py::arg("action")
        )
        .def(
            "get_observation",
            &PythonFlightEnvironment::GetObservation
        )
        .def(
            "get_observation_size",
            &PythonFlightEnvironment::GetObservationSize
        )
        .def(
            "get_position",
            &PythonFlightEnvironment::GetPosition
        )
        .def(
            "get_target_position",
            &PythonFlightEnvironment::GetTargetPosition
        )
        .def(
            "get_velocity",
            &PythonFlightEnvironment::GetVelocity
        )
        .def(
            "get_simulation_time",
            &PythonFlightEnvironment::GetSimulationTime
        )
        .def(
            "get_action_mode",
            &PythonFlightEnvironment::GetActionMode
        )
        .def(
            "get_wind_velocity",
            &PythonFlightEnvironment::GetWindVelocity
        );
}

