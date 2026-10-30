// /src/python/aerial_sim_bindings.cpp

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>

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
                aerial::rl::ActionMode::PIDTargets
        )
        {
            aerial::rl::FlightEnvironmentParameters parameters;

            parameters.time_step = time_step;
            parameters.maximum_episode_time =
                maximum_episode_time;

            parameters.action_mode = action_mode;

            environment_ =
                std::make_unique<aerial::rl::FlightEnvironment>(
                    parameters
                );
        }

        aerial::rl::Observation Reset(
            std::uint32_t seed = 42
        )
        {
            environment_->Reset(seed);

            return environment_->GetNormalizedRLObservation();
        }

        std::tuple<
            aerial::rl::Observation,
            double,
            bool,
            bool
        > Step(const aerial::rl::Action& action)
        {
            const auto result = environment_->StepAction(action);

            return {
                environment_->GetNormalizedRLObservation(),
                result.reward,
                result.terminated,
                result.truncated
            };
        }

        aerial::rl::Observation GetObservation() const
        {
            return environment_->GetNormalizedRLObservation();
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

    py::class_<PythonFlightEnvironment>(
        module, "FlightEnvironment"
    )
        .def(
            py::init<
                double,
                double,
                aerial::rl::ActionMode
            >(),
            py::arg("time_step") = 1.0 / 120.0,
            py::arg("maximum_episode_time") = 20.0,
            py::arg("action_mode") =
                aerial::rl::ActionMode::PIDTargets
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
            "get_simulation_time",
            &PythonFlightEnvironment::GetSimulationTime
        )
        .def(
            "get_action_mode",
            &PythonFlightEnvironment::GetActionMode
        );
}

