// /include/aerial/physics/simulation.hpp

#pragma once

#include "aerial/physics/drone_state.hpp"
#include "aerial/physics/drone_parameters.hpp"
#include "aerial/physics/motor_commands.hpp"
#include "aerial/physics/motor_dynamics.hpp"

namespace aerial::physics
{
    void StepSimulation(
            DroneState& drone_state,
            MotorState& motor_state,
            const DroneParameters& parameters,
            const MotorCommands& commands,
            double dt,
            const Eigen::Vector3d& wind_velocity = Eigen::Vector3d::Zero(),
            const Eigen::Vector3d& external_force = Eigen::Vector3d::Zero(),
            const Eigen::Vector3d& external_torque = Eigen::Vector3d::Zero()
            );
}
