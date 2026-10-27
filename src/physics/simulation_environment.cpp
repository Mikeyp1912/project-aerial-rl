// /src/physics/simulation_environment.cpp

#include "aerial/physics/simulation_environment.hpp"

#include <cmath>

namespace aerial::physics
{
    Simulation::Simulation(const DroneParameters& parameters)
        : parameters_(parameters)
    {
        Reset();
    }

    void Simulation::Reset()
    {
        DroneState initial_state;
        initial_state.position = {0.0, 0.0, 1.0};

        Reset(initial_state);
    }

    void Simulation::Reset(const DroneState& initial_state)
    {
        drone_state_ = initial_state;

        // Motors begin stopped
        motor_state_ = MotorState{};

        simulation_time_ = 0.0;
        step_count_ = 0;

        ClearExternalDisturbances();

        world_acceleration_.setZero();
    }

    void Simulation::Step(
            const MotorCommands& commands,
            double dt
            )
    {
        if (dt <= 0.0 || !std::isfinite(dt)) {
            return;
        }

        StepSimulation(
                drone_state_,
                motor_state_,
                parameters_,
                commands,
                dt,
                wind_velocity_,
                external_force_,
                external_torque_,
                &world_acceleration_
                );

        simulation_time_ += dt;
        ++step_count_;
    }

    void Simulation::SetWindVelocity(
            const Eigen::Vector3d& wind_velocity
            )
    {
        if (!wind_velocity.allFinite()) {
            return;
        }

        wind_velocity_ = wind_velocity;
    }

    void Simulation::SetExternalForce(const Eigen::Vector3d& force)
    {
        if (!force.allFinite()) {
            return;
        }

        external_force_ = force;
    }

    void Simulation::SetExternalTorque(const Eigen::Vector3d& torque)
    {
        if (!torque.allFinite())
        {
            return;
        }

        external_torque_ = torque;
    }

    const Eigen::Vector3d& Simulation::GetWindVelocity() const
    {
        return wind_velocity_;
    }

    const Eigen::Vector3d& Simulation::GetExternalForce() const 
    {
        return external_force_;
    }

    const Eigen::Vector3d& Simulation::GetExternalTorque() const
    {
        return external_torque_;
    }

    void Simulation::ClearExternalDisturbances()
    {
        external_force_.setZero();
        external_torque_.setZero();
    }

    const DroneState& Simulation::GetDroneState() const
    {
        return drone_state_;
    }

    const MotorState& Simulation::GetMotorState() const
    {
        return motor_state_;
    }

    const DroneParameters& Simulation::GetParameters() const
    {
        return parameters_;
    }

    const Eigen::Vector3d& Simulation::GetWorldAcceleration() const
    {
        return world_acceleration_;
    }

    double Simulation::GetSimulationTime() const
    {
        return simulation_time_;
    }

    std::uint64_t Simulation::GetStepCount() const
    {
        return step_count_;
    }
}
