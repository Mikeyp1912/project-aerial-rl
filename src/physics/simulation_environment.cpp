// /src/physics/simulation_environment.cpp

#include "aerial/physics/simulation_environment.hpp"

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
    }

    void Simulation::Step(
            const MotorCommands& commands,
            double dt
            )
    {
        StepSimulation(
                drone_state_,
                motor_state_,
                parameters_,
                commands,
                dt
                );
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
}
