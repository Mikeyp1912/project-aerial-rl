// /include/aerial/physics/simulation_environment.hpp

#pragma once

#include "aerial/physics/simulation.hpp"

namespace aerial::physics
{
    class Simulation
    {
        public:
            explicit Simulation(const DroneParameters& parameters = DroneParameters{});

            void Reset();

            void Reset(const DroneState& initial_state);

            void Step(
                    const MotorCommands& commands,
                    double dt
                    );

            const DroneState& GetDroneState() const;
            const MotorState& GetMotorState() const;
            const DroneParameters& GetParameters() const;

        private:
            DroneParameters parameters_;
            DroneState drone_state_;
            MotorState motor_state_;
    };
}
