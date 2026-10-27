// /include/aerial/physics/simulation_environment.hpp

#pragma once

#include <cstdint>

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

            void SetWindVelocity(const Eigen::Vector3d& wind_velocity);
            void SetExternalForce(const Eigen::Vector3d& force);
            void SetExternalTorque(const Eigen::Vector3d& torque);

            const Eigen::Vector3d& GetWindVelocity() const;
            const Eigen::Vector3d& GetExternalForce() const;
            const Eigen::Vector3d& GetExternalTorque() const;

            void ClearExternalDisturbances();

            const DroneState& GetDroneState() const;
            const MotorState& GetMotorState() const;
            const DroneParameters& GetParameters() const;

            double GetSimulationTime() const;
            std::uint64_t GetStepCount() const;

        private:
            DroneParameters parameters_;
            DroneState drone_state_;
            MotorState motor_state_;

            double simulation_time_ = 0.0;
            std::uint64_t step_count_ = 0;

            Eigen::Vector3d wind_velocity_ = Eigen::Vector3d::Zero();
            Eigen::Vector3d external_force_ = Eigen::Vector3d::Zero();
            Eigen::Vector3d external_torque_ = Eigen::Vector3d::Zero();
    };
}
