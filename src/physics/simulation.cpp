// /src/physics/simulation.cpp

#include "aerial/physics/simulation.hpp"

#include <cmath>

#include "aerial/physics/integrator.hpp"
#include "aerial/physics/motor_model.hpp"
#include "aerial/physics/rotational_dynamics.hpp"
#include "aerial/physics/aerodynamic_drag.hpp"

namespace aerial::physics
{
    void StepSimulation(
            DroneState &drone_state,
            MotorState &motor_state,
            const DroneParameters &parameters,
            const MotorCommands &commands,
            double dt,
            const Eigen::Vector3d& wind_velocity,
            const Eigen::Vector3d& external_force,
            const Eigen::Vector3d& external_torque
            )
    {
        if (dt <= 0.0 || !std::isfinite(dt)) {
            return ;
        }

        // 1. Convert commands into target motor thrust
        const auto target_thrusts = CommandsToThrust(parameters, commands);

        // 2. Advance motor response dynamics
        UpdateMotorDynamics(
                motor_state,
                parameters,
                target_thrusts,
                dt
                );

        // 3. Calculate combined body-frame force and torque
        const MotorOutput motor_output = CalculateMotorOutput(
                parameters,
                motor_state.thrusts
                );

        // 4. Apply drag and rotate thrust into world coordinates
        const Eigen::Vector3d thrust_force = drone_state.orientation * motor_output.force;
        
        const Eigen::Vector3d drag_force = CalculateAerodynamicDrag(
                drone_state,
                parameters,
                wind_velocity
                );

        const Eigen::Vector3d world_force = thrust_force + drag_force + external_force;

        // 5. Integrate translational motion
        Integrate(
                drone_state,
                parameters,
                world_force,
                dt
                );

        // 6. Integrate rotational motion
        IntegrateRotation(
                drone_state,
                parameters,
                motor_output.torque + external_torque,
                dt
                );
    }
}
