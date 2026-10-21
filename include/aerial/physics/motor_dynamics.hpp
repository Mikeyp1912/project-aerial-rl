// /include/aerial/physics/motor_dynamics.hpp

#pragma once

#include <array>

#include "aerial/physics/drone_parameters.hpp"

namespace aerial::physics
{
    struct MotorState
    {
        // Actual thrust produced by each motor (N)
        std::array<double, 4> thrusts{};
    };

    void UpdateMotorDynamics(
            MotorState& state,
            const DroneParameters& parameters,
            const std::array<double, 4>& target_thrusts,
            double dt
            );
}
