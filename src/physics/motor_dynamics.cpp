// /src/physics/motor_dynamics.cpp

#include "aerial/physics/motor_dynamics.hpp"

#include <algorithm>
#include <cmath>

namespace aerial::physics
{
    void UpdateMotorDynamics(
            MotorState& state,
            const DroneParameters& parameters,
            const std::array<double, 4>& target_thrusts,
            double dt
            )
    {
        if (dt <= 0.0 || !std::isfinite(dt)) {
            return;
        }

        const double max_thrust = parameters.max_motor_thrust;

        const double tau = parameters.motor_time_constant;
        
        if (!std::isfinite(max_thrust) || max_thrust <= 0.0 || !std::isfinite(tau) || tau <= 0.0) {
            return;
        }

        // Exact response for a constant target over one timestep
        const double response = -std::expm1(-dt / tau);

        for (std::size_t i = 0; i < state.thrusts.size(); ++i) {
            if (!std::isfinite(target_thrusts[i]) || !std::isfinite(state.thrusts[i])) {
                continue;
            }

            const double target = std::clamp(target_thrusts[i], 0.0, max_thrust);

            const double current = std::clamp(state.thrusts[i], 0.0, max_thrust);

            state.thrusts[i] = current + (target - current) * response;
        }
    }
}
