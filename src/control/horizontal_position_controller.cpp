// /src/control/horizontal_position_controller.cpp

#include "aerial/control/horizontal_position_controller.hpp"

#include <cmath>
#include <stdexcept>

namespace aerial::control
{
    HorizontalPositionControllerOutput UpdateHorizontalPositionController(
            const physics::DroneState& drone_state,
            const Eigen::Vector2d& target_position,
            const HorizontalPositionControllerParameters& parameters
            )
    {
        if (!drone_state.position.allFinite() ||
            !target_position.allFinite() ||
            !std::isfinite(parameters.position_kp) ||
            !std::isfinite(parameters.max_horizontal_velocity) ||
            parameters.position_kp < 0.0 ||
            parameters.max_horizontal_velocity < 0.0) {

            throw std::invalid_argument(
                "Invalid horizontal position controller input"
            );
        }

        HorizontalPositionControllerOutput output;

        const Eigen::Vector2d current_position = drone_state.position.head<2>();

        output.position_error = target_position - current_position;

        output.desired_velocity = parameters.position_kp * output.position_error;

        const double speed = output.desired_velocity.norm();

        if (speed > parameters.max_horizontal_velocity && speed > 0.0) {
            output.desired_velocity *= parameters.max_horizontal_velocity / speed;

            output.saturated = true;
        }

        return output;
    }
}
