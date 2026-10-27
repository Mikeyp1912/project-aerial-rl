// /include/aerial/sensors/observed_state.hpp

#pragma once

#include "aerial/physics/drone_state.hpp"
#include "aerial/sensors/sensor_model.hpp"

namespace aerial::sensors
{
    inline physics::DroneState MakeObservedState(
            const SensorMeasurements& measurements
            )
    {
        physics::DroneState state;

        state.position = measurements.position;
        state.velocity = measurements.velocity;
        state.orientation = measurements.orientation;
        state.angular_velocity = measurements.angular_velocity;

        return state;
    }
}
