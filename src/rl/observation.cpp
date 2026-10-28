// /src/rl/observation.cpp

#include "aerial/rl/observation.hpp"

namespace aerial::rl
{
    Observation MakeObservation(
            const sensors::SensorMeasurements& measurements
            )
    {
        return Observation{
            measurements.position.x(),
            measurements.position.y(),
            measurements.position.z(),

            measurements.velocity.x(),
            measurements.velocity.y(),
            measurements.velocity.z(),

            measurements.orientation.w(),
            measurements.orientation.x(),
            measurements.orientation.y(),
            measurements.orientation.z(),

            measurements.angular_velocity.x(),
            measurements.angular_velocity.y(),
            measurements.angular_velocity.z(),

            measurements.acceleration.x(),
            measurements.acceleration.y(),
            measurements.acceleration.z()
        };
    }
}
