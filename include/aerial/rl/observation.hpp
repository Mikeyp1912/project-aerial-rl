// /include/aerial/rl/observation.hpp

#pragma once

#include "aerial/rl/rl_types.hpp"
#include "aerial/sensors/sensor_model.hpp"

namespace aerial::rl
{
    Observation MakeObservation(
            const sensors::SensorMeasurements& measurements
            );
}
