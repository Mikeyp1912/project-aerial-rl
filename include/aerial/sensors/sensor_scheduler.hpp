// /include/aerial/sensors/sensor_scheduler.hpp

#pragma once

#include <cstdint>

#include "aerial/sensors/sensor_model.hpp"

namespace aerial::sensors
{
    struct SensorSchedulerParameters
    {
        double imu_frequency = 120.0;
        double navigation_frequency = 30.0;
    };

    class SensorScheduler{
        public:
            explicit SensorScheduler(
                    const SensorParameters& sensor_parameters = SensorParameters{},
                    const SensorSchedulerParameters& scheduler_parameters = SensorSchedulerParameters{},
                    std::uint32_t seed = 42
                    );

            void Update(
                    const physics::DroneState& true_state,
                    const Eigen::Vector3d& world_acceleration,
                    double simulation_time
                    );

            void Reset(std::uint32_t seed = 42);

            const SensorMeasurements& GetMeasurements() const;

            std::uint64_t GetImuSampleCount() const;
            std::uint64_t GetNavigationSampleCount() const;

        private:
            SensorModel sensor_model_;
            SensorSchedulerParameters parameters_;

            SensorMeasurements measurements_;

            double next_imu_time_ = 0.0;
            double next_navigation_time_ = 0.0;

            std::uint64_t imu_sample_count_ = 0;
            std::uint64_t navigation_sample_count_ = 0;
    };
}
