// /src/sensors/sensor_scheduler.cpp

#include "aerial/sensors/sensor_scheduler.hpp"

#include <cmath>
#include <stdexcept>

namespace aerial::sensors
{
    SensorScheduler::SensorScheduler(
            const SensorParameters& sensor_parameters,
            const SensorSchedulerParameters& scheduler_parameters,
            std::uint32_t seed
            )
        :
            sensor_model_(sensor_parameters, seed),
            parameters_(scheduler_parameters)
    {
        if (!std::isfinite(parameters_.imu_frequency) ||
            !std::isfinite(parameters_.navigation_frequency) ||
            parameters_.imu_frequency <= 0.0 ||
            parameters_.navigation_frequency <= 0.0) {
            throw std::invalid_argument("Invalid sensor frequencies");
        }
    }

    void SensorScheduler::Update(
            const physics::DroneState& true_state,
            const Eigen::Vector3d& world_acceleration,
            double simulation_time
            )
    {
        if (!std::isfinite(simulation_time) || simulation_time < 0.0) {
            return;
        }

        constexpr double epsilon = 1e-9;

        const bool imu_due = simulation_time + epsilon >= next_imu_time_;

        const bool navigation_due = simulation_time + epsilon >= next_navigation_time_;

        if (!imu_due && !navigation_due) {
            return;
        }

        // Generate a measurement only when at least one sensor is due
        const SensorMeasurements sample = sensor_model_.Measure(true_state, world_acceleration);

        if (imu_due) {
            measurements_.orientation = sample.orientation;
            measurements_.angular_velocity = sample.angular_velocity;
            measurements_.acceleration = sample.acceleration;

            ++imu_sample_count_;

            const double period = 1.0 / parameters_.imu_frequency;

            do {
                next_imu_time_ += period;
            }
            while (next_imu_time_ <= simulation_time + epsilon);
        }

        if (navigation_due) {
            measurements_.position = sample.position;
            measurements_.velocity = sample.velocity;

            ++navigation_sample_count_;

            const double period = 1.0 / parameters_.navigation_frequency;

            do {
                next_navigation_time_ += period;
            }
            while (next_navigation_time_ <= simulation_time + epsilon);
        }
    }

    void SensorScheduler::Reset(std::uint32_t seed)
    {
        sensor_model_.Reset(seed);

        measurements_ = SensorMeasurements{};

        next_imu_time_ = 0.0;
        next_navigation_time_ = 0.0;

        imu_sample_count_ = 0;
        navigation_sample_count_ = 0;
    }

    const SensorMeasurements& SensorScheduler::GetMeasurements() const
    {
        return measurements_;
    }

    std::uint64_t SensorScheduler::GetImuSampleCount() const
    {
        return imu_sample_count_;
    }

    std::uint64_t SensorScheduler::GetNavigationSampleCount() const
    {
        return navigation_sample_count_;
    }
}
