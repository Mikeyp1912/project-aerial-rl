// /include/aerial/telemetry/telemetry_history.hpp

#pragma once

#include <array>
#include <cassert>
#include <cstddef>

namespace aerial::telemetry
{
    template<typename T, std::size_t Capacity>
    class TelemetryHistory
    {
    public:
        static_assert(Capacity > 0);

        void Push(const T& value)
        {
            samples_[write_index_] = value;

            write_index_ = (write_index_ + 1) % Capacity;

            if (count_ < Capacity) {
                ++count_;
            }
        }

        void Clear()
        {
            count_ = 0;
            write_index_ = 0;
        }

        [[nodiscard]]
        std::size_t Size() const
        {
            return count_;
        }

        [[nodiscard]]
        constexpr std::size_t MaxSize() const
        {
            return Capacity;
        }

        [[nodiscard]]
        bool Empty() const
        {
            return count_ == 0;
        }

        // Index 0 is the oldest retained sample.
        [[nodiscard]]
        const T& At(std::size_t index) const
        {
            assert(index < count_);

            const std::size_t oldest =
                (write_index_ + Capacity - count_) % Capacity;

            return samples_[(oldest + index) % Capacity];
        }

    private:
        std::array<T, Capacity> samples_{};

        std::size_t count_ = 0;
        std::size_t write_index_ = 0;
    };
}
