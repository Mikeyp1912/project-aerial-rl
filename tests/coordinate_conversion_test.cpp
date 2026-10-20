// /tests/coordinate_conversion_test.cpp

#include <cmath>
#include <iostream>

#include "aerial/rendering/coordinate_conversion.hpp"

namespace 
{
    bool Near(float actual, float expected)
    {
        return std::abs(actual - expected) < 1e-6f;
    }

    bool CheckVector(
            const Vector3& actual,
            float x,
            float y,
            float z
            )
    {
        return Near(actual.x, x) && Near(actual.y, y) && Near(actual.z, z);
    }
}

int main() {
    using aerial::rendering::ToRaylib;

    if (!CheckVector(ToRaylib({0.0, 0.0, 1.0}), 0.0f, 1.0f, 0.0f)) {
        std::cerr << "Altitude conversion failed\n";
        return 1;
    }

    if (!CheckVector(ToRaylib({1.0, 2.0, 3.0}), 1.0f, 3.0f, -2.0f)) {
        std::cerr << "Coordinate conversion failed\n";
        return 1;
    }

    if (!CheckVector(ToRaylib({0.0, 0.0, 0.0}), 0.0f, 0.0f, 0.0f)) {
        std::cerr << "Origin conversion failed\n";
        return 1;
    }

    std::cout << "Coordinate Conversion tests passed\n";
    return 0;
}
