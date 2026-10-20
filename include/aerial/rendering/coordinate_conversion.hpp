// /include/aerial/rendering/coordinate_conversion.hpp

#pragma once

#include <Eigen/Core>
#include <raylib.h>

namespace aerial::rendering
{
    inline Vector3 ToRaylib(const Eigen::Vector3d& position)
    {
        return {
            static_cast<float>(position.x()),
            static_cast<float>(position.z()),
            static_cast<float>(-position.y()),
        };
    };
}
