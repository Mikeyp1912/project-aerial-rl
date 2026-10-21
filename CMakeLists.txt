cmake_minimum_required(VERSION 3.20)

project(
    ProjectAerialRL
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Dependencies
find_package(Eigen3 REQUIRED)

# raylib
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(raylib)

# Physics library — independent of raylib
add_library(aerial_physics STATIC
    src/physics/integrator.cpp
    src/physics/thrust.cpp
    src/physics/rotational_dynamics.cpp
    src/physics/motor_model.cpp
)

target_include_directories(aerial_physics
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(aerial_physics
    PUBLIC
        Eigen3::Eigen
)

target_compile_options(aerial_physics
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Simulator executable
add_executable(simulate
    apps/simulate.cpp
    src/rendering/drone_renderer.cpp
    src/rendering/camera_controller.cpp
)

target_include_directories(simulate
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(simulate
    PRIVATE
        aerial_physics
        Eigen3::Eigen
        raylib
)

target_compile_options(simulate
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Tests
include(CTest)

if(BUILD_TESTING)
    add_executable(drone_state_test
        tests/drone_state_test.cpp
    )

    target_include_directories(drone_state_test
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(drone_state_test
        PRIVATE
            Eigen3::Eigen
    )

    target_compile_options(drone_state_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME drone_state
        COMMAND drone_state_test
    )

    add_executable(coordinate_conversion_test
        tests/coordinate_conversion_test.cpp
    )

    target_include_directories(coordinate_conversion_test
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(coordinate_conversion_test
        PRIVATE
            Eigen3::Eigen
            raylib
    )

    target_compile_options(coordinate_conversion_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME coordinate_conversion
        COMMAND coordinate_conversion_test
    )

    add_executable(integrator_test
        tests/integrator_test.cpp
    )

    target_link_libraries(integrator_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(integrator_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME integrator
        COMMAND integrator_test
    )

    add_executable(thrust_test
        tests/thrust_test.cpp
    )

    target_link_libraries(thrust_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(thrust_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME thrust
        COMMAND thrust_test
    )

    add_executable(rotational_dynamics_test
        tests/rotational_dynamics_test.cpp
    )

    target_link_libraries(rotational_dynamics_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(rotational_dynamics_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME rotational_dynamics
        COMMAND rotational_dynamics_test
    )

    add_executable(motor_model_test
        tests/motor_model_test.cpp
    )

    target_link_libraries(motor_model_test
        PRIVATE
            aerial_physics
    )

    target_compile_options(motor_model_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    add_test(
        NAME motor_model
        COMMAND motor_model_test
    )
endif()
