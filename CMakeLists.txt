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

# Simulator executable
add_executable(simulate
    apps/simulate.cpp
)

target_include_directories(simulate
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(simulate
    PRIVATE
        Eigen3::Eigen
        raylib
)

target_compile_options(simulate
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)
