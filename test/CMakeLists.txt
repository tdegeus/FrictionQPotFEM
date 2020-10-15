
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotFEM-test)
    find_package(FrictionQPotFEM REQUIRED CONFIG)
endif()

set(WARNINGS ON)
set(ASSERT ON)
set(DEBUG ON)
set(XSIMD OFF)

set(CMAKE_BUILD_TYPE Release)

find_package(Catch2 REQUIRED)
add_executable(main main.cpp UniformSingleLayer2d.cpp)
target_link_libraries(main PRIVATE Catch2::Catch2 FrictionQPotFEM)

if(WARNINGS)
    target_link_libraries(main PRIVATE FrictionQPotFEM::compiler_warnings)
endif()

if(ASSERT)
    target_link_libraries(main PRIVATE FrictionQPotFEM::assert)
endif()

if(DEBUG)
    target_link_libraries(main PRIVATE FrictionQPotFEM::debug)
endif()

if(XSIMD)
    find_package(xtensor REQUIRED)
    find_package(xsimd REQUIRED)
    target_link_libraries(main PRIVATE xtensor::optimize xtensor::use_xsimd)
endif()
