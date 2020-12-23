
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotFEM-test)
    find_package(FrictionQPotFEM REQUIRED CONFIG)
endif()

set(ASSERT ON)
set(DEBUG OFF)
option(XSIMD "Use xsimd optimisations" ON)

set(CMAKE_BUILD_TYPE Release)

set(test_name "unit-tests")

find_package(Catch2 REQUIRED)
find_package(xtensor REQUIRED)

add_executable(${test_name}
    main.cpp
    UniformSingleLayer2d.cpp
    UniformSingleLayer2d_LocalTrigger.cpp
    UniformSingleLayer2d_historic.cpp)

target_link_libraries(${test_name} PRIVATE
    Catch2::Catch2
    FrictionQPotFEM
    FrictionQPotFEM::compiler_warnings)

if (NOT (WIN32 AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
    target_link_libraries(${test_name} PRIVATE xtensor::optimize)
else()
    message(STATUS "Skipping 'xtensor::optimize' because of known bug on Windows/clang.")
endif()


if(ASSERT)
    target_link_libraries(${test_name} PRIVATE FrictionQPotFEM::assert)
endif()

if(DEBUG)
    target_link_libraries(${test_name} PRIVATE FrictionQPotFEM::debug)
endif()

if(XSIMD)
    find_package(xsimd REQUIRED)
    target_link_libraries(${test_name} PRIVATE xtensor::use_xsimd)
endif()

add_test(NAME ${test_name} COMMAND ${test_name})
