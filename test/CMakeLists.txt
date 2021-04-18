
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotFEM-test)
    find_package(FrictionQPotFEM REQUIRED CONFIG)
endif()

set(ASSERT ON)
set(DEBUG OFF)
set(ALIGN OFF)

set(CMAKE_BUILD_TYPE Release)

set(test_name "unit-tests")

find_package(Catch2 REQUIRED)
find_package(xtensor REQUIRED)

add_executable(${test_name}
    main.cpp
    Generic2d.cpp
    Generic2d_historic.cpp
    UniformSingleLayer2d.cpp
    UniformSingleLayer2d_LocalTrigger.cpp
    UniformMultiLayerIndividualDrive2d.cpp)

target_link_libraries(${test_name} PRIVATE
    Catch2::Catch2
    FrictionQPotFEM
    FrictionQPotFEM::compiler_warnings)

if(ASSERT)
    target_link_libraries(${test_name} PRIVATE FrictionQPotFEM::assert)
endif()

if(DEBUG)
    target_link_libraries(${test_name} PRIVATE FrictionQPotFEM::debug)
endif()

if(ALIGN)
    find_package(xsimd REQUIRED)
    target_link_libraries(${test_name} PRIVATE xtensor::use_xsimd xtensor::optimize)
endif()

add_test(NAME ${test_name} COMMAND ${test_name})
