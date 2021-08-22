
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotFEM-test)
    find_package(FrictionQPotFEM REQUIRED CONFIG)
    option(USE_ASSERT "${PROJECT_NAME}: Build with FrictionQPotFEM::assert" ON)
    option(USE_DEBUG "${PROJECT_NAME}: Build with FrictionQPotFEM::debug" OFF)
    option(USE_SIMD "${PROJECT_NAME}: Use xsimd and -march=native if available" OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(testname "unit-tests")

find_package(Catch2 REQUIRED)
find_package(xtensor REQUIRED)

set(exec
    main.cpp
    Generic2d.cpp
    Generic2d_historic.cpp
    UniformSingleLayer2d.cpp
    UniformSingleLayer2d_LocalTrigger.cpp
    UniformMultiLayerIndividualDrive2d.cpp
    UniformMultiLayerLeverDrive2d.cpp)

add_executable(${testname} ${exec})

target_link_libraries(${testname} PRIVATE
    Catch2::Catch2
    FrictionQPotFEM
    FrictionQPotFEM::compiler_warnings)

if(USE_ASSERT)
    target_link_libraries(${testname} PRIVATE FrictionQPotFEM::assert)
    message(STATUS "Compiling ${PROJECT_NAME}-test with FrictionQPotFEM::assert")
endif()

if(USE_DEBUG)
    target_link_libraries(${testname} PRIVATE FrictionQPotFEM::debug)
    message(STATUS "Compiling ${PROJECT_NAME}-test with FrictionQPotFEM::debug")
endif()

if(USE_SIMD)
    find_package(xsimd REQUIRED)
    target_link_libraries(${testname} PRIVATE xtensor::use_xsimd xtensor::optimize)
    message(STATUS "Compiling ${PROJECT_NAME}-test with hardware optimization")
endif()

add_test(NAME ${testname} COMMAND ${testname})
