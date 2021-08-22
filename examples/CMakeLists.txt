
cmake_minimum_required(VERSION 3.16)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotFEM-examples)
    find_package(FrictionQPotFEM REQUIRED CONFIG)
    option(USE_ASSERT "${PROJECT_NAME}: Build with FrictionQPotFEM::assert" ON)
    option(USE_DEBUG "${PROJECT_NAME}: Build with FrictionQPotFEM::debug" OFF)
    option(USE_SIMD "${PROJECT_NAME}: Use xsimd and -march=native if available" OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB APP_SOURCES *.cpp)

foreach(testsourcefile ${APP_SOURCES})

    string(REPLACE ".cpp" "" testname ${testsourcefile})
    get_filename_component(testname ${testname} NAME)

    add_executable(${testname} ${testsourcefile})

    set(HIGHFIVE_USE_BOOST 0)
    set(HIGHFIVE_USE_XTENSOR 1)

    find_package(xtensor REQUIRED)
    find_package(HighFive REQUIRED)
    find_package(fmt REQUIRED)

    target_link_libraries(${testname} PUBLIC
        FrictionQPotFEM
        FrictionQPotFEM::compiler_warnings
        fmt::fmt
        HighFive)

    if(USE_ASSERT)
        target_link_libraries(${testname} PRIVATE FrictionQPotFEM::assert)
        message(STATUS "Compiling ${PROJECT_NAME}-examples(${testname}) with FrictionQPotFEM::assert")
    endif()

    if(USE_DEBUG)
        target_link_libraries(${testname} PRIVATE FrictionQPotFEM::debug)
        message(STATUS "Compiling ${PROJECT_NAME}-examples(${testname}) with FrictionQPotFEM::debug")
    endif()

    if(USE_SIMD)
        find_package(xsimd REQUIRED)
        target_link_libraries(${testname} PRIVATE xtensor::use_xsimd xtensor::optimize)
        message(STATUS "Compiling ${PROJECT_NAME}-examples(${testname}) with hardware optimization")
    endif()

endforeach()

