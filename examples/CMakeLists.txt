
cmake_minimum_required(VERSION 3.16)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotFEM-examples)
    find_package(FrictionQPotFEM REQUIRED CONFIG)
endif()

if(NOT USE_WARNINGS)
    set(USE_WARNINGS 1)
endif()

if(NOT USE_ASSERT)
    set(USE_ASSERT 1)
endif()

if(NOT USE_DEBUG)
    set(USE_DEBUG 1)
endif()

if(NOT USE_SIMD)
    set(USE_SIMD 0)
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
        fmt::fmt
        HighFive)

    if(USE_SIMD)
        find_package(xsimd REQUIRED)
        target_link_libraries(${testname} PRIVATE xtensor::use_xsimd xtensor::optimize)
        message(STATUS "Compiling ${PROJECT_NAME}-examples(${testname}) with hardware optimization")
    endif()

    if(USE_ASSERT)
        target_link_libraries(${testname} PRIVATE FrictionQPotFEM::assert)
        message(STATUS "Compiling ${PROJECT_NAME}-examples(${testname}) with FrictionQPotFEM::assert")
    endif()

    if(USE_DEBUG)
        target_link_libraries(${testname} PRIVATE FrictionQPotFEM::debug)
        message(STATUS "Compiling ${PROJECT_NAME}-examples(${testname}) with FrictionQPotFEM::debug")
    endif()

    if(USE_WARNINGS)
        target_link_libraries(${testname} PRIVATE FrictionQPotFEM::warnings)
        message(STATUS "Compiling ${PROJECT_NAME}-examples(${testname}) with FrictionQPotFEM::warnings")
    endif()

endforeach()

