
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotFEM-examples)
    find_package(FrictionQPotFEM REQUIRED CONFIG)
endif()

set(WARNINGS ON)
set(ASSERT OFF)

set(CMAKE_BUILD_TYPE Release)

file(GLOB APP_SOURCES *.cpp)

foreach(testsourcefile ${APP_SOURCES})

    string(REPLACE ".cpp" "" testname ${testsourcefile})
    get_filename_component(testname ${testname} NAME)

    add_executable(${testname} ${testsourcefile})

    find_package(xtensor REQUIRED)
    find_package(xsimd REQUIRED)

    target_link_libraries(${testname} PUBLIC
        FrictionQPotFEM
        xtensor::optimize
        xtensor::use_xsimd)

    if(WARNINGS)
        target_link_libraries(${testname} PRIVATE FrictionQPotFEM::compiler_warnings)
    endif()

    if(ASSERT)
        target_link_libraries(${testname} PRIVATE FrictionQPotFEM::assert)
    endif()

endforeach()

