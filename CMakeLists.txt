
cmake_minimum_required(VERSION 3.0)

# Basic settings
# ==============

project(FrictionQPotFEM)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)
option(BUILD_EXAMPLES "${PROJECT_NAME} Build examples" OFF)

# Version
# =======

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/FrictionQPotFEM/config.h"
             _FRICTIONQPOTFEM_version_defines
     REGEX "#define FRICTIONQPOTFEM_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${_FRICTIONQPOTFEM_version_defines})
    if(ver MATCHES "#define FRICTIONQPOTFEM_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(_FRICTIONQPOTFEM_VERSION_${CMAKE_MATCH_1}
            "${CMAKE_MATCH_2}"
            CACHE INTERNAL "")
    endif()
endforeach()

set(_FRICTIONQPOTFEM_VERSION ${_FRICTIONQPOTFEM_VERSION_MAJOR}.${_FRICTIONQPOTFEM_VERSION_MINOR}.${_FRICTIONQPOTFEM_VERSION_PATCH})

message(STATUS "Building FrictionQPotFEM v${_FRICTIONQPOTFEM_VERSION}")

# Set target
# ==========

find_package(xtensor REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(GMatElastoPlasticQPot REQUIRED)
find_package(fmt REQUIRED)

add_library(FrictionQPotFEM INTERFACE)

target_include_directories(FrictionQPotFEM INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(FrictionQPotFEM INTERFACE xtensor GooseFEM GMatElastoPlasticQPot fmt::fmt)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

install(TARGETS FrictionQPotFEM EXPORT FrictionQPotFEM-targets)

install(
    EXPORT FrictionQPotFEM-targets
    FILE FrictionQPotFEMTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FrictionQPotFEM")

set(_FRICTIONQPOTFEM ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FrictionQPotFEMConfigVersion.cmake"
    VERSION ${_FRICTIONQPOTFEM_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_FRICTIONQPOTFEM})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/FrictionQPotFEMConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/FrictionQPotFEMConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FrictionQPotFEM")

# Add builds
# ==========

include("FrictionQPotFEMConfig.cmake")

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
