#
# (c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM
#

cmake_minimum_required(VERSION 3.0)

# Basic settings
# ==============

project(FrictionQPotFEM)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)
option(BUILD_EXAMPLES "${PROJECT_NAME} Build examples" OFF)

# Version
# =======

IF(DEFINED ENV{PKG_VERSION})
    set(FRICTIONQPOTFEM_VERSION $ENV{PKG_VERSION})
else()
    execute_process(
        COMMAND python -c "from setuptools_scm import get_version; print(get_version())"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE FRICTIONQPOTFEM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

message(STATUS "Building FrictionQPotFEM ${FRICTIONQPOTFEM_VERSION}")

# Set target
# ==========

find_package(xtensor REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(GMatElastoPlasticQPot REQUIRED)

add_library(FrictionQPotFEM INTERFACE)

target_include_directories(FrictionQPotFEM INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(FrictionQPotFEM INTERFACE xtensor GooseFEM GMatElastoPlasticQPot)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

configure_file("include/${PROJECT_NAME}/version.h"
               "${CMAKE_CURRENT_BINARY_DIR}/version.h"
                @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h"
        DESTINATION "include/${PROJECT_NAME}/")

install(TARGETS FrictionQPotFEM EXPORT FrictionQPotFEM-targets)

install(
    EXPORT FrictionQPotFEM-targets
    FILE FrictionQPotFEMTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FrictionQPotFEM")

set(_FRICTIONQPOTFEM ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FrictionQPotFEMConfigVersion.cmake"
    VERSION ${FRICTIONQPOTFEM_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_FRICTIONQPOTFEM})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/FrictionQPotFEMConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/FrictionQPotFEMConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FrictionQPotFEM")

# Add builds
# ==========

include("FrictionQPotFEMConfig.cmake")

include(CTest)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
