# FrictionQPotFEM cmake module
#
# This module sets the target:
#
#     FrictionQPotFEM
#
# In addition, it sets the following variables:
#
#     FrictionQPotFEM_FOUND - true if the library is found
#     FrictionQPotFEM_VERSION - the library's version
#     FrictionQPotFEM_INCLUDE_DIRS - directory containing the library's headers
#
# The following support targets are defined to simplify things:
#
#     FrictionQPotFEM::compiler_warnings - enable compiler warnings
#     FrictionQPotFEM::assert - enable library assertions
#     FrictionQPotFEM::debug - enable all assertions (slow)

include(CMakeFindDependencyMacro)

# Define target "FrictionQPotFEM"

if(NOT TARGET FrictionQPotFEM)
    include("${CMAKE_CURRENT_LIST_DIR}/FrictionQPotFEMTargets.cmake")
endif()

# Define "FrictionQPotFEM_INCLUDE_DIRS"

get_target_property(
    FrictionQPotFEM_INCLUDE_DIRS
    FrictionQPotFEM
    INTERFACE_INCLUDE_DIRECTORIES)

# Find dependencies

find_dependency(xtensor)
find_dependency(GooseFEM)
find_dependency(GMatElastoPlasticQPot)

# Define support target "FrictionQPotFEM::compiler_warnings"

if(NOT TARGET FrictionQPotFEM::compiler_warnings)
    add_library(FrictionQPotFEM::compiler_warnings INTERFACE IMPORTED)
    if(MSVC)
        set_property(
            TARGET FrictionQPotFEM::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            /W4)
    else()
        set_property(
            TARGET FrictionQPotFEM::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    endif()
endif()

# Define support target "FrictionQPotFEM::warnings"

if(NOT TARGET FrictionQPotFEM::warnings)
    add_library(FrictionQPotFEM::warnings INTERFACE IMPORTED)
    set_property(
        TARGET FrictionQPotFEM::warnings
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        FRICTIONQPOTFEM_WARNING_PYTHON)
endif()

# Define support target "FrictionQPotFEM::assert"

if(NOT TARGET FrictionQPotFEM::assert)
    add_library(FrictionQPotFEM::assert INTERFACE IMPORTED)
    set_property(
        TARGET FrictionQPotFEM::assert
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        FRICTIONQPOTFEM_ENABLE_ASSERT)
endif()

# Define support target "FrictionQPotFEM::debug"

if(NOT TARGET FrictionQPotFEM::debug)
    add_library(FrictionQPotFEM::debug INTERFACE IMPORTED)
    set_property(
        TARGET FrictionQPotFEM::debug
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        XTENSOR_ENABLE_ASSERT
        QPOT_ENABLE_ASSERT
        GOOSEFEM_ENABLE_ASSERT
        GMATELASTOPLASTICQPOT_ENABLE_ASSERT
        FRICTIONQPOTFEM_ENABLE_ASSERT)
endif()
