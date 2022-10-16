cmake_minimum_required(VERSION 3.18..3.21)

project(FrictionQPotFEM)

string(TOUPPER "${PROJECT_NAME}" PROJECT_NAME_UPPER)

# Command-line options
# ====================

option(BUILD_ALL "${PROJECT_NAME}: Build tests, Python API & docs" OFF)
option(BUILD_TESTS "${PROJECT_NAME}: Build tests" OFF)
option(BUILD_PYTHON "${PROJECT_NAME}: Build Python API" OFF)
option(BUILD_DOCS "${PROJECT_NAME}: Build docs (use `make html`)" OFF)
option(USE_WARNINGS "${PROJECT_NAME}: Build with runtime warnings" ON)
option(USE_ASSERT "${PROJECT_NAME}: Build with assertions" ON)
option(USE_DEBUG "${PROJECT_NAME}: Build in debug mode" OFF)
option(USE_SIMD "${PROJECT_NAME}: Build with hardware optimization" OFF)

if(SKBUILD)
    set(BUILD_ALL 0)
    set(BUILD_TESTS 0)
    set(BUILD_PYTHON 1)
    set(BUILD_DOCS 0)
endif()

# Read version
# ============

if (DEFINED ENV{SETUPTOOLS_SCM_PRETEND_VERSION})
    set(PROJECT_VERSION $ENV{SETUPTOOLS_SCM_PRETEND_VERSION})
    message(STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION} (read from SETUPTOOLS_SCM_PRETEND_VERSION)")
else()
    execute_process(
        COMMAND python -c "from setuptools_scm import get_version; print(get_version())"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE PROJECT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION}")
endif()

# Set target
# ==========

find_package(xtensor REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(GMatElastoPlasticQPot REQUIRED)
find_package(prrng REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(${PROJECT_NAME} INTERFACE xtensor GooseFEM GMatElastoPlasticQPot prrng)

target_compile_definitions(${PROJECT_NAME} INTERFACE
    ${PROJECT_NAME_UPPER}_VERSION="${PROJECT_VERSION}")

# Libraries
# =========

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CTest)
include("${PROJECT_NAME}Config.cmake")

# Installation headers / CMake / pkg-config
# =========================================

if(NOT SKBUILD)

    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

    configure_file("include/${PROJECT_NAME}/config.h"
                   "${CMAKE_CURRENT_BINARY_DIR}/config.h"
                    @ONLY)

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h"
            DESTINATION "include/${PROJECT_NAME}/")

    install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets)

    install(
        EXPORT ${PROJECT_NAME}-targets
        FILE "${PROJECT_NAME}Targets.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

    set(${PROJECT_NAME}_TMP ${CMAKE_SIZEOF_VOID_P})
    unset(CMAKE_SIZEOF_VOID_P)

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion)

    set(CMAKE_SIZEOF_VOID_P ${${PROJECT_NAME}_TMP})

    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

endif()

# Build tests
# ===========

if(BUILD_TESTS OR BUILD_ALL)

    enable_testing()
    add_subdirectory(tests)

endif()

# Build Python API
# ================

if(BUILD_PYTHON OR BUILD_ALL)

    # The C++ functions are build to a library with name "_${PROJECT_NAME}"
    # The Python library simply loads all functions
    set(PYPROJECT_NAME "_${PROJECT_NAME}")

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    find_package(pybind11 REQUIRED CONFIG)
    find_package(xtensor-python REQUIRED)

    if (SKBUILD)
        find_package(NumPy REQUIRED)
    else()
        find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
    endif()

    pybind11_add_module(${PYPROJECT_NAME} python/main.cpp)

    target_compile_definitions(${PYPROJECT_NAME} PUBLIC VERSION_INFO=${PROJECT_VERSION})
    target_link_libraries(${PYPROJECT_NAME} PUBLIC ${PROJECT_NAME} xtensor-python)

    if (SKBUILD)
        target_include_directories(${PYPROJECT_NAME} PUBLIC ${NumPy_INCLUDE_DIRS})
    else()
        target_link_libraries(${PYPROJECT_NAME} PUBLIC ${PROJECT_NAME} pybind11::module Python::NumPy)
    endif()

    if (USE_WARNINGS)
        target_link_libraries(${PYPROJECT_NAME} PUBLIC ${PROJECT_NAME}::warnings)
        message(STATUS "Compiling ${PROJECT_NAME}-Python with run-time warnings")
    endif()

    if (USE_ASSERT)
        target_link_libraries(${PYPROJECT_NAME} PUBLIC ${PROJECT_NAME}::assert)
        message(STATUS "Compiling ${PROJECT_NAME}-Python with assertions")
    endif()

    if (USE_DEBUG)
        target_link_libraries(${PYPROJECT_NAME} PUBLIC ${PROJECT_NAME}::debug)
        message(STATUS "Compiling ${PROJECT_NAME}-Python in debug mode")
    endif()

    if (USE_SIMD)
        find_package(xtensor REQUIRED)
        find_package(xsimd REQUIRED)
        target_link_libraries(${PYPROJECT_NAME} PUBLIC xtensor::optimize xtensor::use_xsimd)
        message(STATUS "Compiling ${PROJECT_NAME}-Python with hardware optimization")
    endif()

    if (SKBUILD)
        if(APPLE)
            set_target_properties(${PYPROJECT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
        else()
            set_target_properties(${PYPROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
        endif()
        install(TARGETS ${PYPROJECT_NAME} DESTINATION .)
    endif()

endif()

# Build documentation
# ===================

if(BUILD_DOCS OR BUILD_ALL)

    find_package(Doxygen REQUIRED)
    find_package(GMatTensor REQUIRED)
    find_package(QPot REQUIRED)

    get_target_property(GooseFEM_INCLUDE GooseFEM INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(GMatElastoPlasticQPot_INCLUDE GMatElastoPlasticQPot INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(GMatTensor_INCLUDE GMatTensor INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(QPot_INCLUDE QPot INTERFACE_INCLUDE_DIRECTORIES)

    set(${PROJECT_NAME}_DOCS_INCL
        "${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}"
        "${GooseFEM_INCLUDE}/GooseFEM"
        "${GMatElastoPlasticQPot_INCLUDE}/GMatElastoPlasticQPot"
        "${GMatTensor_INCLUDE}/GMatTensor")

    set(${PROJECT_NAME}_DOCS_STRIP
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${GooseFEM_INCLUDE}"
        "${GMatElastoPlasticQPot_INCLUDE}"
        "${GMatTensor_INCLUDE}"
        "${QPot_INCLUDE}")

    # deal with strange doxygen bug
    string(REGEX REPLACE "(.*)(include)" "\\1" GooseFEM_STRIP ${GooseFEM_INCLUDE})
    string(REGEX REPLACE "(.*)(include)" "\\1" GMatElastoPlasticQPot_STRIP ${GMatElastoPlasticQPot_INCLUDE})
    string(REGEX REPLACE "(.*)(include)" "\\1" GMatTensor_STRIP ${GMatTensor_INCLUDE})
    string(REGEX REPLACE "(.*)(include)" "\\1" QPot_STRIP ${QPot_INCLUDE})

    set(${PROJECT_NAME}_DOCS_STRIP_INC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${GooseFEM_STRIP}"
        "${GMatElastoPlasticQPot_STRIP}"
        "${GMatTensor_STRIP}"
        "${QPot_STRIP}")

    set(DOXYGEN_EXCLUDE_SYMBOLS detail)
    set(DOXYGEN_CASE_SENSE_NAMES YES)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_MACRO_EXPANSION YES)
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_WARN_AS_ERROR NO)
    set(DOXYGEN_HTML_COLORSTYLE TOGGLE)
    set(DOXYGEN_WARN_LOGFILE "${CMAKE_CURRENT_BINARY_DIR}/doxygen_warnings.log")
    set(DOXYGEN_ALIASES "license=@par License:")
    set(DOXYGEN_STRIP_FROM_INC_PATH ${${PROJECT_NAME}_DOCS_STRIP_INC})
    set(DOXYGEN_STRIP_FROM_PATH ${${PROJECT_NAME}_DOCS_STRIP})
    doxygen_add_docs(html ${${PROJECT_NAME}_DOCS_INCL})

endif()
