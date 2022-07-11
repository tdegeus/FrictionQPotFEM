/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_CONFIG_H
#define FRICTIONQPOTFEM_CONFIG_H

#include <algorithm>
#include <string>
#include <xtensor/xtensor.hpp>

/**
\cond
*/
#define Q(x) #x
#define QUOTE(x) Q(x)

#define FRICTIONQPOTFEM_ASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

#define FRICTIONQPOTFEM_WIP_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::out_of_range( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

#define FRICTIONQPOTFEM_WARNING_IMPL(message, file, line) \
    std::cout << std::string(file) + ':' + std::to_string(line) + ": " message ") \n\t";

/**
\endcond
*/

/**
All assertions are implementation as::

    FRICTIONQPOTFEM_ASSERT(...)

They can be enabled by::

    #define FRICTIONQPOTFEM_ENABLE_ASSERT

(before including FrictionQPotFEM).
The advantage is that:

-   File and line-number are displayed if the assertion fails.
-   FrictionQPotFEM's assertions can be enabled/disabled independently from those of other
    libraries.

\throw std::runtime_error
*/
#ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
#define FRICTIONQPOTFEM_ASSERT(expr) FRICTIONQPOTFEM_ASSERT_IMPL(expr, __FILE__, __LINE__)
#else
#define FRICTIONQPOTFEM_ASSERT(expr)
#endif

/**
Assertions that cannot be disabled.

\throw std::runtime_error
*/
#define FRICTIONQPOTFEM_REQUIRE(expr) FRICTIONQPOTFEM_ASSERT_IMPL(expr, __FILE__, __LINE__)

/**
Assertions on a implementation limitation (that in theory can be resolved)

\throw std::out_of_range
*/
#define FRICTIONQPOTFEM_WIP(expr) FRICTIONQPOTFEM_WIP_IMPL(expr, __FILE__, __LINE__)

/**
All warnings are implemented as::

    FRICTIONQPOTFEM_WARNING(...)

They can be disabled by::

    #define FRICTIONQPOTFEM_DISABLE_WARNING
*/
#ifdef FRICTIONQPOTFEM_DISABLE_WARNING
#define FRICTIONQPOTFEM_WARNING(message)
#else
#define FRICTIONQPOTFEM_WARNING(message) FRICTIONQPOTFEM_WARNING_IMPL(message, __FILE__, __LINE__)
#endif

/**
All warnings specific to the Python API are implemented as::

    FRICTIONQPOTFEM_WARNING_PYTHON(...)

They can be enabled by::

    #define FRICTIONQPOTFEM_ENABLE_WARNING_PYTHON
*/
#ifdef FRICTIONQPOTFEM_ENABLE_WARNING_PYTHON
#define FRICTIONQPOTFEM_WARNING_PYTHON(message) \
    FRICTIONQPOTFEM_WARNING_IMPL(message, __FILE__, __LINE__)
#else
#define FRICTIONQPOTFEM_WARNING_PYTHON(message)
#endif

/**
Current version.

Either:

-   Configure using CMake at install time. Internally uses::

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using::

        ver = `python -c "from setuptools_scm import get_version; print(get_version())"`
        -DFRICTIONQPOTFEM_VERSION="${ver}"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
``setuptools_scm``.
Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to overwrite
the automatic version.
*/
#ifndef FRICTIONQPOTFEM_VERSION
#define FRICTIONQPOTFEM_VERSION "@PROJECT_VERSION@"
#endif

/**
Friction simulations based on a disorder potential energy landscape and the finite element method.
*/
namespace FrictionQPotFEM {

/**
Container type.
*/
namespace array_type {

#ifdef FRICTIONQPOTFEM_USE_XTENSOR_PYTHON

/**
Fixed (static) rank array.
*/
template <typename T, size_t N>
using tensor = xt::pytensor<T, N>;

#else

/**
Fixed (static) rank array.
*/
template <typename T, size_t N>
using tensor = xt::xtensor<T, N>;

#endif

} // namespace array_type

namespace detail {

inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

/**
Return version string, e.g. `"0.8.0"`
\return Version string.
*/
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(FRICTIONQPOTFEM_VERSION)));
}

} // namespace FrictionQPotFEM

#endif
