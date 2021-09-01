/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_CONFIG_H
#define FRICTIONQPOTFEM_CONFIG_H

#include <algorithm>
#include <string>

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
Assertions that cannot be disable.

\throw std::runtime_error
*/
#define FRICTIONQPOTFEM_REQUIRE(expr) FRICTIONQPOTFEM_ASSERT_IMPL(expr, __FILE__, __LINE__)

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
Friction simulations based on a disorder potential energy landscape and the finite element method.
*/
namespace FrictionQPotFEM {
}

#endif
