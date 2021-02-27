/**
Defines used in the library.

\file config.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_CONFIG_H
#define FRICTIONQPOTFEM_CONFIG_H

#include <string>
#include <algorithm>

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
/**
\endcond
*/

/**
All assertions are implementation as::

    FRICTIONQPOTFEM_ASSERT(...)

They can be enabled by::

    #define FRICTIONQPOTFEM_ENABLE_ASSERT

(before including FrictionQPotFEM/).
The advantage is that:

-   File and line-number are displayed if the assertion fails.
-   FrictionQPotFEM's assertions can be enabled/disabled independently from those of other libraries.

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

#endif
