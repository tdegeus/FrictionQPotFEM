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

#define Q(x) #x
#define QUOTE(x) Q(x)

#ifdef FRICTIONQPOTFEM_ENABLE_ASSERT

    #define FRICTIONQPOTFEM_ASSERT(expr) \
        FRICTIONQPOTFEM_ASSERT_IMPL(expr, __FILE__, __LINE__)

    #define FRICTIONQPOTFEM_ASSERT_IMPL(expr, file, line) \
        if (!(expr)) { \
            throw std::runtime_error( \
                std::string(file) + ':' + std::to_string(line) + \
                ": assertion failed (" #expr ") \n\t"); \
        }

#else

    #define FRICTIONQPOTFEM_ASSERT(expr)

#endif

#define FRICTIONQPOTFEM_REQUIRE(expr) \
    FRICTIONQPOTFEM_REQUIRE_IMPL(expr, __FILE__, __LINE__)

#define FRICTIONQPOTFEM_REQUIRE_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

#endif
