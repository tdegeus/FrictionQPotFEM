/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM

*/

#ifndef FRICTIONQPOTFEM_CONFIG_H
#define FRICTIONQPOTFEM_CONFIG_H

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

#define FRICTIONQPOTFEM_VERSION_MAJOR 0
#define FRICTIONQPOTFEM_VERSION_MINOR 4
#define FRICTIONQPOTFEM_VERSION_PATCH 0

#define FRICTIONQPOTFEM_VERSION_AT_LEAST(x, y, z) \
    (FRICTIONQPOTFEM_VERSION_MAJOR > x || (FRICTIONQPOTFEM_VERSION_MAJOR >= x && \
    (FRICTIONQPOTFEM_VERSION_MINOR > y || (FRICTIONQPOTFEM_VERSION_MINOR >= y && \
                                           FRICTIONQPOTFEM_VERSION_PATCH >= z))))

#define FRICTIONQPOTFEM_VERSION(x, y, z) \
    (FRICTIONQPOTFEM_VERSION_MAJOR == x && \
     FRICTIONQPOTFEM_VERSION_MINOR == y && \
     FRICTIONQPOTFEM_VERSION_PATCH == z)

#endif
