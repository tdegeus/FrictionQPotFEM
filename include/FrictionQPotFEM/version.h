/**
\file config.h

\brief
Defines version.

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM
*/

#ifndef FRICTIONQPOTFEM_VERSION_H
#define FRICTIONQPOTFEM_VERSION_H

/**
Current version.

Either:
-   Configure using CMake at install time.
-   Define externally using::

        -DFRICTIONQPOTFEM_VERSION="`python -c "from setuptools_scm import get_version; print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.
*/
#ifndef FRICTIONQPOTFEM_VERSION
#define FRICTIONQPOTFEM_VERSION "@FRICTIONQPOTFEM_VERSION@"
#endif

#include <string>

namespace FrictionQPotFEM {

/**
Return version string.

\return std::string
*/
inline std::string version()
{
    return std::string(FRICTIONQPOTFEM_VERSION);
}

} // namespace FrictionQPotFEM

#endif
