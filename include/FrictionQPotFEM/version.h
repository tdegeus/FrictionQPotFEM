/**
Version information.

\file version.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_VERSION_H
#define FRICTIONQPOTFEM_VERSION_H

#include "config.h"

/**
Current version.

Either:

-   Configure using CMake at install time. Internally uses::

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using::

        -DFRICTIONQPOTFEM_VERSION="`python -c "from setuptools_scm import get_version; print(get_version())"`"

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version string using
``setuptools_scm`` **unless** an environment ``PKG_VERSION`` is defined.
If ``PKG_VERSION`` is defined the version string will be read from that variable.
*/
#ifndef FRICTIONQPOTFEM_VERSION
#define FRICTIONQPOTFEM_VERSION "@FRICTIONQPOTFEM_VERSION@"
#endif

namespace FrictionQPotFEM {

/**
Return version string, e.g.::

    "0.8.0"

\return std::string
*/
inline std::string version();

} // namespace FrictionQPotFEM

#include "version.hpp"

#endif
