/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_VERSION_HPP
#define FRICTIONQPOTFEM_VERSION_HPP

#include "version.h"

namespace FrictionQPotFEM {

namespace detail {

inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

inline std::string version()
{
    return detail::unquote(std::string(QUOTE(FRICTIONQPOTFEM_VERSION)));
}

} // namespace FrictionQPotFEM

#endif
