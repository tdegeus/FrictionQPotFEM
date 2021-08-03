/**
\file UniformMultiLayerLeverDrive2d.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_HPP
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_HPP

#include "UniformMultiLayerLeverDrive2d.h"

namespace FrictionQPotFEM {
namespace UniformMultiLayerLeverDrive2d {

inline std::vector<std::string> version_dependencies()
{
    return Generic2d::version_dependencies();
}

inline System::System(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const std::vector<xt::xtensor<size_t, 1>>& elem,
    const std::vector<xt::xtensor<size_t, 1>>& node,
    const xt::xtensor<bool, 1>& layer_is_plastic)
{
    this->init(coor, conn, dofs, iip, elem, node, layer_is_plastic);
}

inline void System::updated_u()
{
    this->computeForceMaterial();
    this->computeForceDrive();
    // todo: ....
}

} // namespace UniformMultiLayerLeverDrive2d
} // namespace FrictionQPotFEM

#endif
