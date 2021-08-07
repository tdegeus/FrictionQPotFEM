/**
 *  \file UniformMultiLayerLeverDrive2d.hpp
 *  \copyright Copyright 2020. Tom de Geus. All rights reserved.
 *  \license This project is released under the GNU Public License (MIT).
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

    m_lever_hi.resize({m_n_layer});
    m_lever_hi.fill(0.0);

    m_lever_hi_H.resize({m_n_layer});
    m_lever_hi_H.fill(0.0);

    m_lever_hi_H_2.resize({m_n_layer});
    m_lever_hi_H_2.fill(0.0);

    m_lever_H = 0.0;
    m_lever_target = 0.0;
    m_lever_u = 0.0;
}

template <class T>
inline void System::setLeverProperties(double H, const T& hi)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(hi, m_lever_hi.shape()));
    m_lever_hi = hi;
    m_lever_H = H;

    m_lever_hi_H = m_lever_hi / H;
    m_lever_hi_H_2 = xt::pow(m_lever_hi / H, 2.0);

    this->updated_u(); // updates forces
}

inline void System::setLeverTarget(double u)
{
    m_lever_target = u;
    this->updated_u(); // updates target average displacement per layer, and forces
}

inline double System::leverTarget() const
{
    return m_lever_target;
}

inline double System::leverPosition() const
{
    return m_lever_u;
}

inline void System::updated_u()
{
    this->computeLayerUbarActive();

    // Position of the lever based on equilibrium

    m_lever_u = m_lever_target;
    double n = 1.0;

    for (size_t i = 0; i < m_n_layer; ++i) {
        if (m_layerdrive_active(i, 0)) {
            m_lever_u += m_layer_ubar(i, 0) * m_lever_hi_H(i);
            n += m_lever_hi_H_2(i);
        }
    }

    m_lever_u /= n;

    // Update position of driving springs

    for (size_t i = 0; i < m_n_layer; ++i) {
        if (m_layerdrive_active(i, 0)) {
            m_layerdrive_targetubar(i, 0) = m_lever_hi_H(i) * m_lever_u;
        }
    }

    // Update forces
    this->computeForceFromTargetUbar();
    this->computeForceMaterial();
}

} // namespace UniformMultiLayerLeverDrive2d
} // namespace FrictionQPotFEM

#endif
