/**
\file
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
    this->init_lever(coor, conn, dofs, iip, elem, node, layer_is_plastic);
}

inline void System::init_lever(
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

inline std::string System::type() const
{
    return "FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System";
}

template <class T>
inline void System::setLeverProperties(double H, const T& hi)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(hi, m_lever_hi.shape()));
    m_lever_set = true;
    m_lever_hi = hi;
    m_lever_H = H;

    m_lever_hi_H = m_lever_hi / H;
    m_lever_hi_H_2 = xt::pow(m_lever_hi / H, 2.0);

    this->updated_u(); // updates forces
}

inline void System::setLeverTarget(double xdrive)
{
    m_lever_target = xdrive;
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

template <class T>
inline void System::initEventDriven(double xlever, const T& active)
{
    FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));

    // backup system

    auto u0 = m_u;
    auto v0 = m_v;
    auto a0 = m_a;
    auto active0 = m_layerdrive_active;
    auto ubar0 = m_layerdrive_targetubar;
    auto xdrive0 = m_lever_target;
    auto t0 = m_t;
    std::vector<std::vector<xt::xtensor<double, 1>>> epsy0;
    epsy0.resize(m_nelem_plas);
    double i = std::numeric_limits<double>::max();
    xt::xtensor<double, 1> epsy_elas = {-i, i};

    for (size_t e = 0; e < m_nelem_plas; ++e) {
        epsy0[e].resize(m_nip);
        for (size_t q = 0; q < m_nip; ++q) {
            auto& cusp = m_material_plas.refCusp({e, q});
            epsy0[e][q] = cusp.epsy();
            cusp.reset_epsy(epsy_elas, false);
        }
    }

    // perturbation

    m_u.fill(0.0);
    m_v.fill(0.0);
    m_a.fill(0.0);
    this->updated_u();
    FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(this->plastic_CurrentIndex(), 0)));
    this->layerSetTargetActive(active);
    this->setLeverTarget(xlever);
    this->minimise();
    FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(this->plastic_CurrentIndex(), 0)));
    auto up = m_u;
    m_u.fill(0.0);

    // restore yield strains
    // to avoid running out-of-bounds there  the displacements had to be reset to zero
    // if the yield strains are result only later scaling is buggy because a typical yield strain
    // can be inaccurate

    for (size_t e = 0; e < m_nelem_plas; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {
            auto& cusp = m_material_plas.refCusp({e, q});
            cusp.reset_epsy(epsy0[e][q], false);
        }
    }

    // store result

    auto c = this->eventDriven_setDeltaU(up);
    m_pert_layerdrive_active = active;
    m_pert_layerdrive_targetubar = c * m_layerdrive_targetubar;
    m_pert_lever_target = c * xlever;

    // restore system

    this->setU(u0);
    this->setV(v0);
    this->setA(a0);
    this->layerSetTargetActive(active0);
    this->layerSetTargetUbar(ubar0);
    this->setLeverTarget(xdrive0);
    this->setT(t0);
}

template <class T, class U, class W>
inline double System::initEventDriven(double xlever, const T& active, const U& u, const W& ubar)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(ubar, m_layerdrive_targetubar.shape()));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, m_u.shape()));
    double c = this->eventDriven_setDeltaU(u);
    m_pert_layerdrive_active = active;
    m_pert_layerdrive_targetubar = c * ubar;
    m_pert_lever_target = c * xlever;
    return c;
}

inline double
System::eventDrivenStep(double deps, bool kick, int direction, bool yield_element, bool fallback)
{
    double c = Generic2d::System::eventDrivenStep(deps, kick, direction, yield_element, fallback);
    this->layerSetTargetUbar(m_layerdrive_targetubar + c * m_pert_layerdrive_targetubar);
    this->setLeverTarget(m_lever_target + c * m_pert_lever_target);
    return c;
}

inline double System::eventDriven_deltaLeverPosition() const
{
    return m_pert_lever_target;
}

inline void System::updated_u()
{
    FRICTIONQPOTFEM_ASSERT(m_lever_set);

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
