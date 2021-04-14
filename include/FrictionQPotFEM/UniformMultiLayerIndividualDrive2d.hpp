/**
\file UniformMultiLayerIndividualDrive2d.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_HPP
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_HPP

#include "UniformMultiLayerIndividualDrive2d.h"

namespace FrictionQPotFEM {
namespace UniformMultiLayerIndividualDrive2d {

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
    const xt::xtensor<bool, 1>& layer_is_plastic,
    const xt::xtensor<bool, 1>& node_is_virtual)
{
    this->init(coor, conn, dofs, iip, elem, node, layer_is_plastic, node_is_virtual);
}

inline void System::init(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const std::vector<xt::xtensor<size_t, 1>>& elem,
    const std::vector<xt::xtensor<size_t, 1>>& node,
    const xt::xtensor<bool, 1>& layer_is_plastic,
    const xt::xtensor<bool, 1>& node_is_virtual)
{
    FRICTIONQPOTFEM_ASSERT(layer_is_plastic.size() == elem.size());
    FRICTIONQPOTFEM_ASSERT(layer_is_plastic.size() == node.size());
    FRICTIONQPOTFEM_ASSERT(node_is_virtual.size() == coor.shape(0));

    m_n_layer = node.size();
    m_layer_node = node;
    m_layer_elem = elem;
    m_node_is_virtual = node_is_virtual;
    m_layer_is_plastic = layer_is_plastic;

    m_layer_has_drive.resize({m_n_layer});
    m_layer_ubar.resize({m_n_layer, size_t(2)});
    m_slice_index.resize({m_n_layer});

    m_layer_has_drive.fill(false);

    size_t n_elem_plas = 0;
    size_t n_elem_elas = 0;
    size_t n_layer_plas = 0;
    size_t n_layer_elas = 0;

    for (size_t i = 0; i < elem.size(); ++i) {
        if (m_layer_is_plastic(i)) {
            n_elem_plas += elem[i].size();
            n_layer_plas++;
        }
        else {
            n_elem_elas += elem[i].size();
            n_layer_elas++;
        }
    }

    xt::xtensor<size_t, 1> plas = xt::empty<size_t>({n_elem_plas});
    xt::xtensor<size_t, 1> elas = xt::empty<size_t>({n_elem_elas});
    m_slice_plas = xt::empty<size_t>({n_layer_plas + size_t(1)});
    m_slice_elas = xt::empty<size_t>({n_layer_elas + size_t(1)});
    m_slice_plas(0) = 0;
    m_slice_elas(0) = 0;

    n_elem_plas = 0;
    n_elem_elas = 0;
    n_layer_plas = 0;
    n_layer_elas = 0;

    for (size_t i = 0; i < m_n_layer; ++i) {
        if (m_layer_is_plastic(i)) {
            m_slice_index(i) = n_layer_plas;
            m_slice_plas(n_layer_plas + 1) = n_elem_plas + elem[i].size();
            xt::view(plas, xt::range(m_slice_plas(n_layer_plas), m_slice_plas(n_layer_plas + 1))) = elem[i];
            n_elem_plas += elem[i].size();
            n_layer_plas++;
        }
        else {
            m_slice_index(i) = n_layer_elas;
            m_slice_elas(n_layer_elas + 1) = n_elem_elas + elem[i].size();
            xt::view(elas, xt::range(m_slice_elas(n_layer_elas), m_slice_elas(n_layer_elas + 1))) = elem[i];
            n_elem_elas += elem[i].size();
            n_layer_elas++;
        }
    }

    this->initHybridSystem(coor, conn, dofs, iip, elas, plas);

    m_fdrive = m_vector.allocate_nodevec(0.0);

    // sanity check nodes per layer
    #ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
    for (size_t i = 0; i < m_n_layer; ++i) {
        auto e = this->layerElements(i);
        auto n = xt::unique(xt::view(m_conn, xt::keep(e)));
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(xt::sort(n), xt::sort(node[i]))));
    }
    #endif

    // sanity check elements per layer + slice of elas/plas element sets
    #ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
    for (size_t i = 0; i < m_n_layer; ++i) {
        xt::xtensor<size_t, 1> e;
        size_t j = m_slice_index(i);
        if (m_layer_is_plastic(i)) {
            e = xt::view(m_elem_plas, xt::range(m_slice_plas(j), m_slice_plas(j + 1)));
        }
        else {
            e = xt::view(m_elem_elas, xt::range(m_slice_elas(j), m_slice_elas(j + 1)));
        }
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(xt::sort(e), xt::sort(elem[i]))));
    }
    #endif
}

inline xt::xtensor<size_t, 1> System::layerElements(size_t i) const
{
    FRICTIONQPOTFEM_ASSERT(i < m_n_layer);
    return m_layer_elem[i];
}

inline xt::xtensor<size_t, 1> System::layerNodes(size_t i) const
{
    FRICTIONQPOTFEM_ASSERT(i < m_n_layer);
    return m_layer_node[i];
}

inline bool System::layerIsPlastic(size_t i) const
{
    FRICTIONQPOTFEM_ASSERT(i < m_n_layer);
    return m_layer_is_plastic(i);
}

inline void System::layerSetUbar(size_t i, xt::xtensor<double, 1>& ubar)
{
    FRICTIONQPOTFEM_ASSERT(i < m_n_layer);
    FRICTIONQPOTFEM_ASSERT(ubar.size() == 2);
    for (size_t d = 0; d < 2; ++d) {
        m_layer_ubar(i, d) = ubar(d);
    }
    m_layer_has_drive(i) = true;
    this->computeForceDrive();
}

inline void System::setDriveStiffness(double k)
{
    m_k_drive = k;
}

inline void System::computeForceDrive()
{
    m_fdrive.fill(0.0);

    for (size_t i = 0; i < m_n_layer; ++i) {

        if (!m_layer_has_drive(i)) {
            continue;
        }

        std::array<double, 2> ubar = {0.0, 0.0};
        size_t norm = 0;

        for (auto& n : m_layer_node[i]) {
            if (!m_node_is_virtual(n)) {
                for (size_t d = 0; d < 2; ++d) {
                    ubar[d] += m_u(n, d);
                }
                norm++;
            }
        }

        for (size_t d = 0; d < 2; ++d) {
            ubar[d] /= static_cast<double>(norm);
        }

        std::array<double, 2> f;
        for (size_t d = 0; d < 2; ++d) {
            f[d] = m_k_drive * (m_layer_ubar(i, d) - ubar[d]);
        }

        for (auto& n : m_layer_node[i]) {
            if (!m_node_is_virtual(n)) {
                for (size_t d = 0; d < 2; ++d) {
                    m_fdrive(n, d) += f[d];
                }
            }
        }
    }
}

inline void System::setU(const xt::xtensor<double, 2>& u)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, {m_nnode, m_ndim}));
    xt::noalias(m_u) = u;
    this->computeForceMaterial();
    this->computeForceDrive();
}

inline xt::xtensor<double, 2> System::fdrive() const
{
    return m_fdrive;
}

inline void System::timeStep()
{
    FRICTIONQPOTFEM_ASSERT(m_allset);

    // history

    m_t += m_dt;

    xt::noalias(m_v_n) = m_v;
    xt::noalias(m_a_n) = m_a;

    // new displacement

    xt::noalias(m_u) = m_u + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
    this->computeForceMaterial();
    this->computeForceDrive();

    // estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;

    m_D.dot(m_v, m_fdamp);

    // compute residual force & solve

    xt::noalias(m_fint) = m_fdrive + m_fmaterial + m_fdamp;

    m_vector.copy_p(m_fint, m_fext);

    xt::noalias(m_fres) = m_fext - m_fint;

    m_M.solve(m_fres, m_a);

    // re-estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);

    m_D.dot(m_v, m_fdamp);

    // compute residual force & solve

    xt::noalias(m_fint) = m_fdrive + m_fmaterial + m_fdamp;

    m_vector.copy_p(m_fint, m_fext);

    xt::noalias(m_fres) = m_fext - m_fint;

    m_M.solve(m_fres, m_a);

    // new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);

    m_D.dot(m_v, m_fdamp);

    // compute residual force & solve

    xt::noalias(m_fint) = m_fdrive + m_fmaterial + m_fdamp;

    m_vector.copy_p(m_fint, m_fext);

    xt::noalias(m_fres) = m_fext - m_fint;

    m_M.solve(m_fres, m_a);
}

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#endif
