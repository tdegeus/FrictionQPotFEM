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
    m_layer_ubar.resize({m_n_layer});
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

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#endif
