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
    const xt::xtensor<bool, 1>& layer_is_plastic,
    const xt::xtensor<bool, 1>& layer_is_plastic)
{
    this->init(coor, conn, dofs, elem, layer_is_plastic)
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

    size_t nplas = 0;
    size_t nelas = 0;

    for (size_t i = 0; i < elem.size(); ++i) {
        if (m_layer_is_plastic(i)) {
            nplas += elem[i].size();
        }
        else {
            nelas += elem[i].size();
        }
    }

    xt::xtensor<size_t, 1> plas = xt::empty<size_t>({nplas});
    xt::xtensor<size_t, 1> elas = xt::empty<size_t>({nelas});
    m_slice_plas = xt::empty<size_t>({nplas + size_t(1)});
    m_slice_elas = xt::empty<size_t>({nelas + size_t(1)});
    m_slice_plas(0) = 0;
    m_slice_elas(0) = 0;

    nplas = 0;
    nelas = 0;

    for (size_t i = 0; i < elem.size(); ++i) {
        if (m_layer_is_plastic(i)) {
            m_slice_index(i) = nplas;
            m_slice_plas(nplas + 1) = nplas + elem[i].size();
            xt::view(plas, xt::range(m_slice_plas(nplas), m_slice_plas(nplas + 1))) = elem[i];
            nplas += elem[i].size();
        }
        else {
            m_slice_index(i) = nelas;
            m_slice_elas(nelas + 1) = nelas + elem[i].size();
            xt::view(elas, xt::range(m_slice_elas(nelas), m_slice_elas(nelas + 1))) = elem[i];
            nelas += elem[i].size();
        }
    }

    this->initHybridSystem(coor, conn, dofs, iip, elas, plas);
}

inline xt::xtensor<size_t, 1> System::layerElements(size_t i) const
{
    FRICTIONQPOTFEM_ASSERT(i < m_layer_is_plastic.size());
    size_t j = m_slice_index(i);

    if (m_layer_is_plastic(i)) {
        return xt::view(m_elem_plas, xt::range(m_slice_plas(j), m_slice_plas(j + 1)));
    }

    return xt::view(m_elem_elas, xt::range(m_slice_elas(j), m_slice_elas(j + 1)));
}

inline xt::xtensor<size_t, 1> System::layerNodes(size_t i) const
{
    auto elem = this->layerElements(i);
    return xt::unique(xt::view(m_conn, xt::keep(elem)));
}

inline bool System::layerIsPlastic(size_t i) const
{
    FRICTIONQPOTFEM_ASSERT(i < m_layer_is_plastic.size());
    return m_layer_is_plastic(i);
}

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#endif
