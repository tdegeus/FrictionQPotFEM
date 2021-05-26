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
    const xt::xtensor<bool, 1>& layer_is_plastic)
{
    this->init(coor, conn, dofs, iip, elem, node, layer_is_plastic);
}

inline void System::init(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const std::vector<xt::xtensor<size_t, 1>>& elem,
    const std::vector<xt::xtensor<size_t, 1>>& node,
    const xt::xtensor<bool, 1>& layer_is_plastic)
{
    FRICTIONQPOTFEM_ASSERT(layer_is_plastic.size() == elem.size());
    FRICTIONQPOTFEM_ASSERT(layer_is_plastic.size() == node.size());

    m_n_layer = node.size();
    m_layer_node = node;
    m_layer_elem = elem;
    m_layer_is_plastic = layer_is_plastic;

    m_layer_ubar_set.resize({m_n_layer, size_t(2)});
    m_layer_ubar_target.resize({m_n_layer, size_t(2)});
    m_layer_ubar_value.resize({m_n_layer, size_t(2)});
    m_layer_dV1.resize({m_n_layer, size_t(2)});
    m_slice_index.resize({m_n_layer});

    m_layer_ubar_set.fill(false);

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
    m_uq = m_quad.allocate_qtensor<1>(0.0);
    m_dV = m_quad.dV();

    size_t nip = m_quad.nip();
    m_layer_dV1.fill(0.0);

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (auto& e : m_layer_elem[i]) {
            for (size_t q = 0; q < nip; ++q) {
                for (size_t d = 0; d < 2; ++d) {
                    m_layer_dV1(i, d) += m_dV(e, q);
                }
            }
        }
    }

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

inline xt::xtensor<double, 2> System::layerUbar()
{
    m_layer_ubar_value.fill(0.0);
    size_t nip = m_quad.nip();

    m_vector.asElement(m_u, m_ue);
    m_quad.interpQuad_vector(m_ue, m_uq);

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (auto& e : m_layer_elem[i]) {
            for (size_t d = 0; d < 2; ++d) {
                for (size_t q = 0; q < nip; ++q) {
                    m_layer_ubar_value(i, d) += m_uq(e, q, d) * m_dV(e, q);
                }
            }
        }
    }

    m_layer_ubar_value /= m_layer_dV1;


    return m_layer_ubar_value;
}

template <class S, class T>
inline void System::layerSetUbar(const S& ubar, const T& prescribe)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(ubar, m_layer_ubar_target.shape()));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(prescribe, m_layer_ubar_set.shape()));
    m_layer_ubar_target = ubar;
    m_layer_ubar_set = prescribe;
    this->computeForceDrive();
}

template <class S, class T>
inline void System::layerSetDistributeUbar(const S& ubar, const T& prescribe)
{
    this->layerSetUbar(ubar, prescribe);

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (size_t d = 0; d < 2; ++d) {
            if (m_layer_ubar_set(i, d)) {
                double du = m_layer_ubar_target(i, d) - m_layer_ubar_value(i, d);
                for (auto& n : m_layer_node[i]) {
                    m_u(n, d) += du;
                }
            }
        }
    }

    this->updated_u();
}

template <class T, class S>
inline void System::addAffineSimpleShear(double delta_gamma, const S& prescribe, const T& height)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(prescribe, m_layer_ubar_set.shape()));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(height, {m_n_layer}));

    m_layer_ubar_set = prescribe;

    for (size_t i = 0; i < m_n_layer; ++i) {
        m_layer_ubar_target(i, 0) += 2.0 * delta_gamma * height(i);
    }

    for (size_t n = 0; n < m_nnode; ++n) {
        m_u(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - m_coor(0, 1));
    }

    this->updated_u();
}

inline void System::setDriveStiffness(double k, bool symmetric)
{
    m_drive_spring_symmetric = symmetric;
    m_k_drive = k;
}

inline void System::computeForceDrive()
{
    // compute the average displacement per layer
    // (skip all layers that are not driven, the average displacement will never be used)

    m_layer_ubar_value.fill(0.0);
    size_t nip = m_quad.nip();

    m_vector.asElement(m_u, m_ue);
    m_quad.interpQuad_vector(m_ue, m_uq);

    for (size_t i = 0; i < m_n_layer; ++i) {
        if (m_layer_ubar_set(i, 0) || m_layer_ubar_set(i, 1)) {
            for (auto& e : m_layer_elem[i]) {
                for (size_t d = 0; d < 2; ++d) {
                    for (size_t q = 0; q < nip; ++q) {
                        m_layer_ubar_value(i, d) += m_uq(e, q, d) * m_dV(e, q);
                    }
                }
            }
        }
    }

    m_layer_ubar_value /= m_layer_dV1;

    // compute the driving force

    m_uq.fill(0.0);

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (size_t d = 0; d < 2; ++d) {
            if (m_layer_ubar_set(i, d)) {
                double f = m_k_drive * (m_layer_ubar_value(i, d) - m_layer_ubar_target(i, d));
                if (m_drive_spring_symmetric || f < 0) { // buckle under compression
                    for (auto& e : m_layer_elem[i]) {
                        for (size_t q = 0; q < nip; ++q) {
                            m_uq(e, q, d) = f;
                        }
                    }
                }
            }
        }
    }

    m_quad.int_N_vector_dV(m_uq, m_ue);
    m_vector.asNode(m_vector.AssembleDofs(m_ue), m_fdrive);
}

inline xt::xtensor<double, 2> System::fdrivespring() const
{
    xt::xtensor<double, 2> ret = xt::zeros<double>({m_n_layer, size_t(2)});

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (size_t d = 0; d < 2; ++d) {
            if (m_layer_ubar_set(i, d)) {
                ret(i, d) = m_k_drive * (m_layer_ubar_target(i, d) - m_layer_ubar_value(i, d));
            }
        }
    }

    return ret;
}

inline void System::updated_u()
{
    this->computeForceMaterial();
    this->computeForceDrive();
}

inline xt::xtensor<double, 2> System::fdrive() const
{
    return m_fdrive;
}

inline void System::computeInternalExternalResidualForce()
{
    xt::noalias(m_fint) = m_fdrive + m_fmaterial + m_fdamp;
    m_vector.copy_p(m_fint, m_fext);
    xt::noalias(m_fres) = m_fext - m_fint;
}

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#endif
