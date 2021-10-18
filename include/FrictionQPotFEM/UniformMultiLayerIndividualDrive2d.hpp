/**
\file
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

    m_layerdrive_active.resize({m_n_layer, size_t(2)});
    m_layerdrive_targetubar.resize({m_n_layer, size_t(2)});
    m_layer_ubar.resize({m_n_layer, size_t(2)});
    m_layer_dV1.resize({m_n_layer, size_t(2)});
    m_slice_index.resize({m_n_layer});

    m_layerdrive_targetubar.fill(0.0);
    m_layerdrive_active.fill(false);

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

    xt::xtensor<size_t, 1> plas(std::array<size_t, 1>{n_elem_plas});
    xt::xtensor<size_t, 1> elas(std::array<size_t, 1>{n_elem_elas});
    m_slice_plas.resize({n_layer_plas + size_t(1)});
    m_slice_elas.resize({n_layer_elas + size_t(1)});
    m_slice_plas(0) = 0;
    m_slice_elas(0) = 0;
    m_N = 0;

    n_elem_plas = 0;
    n_elem_elas = 0;
    n_layer_plas = 0;
    n_layer_elas = 0;

    for (size_t i = 0; i < m_n_layer; ++i) {
        if (m_layer_is_plastic(i)) {
            size_t l = m_slice_plas(n_layer_plas);
            size_t u = n_elem_plas + elem[i].size();

            m_slice_index(i) = n_layer_plas;
            m_slice_plas(n_layer_plas + 1) = u;

            xt::view(plas, xt::range(l, u)) = elem[i];

            n_elem_plas += elem[i].size();
            n_layer_plas++;

            FRICTIONQPOTFEM_REQUIRE(m_N == elem[i].size() || m_N == 0);
            m_N = elem[i].size();
        }
        else {
            size_t l = m_slice_elas(n_layer_elas);
            size_t u = n_elem_elas + elem[i].size();

            m_slice_index(i) = n_layer_elas;
            m_slice_elas(n_layer_elas + 1) = u;

            xt::view(elas, xt::range(l, u)) = elem[i];

            n_elem_elas += elem[i].size();
            n_layer_elas++;
        }
    }

    this->initHybridSystem(coor, conn, dofs, iip, elas, plas);

    m_fdrive = m_vector.allocate_nodevec(0.0);
    m_ud = m_vector.allocate_dofval(0.0);
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

inline size_t System::N() const
{
    return m_N;
}

inline std::string System::type() const
{
    return "FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System";
}

inline size_t System::nlayer() const
{
    return m_n_layer;
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

inline xt::xtensor<bool, 1> System::layerIsPlastic() const
{
    return m_layer_is_plastic;
}

inline void System::layerSetDriveStiffness(double k, bool symmetric)
{
    m_drive_spring_symmetric = symmetric;
    m_drive_k = k;
    this->computeLayerUbarActive();
    this->computeForceFromTargetUbar();
}

template <class S, class T>
inline void System::initEventDriven(const S& delta_ubar, const T& active)
{
    // backup system

    auto u0 = m_u;
    auto v0 = m_v;
    auto a0 = m_a;
    auto active0 = m_layerdrive_active;
    auto ubar0 = m_layerdrive_targetubar;
    auto t0 = m_t;
    xt::xtensor<double, 3> epsy0(std::array<size_t, 3>{m_nelem_plas, m_nip, 2});

    for (size_t e = 0; e < m_nelem_plas; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {
            auto cusp = m_material_plas.refCusp({e, q});
            auto epsy = cusp.epsy();
            epsy0(e, q, 0) = epsy(0);
            epsy0(e, q, 1) = epsy(1);
            epsy(1) = std::numeric_limits<double>::max();
            epsy(0) = -epsy(1);
            cusp.reset_epsy(epsy, false);
        }
    }

    // perturbation

    m_u.fill(0.0);
    m_v.fill(0.0);
    m_a.fill(0.0);
    this->layerSetTargetActive(active);
    this->layerSetTargetUbar(delta_ubar);
    this->minimise();

    m_pert_layerdrive_active = active;
    m_pert_layerdrive_targetubar = delta_ubar;
    this->eventDriven_setDeltaU(m_u);

    // restore system

    for (size_t e = 0; e < m_nelem_plas; ++e) {
        for (size_t q = 0; q < m_nip; ++q) {
            auto cusp = m_material_plas.refCusp({e, q});
            auto epsy = cusp.epsy();
            epsy(0) = epsy0(e, q, 0);
            epsy(1) = epsy0(e, q, 1);
            cusp.reset_epsy(epsy, false);
        }
    }

    this->setU(u0);
    this->setV(v0);
    this->setA(a0);
    this->layerSetTargetActive(active0);
    this->layerSetTargetUbar(ubar0);
    this->setT(t0);
}

template <class S, class T, class U>
inline void System::initEventDriven(const S& ubar, const T& active, const U& u)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(ubar, m_layerdrive_targetubar.shape()));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, m_u.shape()));
    m_pert_layerdrive_active = active;
    m_pert_layerdrive_targetubar = ubar;
    this->eventDriven_setDeltaU(u);
}

inline double System::eventDrivenStep(double deps_kick, bool kick, int direction)
{
    double c = Generic2d::System::eventDrivenStep(deps_kick, kick, direction);
    this->layerSetTargetUbar(m_layerdrive_targetubar + c * m_pert_layerdrive_targetubar);
    return c;
}

template <class T>
inline void System::layerSetTargetActive(const T& active)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));
    m_layerdrive_active = active;
    this->computeLayerUbarActive();
    this->computeForceFromTargetUbar();
}

inline xt::xtensor<double, 2> System::layerUbar()
{
    // Recompute needed because computeLayerUbarActive() only computes the average
    // on layers with an active spring.
    // This function, in contrast, returns the average on all layers.

    m_layer_ubar.fill(0.0);
    size_t nip = m_quad.nip();

    m_vector.asElement(m_u, m_ue);
    m_quad.interpQuad_vector(m_ue, m_uq);

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (auto& e : m_layer_elem[i]) {
            for (size_t d = 0; d < 2; ++d) {
                for (size_t q = 0; q < nip; ++q) {
                    m_layer_ubar(i, d) += m_uq(e, q, d) * m_dV(e, q);
                }
            }
        }
    }

    m_layer_ubar /= m_layer_dV1;

    return m_layer_ubar;
}

inline xt::xtensor<double, 2> System::layerTargetUbar() const
{
    return m_layerdrive_targetubar;
}

inline xt::xtensor<bool, 2> System::layerTargetActive() const
{
    return m_layerdrive_active;
}

template <class T>
inline void System::layerSetTargetUbar(const T& ubar)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(ubar, m_layerdrive_targetubar.shape()));
    m_layerdrive_targetubar = ubar;
    this->computeForceFromTargetUbar(); // the average displacement and other forces do not change
}

template <class S, class T>
inline void System::layerSetUbar(const S& ubar, const T& prescribe)
{
    auto current = this->layerUbar();

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (size_t d = 0; d < 2; ++d) {
            if (prescribe(i, d)) {
                double du = ubar(i, d) - current(i, d);
                for (auto& n : m_layer_node[i]) {
                    m_u(n, d) += du;
                }
            }
        }
    }

    this->computeLayerUbarActive();
    this->computeForceFromTargetUbar();
    if (m_allset) {
        this->computeForceMaterial();
    }
}

inline void System::addAffineSimpleShear(double delta_gamma)
{
    for (size_t n = 0; n < m_nnode; ++n) {
        m_u(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - m_coor(0, 1));
    }

    this->computeLayerUbarActive();
    this->computeForceFromTargetUbar();
    if (m_allset) {
        this->computeForceMaterial();
    }
}

template <class T>
inline void System::layerTagetUbar_addAffineSimpleShear(double delta_gamma, const T& height)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(height, {m_n_layer}));
    for (size_t i = 0; i < m_n_layer; ++i) {
        m_layerdrive_targetubar(i, 0) += 2.0 * delta_gamma * height(i);
    }
    this->computeForceFromTargetUbar(); // the average displacement and other forces do not change
}

inline void System::computeLayerUbarActive()
{
    m_layer_ubar.fill(0.0);
    size_t nip = m_quad.nip();

    m_vector.asElement(m_u, m_ue);
    m_quad.interpQuad_vector(m_ue, m_uq);

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (size_t d = 0; d < 2; ++d) {
            if (m_layerdrive_active(i, d)) {
                for (auto& e : m_layer_elem[i]) {
                    for (size_t q = 0; q < nip; ++q) {
                        m_layer_ubar(i, d) += m_uq(e, q, d) * m_dV(e, q);
                    }
                }
            }
        }
    }

    m_layer_ubar /= m_layer_dV1;
}

inline void System::computeForceFromTargetUbar()
{
    m_uq.fill(0.0); // pre-allocated value that an be freely used
    size_t nip = m_quad.nip();

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (size_t d = 0; d < 2; ++d) {
            if (m_layerdrive_active(i, d)) {
                double f = m_drive_k * (m_layer_ubar(i, d) - m_layerdrive_targetubar(i, d));
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
    m_vector.assembleDofs(m_ue, m_ud);
    m_vector.asNode(m_ud, m_fdrive);
}

inline xt::xtensor<double, 2> System::layerFdrive() const
{
    xt::xtensor<double, 2> ret = xt::zeros<double>({m_n_layer, size_t(2)});

    for (size_t i = 0; i < m_n_layer; ++i) {
        for (size_t d = 0; d < 2; ++d) {
            if (m_layerdrive_active(i, d)) {
                ret(i, d) = m_drive_k * (m_layerdrive_targetubar(i, d) - m_layer_ubar(i, d));
            }
        }
    }

    return ret;
}

inline void System::updated_u()
{
    this->computeLayerUbarActive();
    this->computeForceFromTargetUbar();
    this->computeForceMaterial();
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
