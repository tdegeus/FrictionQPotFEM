/**
\file Generic2d.hpp
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef FRICTIONQPOTFEM_GENERIC2D_HPP
#define FRICTIONQPOTFEM_GENERIC2D_HPP

#include "Generic2d.h"

namespace FrictionQPotFEM {
namespace Generic2d {

inline std::vector<std::string> version_dependencies()
{
    std::vector<std::string> ret;

    ret.push_back("frictionqpotfem=" + version());
    ret.push_back("goosefem=" + GooseFEM::version());
    ret.push_back("gmatelastoplasticqpot=" + GMatElastoPlasticQPot::version());
    ret.push_back("gmattensor=" + GMatTensor::version());
    ret.push_back("qpot=" + QPot::version());

    ret.push_back("xtensor=" +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MAJOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_MINOR))) + "." +
        detail::unquote(std::string(QUOTE(XTENSOR_VERSION_PATCH))));

    #if defined(GOOSEFEM_EIGEN) || defined(EIGEN_WORLD_VERSION)

        ret.push_back("eigen=" +
            detail::unquote(std::string(QUOTE(EIGEN_WORLD_VERSION))) + "." +
            detail::unquote(std::string(QUOTE(EIGEN_MAJOR_VERSION))) + "." +
            detail::unquote(std::string(QUOTE(EIGEN_MINOR_VERSION))));

    #endif

    return ret;
}

template <class C, class E, class L>
inline System::System(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

template <class C, class E, class L>
inline void System::initSystem(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elem_elastic,
        const L& elem_plastic)
{
    m_coor = coor;
    m_conn = conn;
    m_dofs = dofs;
    m_iip = iip;
    m_elem_elas = elem_elastic;
    m_elem_plas = elem_plastic;

    m_nnode = m_coor.shape(0);
    m_ndim = m_coor.shape(1);
    m_nelem = m_conn.shape(0);
    m_nne = m_conn.shape(1);

    m_nelem_elas = m_elem_elas.size();
    m_nelem_plas = m_elem_plas.size();
    m_N = m_nelem_plas;

    #ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
        // check that "elem_plastic" and "elem_plastic" together span all elements
        xt::xtensor<size_t, 1> elem = xt::concatenate(xt::xtuple(m_elem_elas, m_elem_plas));
        FRICTIONQPOTFEM_ASSERT(xt::sort(elem) == xt::arange<size_t>(m_nelem));
        // check that all "iip" or in "dofs"
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::isin(m_iip, m_dofs)));
    #endif

    m_vector = GooseFEM::VectorPartitioned(m_conn, m_dofs, m_iip);

    m_quad = GooseFEM::Element::Quad4::Quadrature(m_vector.AsElement(m_coor));
    m_nip = m_quad.nip();

    m_u = m_vector.allocate_nodevec(0.0);
    m_v = m_vector.allocate_nodevec(0.0);
    m_a = m_vector.allocate_nodevec(0.0);
    m_v_n = m_vector.allocate_nodevec(0.0);
    m_a_n = m_vector.allocate_nodevec(0.0);

    m_ue = m_vector.allocate_elemvec(0.0);
    m_fe = m_vector.allocate_elemvec(0.0);

    m_fmaterial = m_vector.allocate_nodevec(0.0);
    m_fdamp = m_vector.allocate_nodevec(0.0);
    m_fint = m_vector.allocate_nodevec(0.0);
    m_fext = m_vector.allocate_nodevec(0.0);
    m_fres = m_vector.allocate_nodevec(0.0);

    m_Eps = m_quad.allocate_qtensor<2>(0.0);
    m_Sig = m_quad.allocate_qtensor<2>(0.0);

    m_M = GooseFEM::MatrixDiagonalPartitioned(m_conn, m_dofs, m_iip);
    m_D = GooseFEM::MatrixDiagonal(m_conn, m_dofs);
    m_K = GooseFEM::MatrixPartitioned(m_conn, m_dofs, m_iip);

    m_material = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem, m_nip});
}

inline void System::evalAllSet()
{
    m_allset = m_set_M && m_set_D && m_set_elas && m_set_plas && m_dt > 0.0;
}

template <class T>
inline void System::setMassMatrix(const T& val_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_M);
    FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);

    GooseFEM::Element::Quad4::Quadrature nodalQuad(
        m_vector.AsElement(m_coor),
        GooseFEM::Element::Quad4::Nodal::xi(),
        GooseFEM::Element::Quad4::Nodal::w());

    xt::xtensor<double, 2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});
    for (size_t q = 0; q < nodalQuad.nip(); ++q) {
        xt::view(val_quad, xt::all(), q) = val_elem;
    }

    m_M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    m_set_M = true;
    this->evalAllSet();
}

template <class T>
inline void System::setDampingMatrix(const T& val_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_D);
    FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);

    GooseFEM::Element::Quad4::Quadrature nodalQuad(
        m_vector.AsElement(m_coor),
        GooseFEM::Element::Quad4::Nodal::xi(),
        GooseFEM::Element::Quad4::Nodal::w());

    xt::xtensor<double, 2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});
    for (size_t q = 0; q < nodalQuad.nip(); ++q) {
        xt::view(val_quad, xt::all(), q) = val_elem;
    }

    m_D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    m_set_D = true;
    this->evalAllSet();
}

inline void System::initMaterial()
{
    if (!(m_set_elas && m_set_plas)) {
        return;
    }

    FRICTIONQPOTFEM_REQUIRE(
        xt::all(xt::not_equal(m_material.type(), GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));

    m_material.setStrain(m_Eps);

    m_K.assemble(m_quad.Int_gradN_dot_tensor4_dot_gradNT_dV(m_material.Tangent()));
}

inline void System::setElastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_elas);
    FRICTIONQPOTFEM_ASSERT(K_elem.size() == m_nelem_elas);
    FRICTIONQPOTFEM_ASSERT(G_elem.size() == m_nelem_elas);

    if (m_nelem_elas > 0) {
        xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
        xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
        xt::view(I, xt::keep(m_elem_elas), xt::all()) = 1ul;
        xt::view(idx, xt::keep(m_elem_elas), xt::all()) = xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});
        m_material.setElastic(I, idx, K_elem, G_elem);
    }

    m_set_elas = true;
    this->evalAllSet();
    this->initMaterial();
}

inline void System::setPlastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem,
    const xt::xtensor<double, 2>& epsy_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_plas);
    FRICTIONQPOTFEM_ASSERT(K_elem.size() == m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(G_elem.size() == m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(epsy_elem.shape(0) == m_nelem_plas);

    if (m_nelem_plas > 0) {
        xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
        xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
        xt::view(I, xt::keep(m_elem_plas), xt::all()) = 1ul;
        xt::view(idx, xt::keep(m_elem_plas), xt::all()) = xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});
        m_material.setCusp(I, idx, K_elem, G_elem, epsy_elem);
    }

    m_set_plas = true;
    this->evalAllSet();
    this->initMaterial();
}

inline bool System::isHomogeneousElastic() const
{
    auto K = m_material.K();
    auto G = m_material.G();

    return xt::allclose(K, K.data()[0]) && xt::allclose(G, G.data()[0]);
}

inline void System::setT(double t)
{
    m_t = t;
}

inline void System::setDt(double dt)
{
    m_dt = dt;
    this->evalAllSet();
}

template <class T>
inline void System::setU(const T& u)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, {m_nnode, m_ndim}));
    xt::noalias(m_u) = u;
    this->updated_u();
}

inline void System::updated_u()
{
    this->computeForceMaterial();
}

template <class T>
inline void System::setV(const T& v)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, {m_nnode, m_ndim}));
    xt::noalias(m_v) = v;
}

inline void System::updated_v()
{
    m_D.dot(m_v, m_fdamp);
}

template <class T>
inline void System::setA(const T& a)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(a, {m_nnode, m_ndim}));
    xt::noalias(m_a) = a;
}

template <class T>
inline void System::setFext(const T& fext)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(fext, {m_nnode, m_ndim}));
    xt::noalias(m_fext) = fext;
}

inline void System::quench()
{
    m_v.fill(0.0);
    m_a.fill(0.0);
}

inline auto System::elastic() const
{
    return m_elem_elas;
}

inline auto System::plastic() const
{
    return m_elem_plas;
}

inline auto System::conn() const
{
    return m_conn;
}

inline auto System::coor() const
{
    return m_coor;
}

inline auto System::dofs() const
{
    return m_dofs;
}

inline auto System::u() const
{
    return m_u;
}

inline auto System::v() const
{
    return m_v;
}

inline auto System::a() const
{
    return m_a;
}

inline auto& System::mass() const
{
    return m_M;
}

inline auto& System::damping() const
{
    return m_D;
}

inline auto System::fext() const
{
    return m_fext;
}

inline auto System::fint() const
{
    return m_fint;
}

inline auto System::fmaterial() const
{
    return m_fmaterial;
}

inline auto System::fdamp() const
{
    return m_fdamp;
}

inline double System::residual() const
{
    double r_fres = xt::norm_l2(m_fres)();
    double r_fext = xt::norm_l2(m_fext)();
    if (r_fext != 0.0) {
        return r_fres / r_fext;
    }
    return r_fres;
}

inline double System::t() const
{
    return m_t;
}

inline auto System::dV() const
{
    return m_quad.dV();
}

inline const GooseFEM::MatrixPartitioned& System::stiffness() const
{
    return m_K;
}

inline const GooseFEM::VectorPartitioned& System::vector() const
{
    return m_vector;
}

inline const GooseFEM::Element::Quad4::Quadrature& System::quad() const
{
    return m_quad;
}

inline const GMatElastoPlasticQPot::Cartesian2d::Array<2>& System::material() const
{
    return m_material;
}

inline xt::xtensor<double, 4> System::Sig()
{
    return m_Sig;
}

inline xt::xtensor<double, 4> System::Eps()
{
    return m_Eps;
}

inline xt::xtensor<double, 4> System::plastic_Sig() const
{
    return xt::view(m_Sig, xt::keep(m_elem_plas), xt::all(), xt::all(), xt::all());
}

inline xt::xtensor<double, 4> System::plastic_Eps() const
{
    return xt::view(m_Eps, xt::keep(m_elem_plas), xt::all(), xt::all(), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldLeft() const
{
    return xt::view(m_material.CurrentYieldLeft(), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldRight() const
{
    return xt::view(m_material.CurrentYieldRight(), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldLeft(size_t offset) const
{
    return xt::view(m_material.CurrentYieldLeft(offset), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_CurrentYieldRight(size_t offset) const
{
    return xt::view(m_material.CurrentYieldRight(offset), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<size_t, 2> System::plastic_CurrentIndex() const
{
    return xt::view(m_material.CurrentIndex(), xt::keep(m_elem_plas), xt::all());
}

inline xt::xtensor<double, 2> System::plastic_Epsp() const
{
    return xt::view(m_material.Epsp(), xt::keep(m_elem_plas), xt::all());
}

inline void System::computeStress()
{
    FRICTIONQPOTFEM_ASSERT(m_allset);

    m_vector.asElement(m_u, m_ue);
    m_quad.symGradN_vector(m_ue, m_Eps);
    m_material.setStrain(m_Eps);
    m_material.stress(m_Sig);
}

inline void System::computeForceMaterial()
{
    this->computeStress();

    m_quad.int_gradN_dot_tensor2_dV(m_Sig, m_fe);
    m_vector.assembleNode(m_fe, m_fmaterial);
}

inline void System::computeInternalExternalResidualForce()
{
    xt::noalias(m_fint) = m_fmaterial + m_fdamp;
    m_vector.copy_p(m_fint, m_fext);
    xt::noalias(m_fres) = m_fext - m_fint;
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
    this->updated_u();

    // estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
    this->updated_v();

    // compute residual force & solve

    this->computeInternalExternalResidualForce();
    m_M.solve(m_fres, m_a);

    // re-estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    // compute residual force & solve

    this->computeInternalExternalResidualForce();
    m_M.solve(m_fres, m_a);

    // new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
    this->updated_v();

    // compute residual force & solve

    this->computeInternalExternalResidualForce();
    m_M.solve(m_fres, m_a);
}

inline size_t System::timeStepsUntilEvent(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList stop(niter_tol);

    auto idx_n = this->plastic_CurrentIndex();

    for (size_t iiter = 1; iiter < max_iter; ++iiter) {

        this->timeStep();

        auto idx = this->plastic_CurrentIndex();

        if (xt::any(xt::not_equal(idx, idx_n))) {
            return iiter;
        }

        if (stop.stop(this->residual(), tol)) {
            this->quench();
            return 0;
        }
    }

    FRICTIONQPOTFEM_REQUIRE(false);
}

inline size_t System::minimise(double tol, size_t niter_tol, size_t max_iter)
{
    GooseFEM::Iterate::StopList stop(niter_tol);

    for (size_t iiter = 0; iiter < max_iter ; ++iiter) {

        this->timeStep();

        if (stop.stop(this->residual(), tol)) {
            this->quench();
            return iiter;
        }
    }

    FRICTIONQPOTFEM_REQUIRE(false);
}

template <class C, class E, class L>
inline HybridSystem::HybridSystem(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initHybridSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

template <class C, class E, class L>
inline void HybridSystem::initHybridSystem(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);

    m_conn_elas = xt::view(m_conn, xt::keep(m_elem_elas), xt::all());
    m_conn_plas = xt::view(m_conn, xt::keep(m_elem_plas), xt::all());

    m_vector_elas = GooseFEM::VectorPartitioned(m_conn_elas, m_dofs, m_iip);
    m_vector_plas = GooseFEM::VectorPartitioned(m_conn_plas, m_dofs, m_iip);

    m_quad_elas = GooseFEM::Element::Quad4::Quadrature(m_vector_elas.AsElement(m_coor));
    m_quad_plas = GooseFEM::Element::Quad4::Quadrature(m_vector_plas.AsElement(m_coor));

    m_ue_plas = m_vector_plas.allocate_elemvec(0.0);
    m_fe_plas = m_vector_plas.allocate_elemvec(0.0);

    m_felas = m_vector.allocate_nodevec(0.0);
    m_fplas = m_vector.allocate_nodevec(0.0);

    m_Eps_elas = m_quad_elas.allocate_qtensor<2>(0.0);
    m_Sig_elas = m_quad_elas.allocate_qtensor<2>(0.0);
    m_Eps_plas = m_quad_plas.allocate_qtensor<2>(0.0);
    m_Sig_plas = m_quad_plas.allocate_qtensor<2>(0.0);

    m_material_elas = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem_elas, m_nip});
    m_material_plas = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem_plas, m_nip});
}

inline void HybridSystem::setElastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem)
{
    System::setElastic(K_elem, G_elem);

    xt::xtensor<size_t, 2> I = xt::ones<size_t>({m_nelem_elas, m_nip});
    xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem_elas, m_nip});
    xt::view(idx, xt::range(0, m_nelem_elas), xt::all()) = xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});
    m_material_elas.setElastic(I, idx, K_elem, G_elem);
    m_material_elas.setStrain(m_Eps_elas);
    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::not_equal(
        m_material_elas.type(),
        GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));

    m_K_elas = GooseFEM::Matrix(m_conn_elas, m_dofs);
    m_K_elas.assemble(m_quad_elas.Int_gradN_dot_tensor4_dot_gradNT_dV(m_material_elas.Tangent()));
}

inline void HybridSystem::setPlastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem,
    const xt::xtensor<double, 2>& epsy_elem)
{
    System::setPlastic(K_elem, G_elem, epsy_elem);

    xt::xtensor<size_t, 2> I = xt::ones<size_t>({m_nelem_plas, m_nip});
    xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem_plas, m_nip});
    xt::view(idx, xt::range(0, m_nelem_plas), xt::all()) = xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});
    m_material_plas.setCusp(I, idx, K_elem, G_elem, epsy_elem);
    m_material_plas.setStrain(m_Eps_plas);
    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::not_equal(
        m_material_plas.type(),
        GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));
}

inline const GMatElastoPlasticQPot::Cartesian2d::Array<2>& HybridSystem::material_elastic() const
{
    return m_material_elas;
}

inline const GMatElastoPlasticQPot::Cartesian2d::Array<2>& HybridSystem::material_plastic() const
{
    return m_material_plas;
}

inline void HybridSystem::evalSystem()
{
    if (!m_eval_full) {
        return;
    }
    this->computeStress();
    m_eval_full = false;
}

inline xt::xtensor<double, 4> HybridSystem::Sig()
{
    this->evalSystem();
    return m_Sig;
}

inline xt::xtensor<double, 4> HybridSystem::Eps()
{
    this->evalSystem();
    return m_Eps;
}

inline xt::xtensor<double, 4> HybridSystem::plastic_Sig() const
{
    return m_Sig_plas;
}

inline xt::xtensor<double, 4> HybridSystem::plastic_Eps() const
{
    return m_Eps_plas;
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldLeft() const
{
    return m_material_plas.CurrentYieldLeft();
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldRight() const
{
    return m_material_plas.CurrentYieldRight();
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldLeft(size_t offset) const
{
    return m_material_plas.CurrentYieldLeft(offset);
}

inline xt::xtensor<double, 2> HybridSystem::plastic_CurrentYieldRight(size_t offset) const
{
    return m_material_plas.CurrentYieldRight(offset);
}

inline xt::xtensor<size_t, 2> HybridSystem::plastic_CurrentIndex() const
{
    return m_material_plas.CurrentIndex();
}

inline xt::xtensor<double, 2> HybridSystem::plastic_Epsp() const
{
    return m_material_plas.Epsp();
}

inline void HybridSystem::computeForceMaterial()
{
    FRICTIONQPOTFEM_ASSERT(m_allset);
    m_eval_full = true;

    m_vector_plas.asElement(m_u, m_ue_plas);
    m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);
    m_material_plas.setStrain(m_Eps_plas);
    m_material_plas.stress(m_Sig_plas);
    m_quad_plas.int_gradN_dot_tensor2_dV(m_Sig_plas, m_fe_plas);
    m_vector_plas.assembleNode(m_fe_plas, m_fplas);

    m_K_elas.dot(m_u, m_felas);

    xt::noalias(m_fmaterial) = m_felas + m_fplas;
}

} // namespace Generic2d
} // namespace FrictionQPotFEM

#endif
