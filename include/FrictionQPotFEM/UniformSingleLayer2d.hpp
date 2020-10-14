/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM

*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_HPP
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_HPP

#include "UniformSingleLayer2d.h"

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

inline System::System(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const xt::xtensor<size_t, 1>& elem_elastic,
    const xt::xtensor<size_t, 1>& elem_plastic)
{
    this->initGeometry(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

inline void System::initGeometry(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const xt::xtensor<size_t, 1>& elem_elastic,
    const xt::xtensor<size_t, 1>& elem_plastic)
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
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::in1d(m_iip, m_dofs)));
    #endif

    m_vector = GF::VectorPartitioned(m_conn, m_dofs, m_iip);

    m_quad = QD::Quadrature(m_vector.AsElement(m_coor));
    m_nip = m_quad.nip();

    m_u = m_vector.AllocateNodevec(0.0);
    m_v = m_vector.AllocateNodevec(0.0);
    m_a = m_vector.AllocateNodevec(0.0);
    m_v_n = m_vector.AllocateNodevec(0.0);
    m_a_n = m_vector.AllocateNodevec(0.0);

    m_ue = m_vector.AllocateElemvec(0.0);
    m_fe = m_vector.AllocateElemvec(0.0);

    m_fmaterial = m_vector.AllocateNodevec(0.0);
    m_fdamp = m_vector.AllocateNodevec(0.0);
    m_fint = m_vector.AllocateNodevec(0.0);
    m_fext = m_vector.AllocateNodevec(0.0);
    m_fres = m_vector.AllocateNodevec(0.0);

    m_Eps = m_quad.AllocateQtensor<2>(0.0);
    m_Sig = m_quad.AllocateQtensor<2>(0.0);

    m_M = GF::MatrixDiagonalPartitioned(m_conn, m_dofs, m_iip);
    m_D = GF::MatrixDiagonal(m_conn, m_dofs);

    m_material = GM::Array<2>({m_nelem, m_nip});
}

inline void System::evalAllSet()
{
    m_allset = m_set_M && m_set_D && m_set_elas && m_set_plas && m_dt > 0.0;
}

inline void System::initMaterial()
{
    if (!(m_set_elas && m_set_plas)) {
        return;
    }

    m_material.check();

    m_material.setStrain(m_Eps);
}

template <class T>
inline void System::setMatrix(T& matrix, const xt::xtensor<double, 1>& val_elem)
{
    FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);

    QD::Quadrature nodalQuad(m_vector.AsElement(m_coor), QD::Nodal::xi(), QD::Nodal::w());

    xt::xtensor<double, 2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});
    for (size_t q = 0; q < nodalQuad.nip(); ++q) {
        xt::view(val_quad, xt::all(), q) = val_elem;
    }

    matrix.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
}

inline void System::setMassMatrix(const xt::xtensor<double, 1>& val_elem)
{
    this->setMatrix(m_M, val_elem);
    m_set_M = true;
    this->evalAllSet();
}

inline void System::setDampingMatrix(const xt::xtensor<double, 1>& val_elem)
{
    this->setMatrix(m_D, val_elem);
    m_set_D = true;
    this->evalAllSet();
}

inline void System::setElastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem)
{
    xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
    xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
    xt::view(I, xt::keep(m_elem_elas), xt::all()) = 1ul;
    xt::view(idx, xt::keep(m_elem_elas), xt::all()) = xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});
    m_material.setElastic(I, idx, K_elem, G_elem);

    m_set_elas = true;
    this->evalAllSet();
    this->initMaterial();
}

inline void System::setPlastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem,
    const xt::xtensor<double, 2>& epsy_elem)
{
    xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
    xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
    xt::view(I, xt::keep(m_elem_plas), xt::all()) = 1ul;
    xt::view(idx, xt::keep(m_elem_plas), xt::all()) = xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});
    m_material.setCusp(I, idx, K_elem, G_elem, epsy_elem);

    m_set_plas = true;
    this->evalAllSet();
    this->initMaterial();
}

inline void System::setDt(double dt)
{
    m_dt = dt;
    this->evalAllSet();
}

inline void System::setU(const xt::xtensor<double, 2>& u)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, {m_nnode, m_ndim}));
    m_u = u;
    this->computeStress();
}

inline void System::quench()
{
    m_v.fill(0.0);
    m_a.fill(0.0);
}

inline auto System::nelem() const
{
    return m_nelem;
}

inline auto System::forceMaterial() const
{
    return m_fmaterial;
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

inline auto System::u() const
{
    return m_u;
}

template <size_t rank, class T>
inline auto System::AsTensor(const T& arg) const
{
    return m_quad.AsTensor<rank>(arg);
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
    computeStress();

    m_quad.int_gradN_dot_tensor2_dV(m_Sig, m_fe);
    m_vector.assembleNode(m_fe, m_fmaterial);
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

    computeForceMaterial();

    // estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + m_dt * m_a_n;

    m_D.dot(m_v, m_fdamp);

    // compute residual force & solve

    xt::noalias(m_fint) = m_fmaterial + m_fdamp;

    m_vector.copy_p(m_fint, m_fext);

    xt::noalias(m_fres) = m_fext - m_fint;

    m_M.solve(m_fres, m_a);

    // re-estimate new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);

    m_D.dot(m_v, m_fdamp);

    // compute residual force & solve

    xt::noalias(m_fint) = m_fmaterial + m_fdamp;

    m_vector.copy_p(m_fint, m_fext);

    xt::noalias(m_fres) = m_fext - m_fint;

    m_M.solve(m_fres, m_a);

    // new velocity, update corresponding force

    xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);

    m_D.dot(m_v, m_fdamp);

    // compute residual force & solve

    xt::noalias(m_fint) = m_fmaterial + m_fdamp;

    m_vector.copy_p(m_fint, m_fext);

    xt::noalias(m_fres) = m_fext - m_fint;

    m_M.solve(m_fres, m_a);
}

inline HybridSystem::HybridSystem(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const xt::xtensor<size_t, 1>& elem_elastic,
    const xt::xtensor<size_t, 1>& elem_plastic)
{
    this->initGeometry(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

inline void HybridSystem::computeStressPlastic()
{
    FRICTIONQPOTFEM_ASSERT(m_allset);

    m_vector_plas.asElement(m_u, m_ue_plas);
    m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);
    m_material_plas.setStrain(m_Eps_plas);
    m_material_plas.stress(m_Sig_plas);
}

inline void HybridSystem::computeForceMaterial()
{
    computeStressPlastic();

    m_quad_plas.int_gradN_dot_tensor2_dV(m_Sig_plas, m_fe_plas);
    m_vector_plas.assembleNode(m_fe_plas, m_fplas);
    m_K_elas.dot(m_u, m_felas);
}


} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#endif
