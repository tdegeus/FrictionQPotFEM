/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM

*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_HPP
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_HPP

#include "UniformSingleLayer2d.h"

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

inline std::vector<std::string> versionInfo()
{
    std::vector<std::string> ret;

    ret.push_back(fmt::format("xtensor={0:d}.{1:d}.{2:d}",
        XTENSOR_VERSION_MAJOR,
        XTENSOR_VERSION_MINOR,
        XTENSOR_VERSION_PATCH));

    ret.push_back(fmt::format("frictionqpotfem={0:d}.{1:d}.{2:d}",
        FRICTIONQPOTFEM_VERSION_MAJOR,
        FRICTIONQPOTFEM_VERSION_MINOR,
        FRICTIONQPOTFEM_VERSION_PATCH));

    ret.push_back(fmt::format("goosefem={0:d}.{1:d}.{2:d}",
        GOOSEFEM_VERSION_MAJOR,
        GOOSEFEM_VERSION_MINOR,
        GOOSEFEM_VERSION_PATCH));

    ret.push_back(fmt::format("gmatelastoplasticqpot={0:d}.{1:d}.{2:d}",
        GMATELASTOPLASTICQPOT_VERSION_MAJOR,
        GMATELASTOPLASTICQPOT_VERSION_MINOR,
        GMATELASTOPLASTICQPOT_VERSION_PATCH));

    return ret;
}

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
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::isin(m_iip, m_dofs)));
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

inline void System::initMaterial()
{
    if (!(m_set_elas && m_set_plas)) {
        return;
    }

    m_material.check();

    m_material.setStrain(m_Eps);
}

inline void System::setElastic(
    const xt::xtensor<double, 1>& K_elem,
    const xt::xtensor<double, 1>& G_elem)
{
    FRICTIONQPOTFEM_ASSERT(K_elem.size() == m_nelem_elas);
    FRICTIONQPOTFEM_ASSERT(G_elem.size() == m_nelem_elas);

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
    FRICTIONQPOTFEM_ASSERT(K_elem.size() == m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(G_elem.size() == m_nelem_plas);
    FRICTIONQPOTFEM_ASSERT(epsy_elem.shape(0) == m_nelem_plas);

    xt::xtensor<size_t, 2> I = xt::zeros<size_t>({m_nelem, m_nip});
    xt::xtensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem, m_nip});
    xt::view(I, xt::keep(m_elem_plas), xt::all()) = 1ul;
    xt::view(idx, xt::keep(m_elem_plas), xt::all()) = xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});
    m_material.setCusp(I, idx, K_elem, G_elem, epsy_elem);

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

inline void System::setDt(double dt)
{
    m_dt = dt;
    this->evalAllSet();
}

inline void System::setU(const xt::xtensor<double, 2>& u)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, {m_nnode, m_ndim}));
    xt::noalias(m_u) = u;
    this->computeForceMaterial();
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

inline auto System::fmaterial() const
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

template <size_t rank, class T>
inline auto System::AsTensor(const T& arg) const
{
    return m_quad.AsTensor<rank>(arg);
}

template <class T>
inline auto System::AsDofs(const T& arg) const
{
    return m_vector.AsDofs(arg);
}

template <class T>
inline auto System::AsNode(const T& arg) const
{
    return m_vector.AsNode(arg);
}

inline auto System::vector() const
{
    return m_vector;
}

inline auto System::quad() const
{
    return m_quad;
}

inline auto System::Sig() const
{
    return m_Sig;
}

inline auto System::Eps() const
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

inline xt::xtensor<size_t, 2> System::plastic_CurrentIndex() const
{
    return xt::view(m_material.CurrentIndex(), xt::keep(m_elem_plas), xt::all());
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

    throw std::runtime_error("Maximum number of iterations exceeded");
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

inline HybridSystem::HybridSystem(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const xt::xtensor<size_t, 1>& elem_elastic,
    const xt::xtensor<size_t, 1>& elem_plastic)
{
    this->initGeometry(coor, conn, dofs, iip, elem_elastic, elem_plastic);
    this->initHybridSystem();
}

inline void HybridSystem::initHybridSystem()
{
    m_conn_elas = xt::view(m_conn, xt::keep(m_elem_elas), xt::all());
    m_conn_plas = xt::view(m_conn, xt::keep(m_elem_plas), xt::all());

    m_vector_elas = GF::VectorPartitioned(m_conn_elas, m_dofs, m_iip);
    m_vector_plas = GF::VectorPartitioned(m_conn_plas, m_dofs, m_iip);

    m_quad_elas = QD::Quadrature(m_vector_elas.AsElement(m_coor));
    m_quad_plas = QD::Quadrature(m_vector_plas.AsElement(m_coor));

    m_ue_plas = m_vector_plas.AllocateElemvec(0.0);
    m_fe_plas = m_vector_plas.AllocateElemvec(0.0);

    m_felas = m_vector.AllocateNodevec(0.0);
    m_fplas = m_vector.AllocateNodevec(0.0);

    m_Eps_elas = m_quad_elas.AllocateQtensor<2>(0.0);
    m_Sig_elas = m_quad_elas.AllocateQtensor<2>(0.0);
    m_Eps_plas = m_quad_plas.AllocateQtensor<2>(0.0);
    m_Sig_plas = m_quad_plas.AllocateQtensor<2>(0.0);

    m_material_elas = GM::Array<2>({m_nelem_elas, m_nip});
    m_material_plas = GM::Array<2>({m_nelem_plas, m_nip});
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

    m_material_elas.check();
    m_material_elas.setStrain(m_Eps_elas);

    m_K_elas = GF::Matrix(m_conn_elas, m_dofs);
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

    m_material_plas.check();
    m_material_plas.setStrain(m_Eps_plas);
}

inline auto HybridSystem::Sig()
{
    if (m_eval_full) {
        this->computeStress();
        m_eval_full = false;
    }

    return m_Sig;
}

inline auto HybridSystem::Eps()
{
    if (m_eval_full) {
        this->computeStress();
        m_eval_full = false;
    }

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

inline xt::xtensor<size_t, 2> HybridSystem::plastic_CurrentIndex() const
{
    return m_material_plas.CurrentIndex();
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

inline void addEventDrivenShear(System& sys, double deps_kick, bool kick)
{
    FRICTIONQPOTFEM_ASSERT(sys.isHomogeneousElastic());

    auto coor = sys.coor();
    auto idx = sys.plastic_CurrentIndex();
    auto eps = GM::Epsd(sys.plastic_Eps());
    auto Epsd = GM::Deviatoric(sys.plastic_Eps());
    auto epsxx = xt::view(Epsd, xt::all(), xt::all(), 0, 0);
    auto epsxy = xt::view(Epsd, xt::all(), xt::all(), 0, 1);

    // displacement perturbation to determine the sign in equivalent strain space
    // --------------------------------------------------------------------------

    auto u_new = sys.u();
    auto u_pert = sys.u();

    for (size_t n = 0; n < coor.shape(0); ++n) {
        u_pert(n, 0) += deps_kick * (coor(n, 1) - coor(0, 1));
    }

    sys.setU(u_pert);
    auto eps_pert = GM::Epsd(sys.plastic_Eps());
    xt::xtensor<double, 2> sign = xt::sign(eps_pert - eps);
    sys.setU(u_new);

    // determine strain increment
    // --------------------------

    // distance to yielding
    xt::xtensor<double, 2> epsy_l = xt::abs(sys.plastic_CurrentYieldLeft());
    xt::xtensor<double, 2> epsy_r = xt::abs(sys.plastic_CurrentYieldRight());
    xt::xtensor<double, 2> epsy = xt::where(sign > 0, epsy_r, epsy_l);
    xt::xtensor<double, 2> deps = xt::abs(eps - epsy);

    // no kick & current strain sufficiently close the next yield strain: don't move
    if (!kick && xt::amin(deps)() < deps_kick / 2.0) {
        return;
    }

    // set yield strain close to next yield strain
    xt::xtensor<double, 2> eps_new = epsy + sign * (-deps_kick / 2.0);

    // or, apply a kick instead
    if (kick) {
        eps_new = eps + sign * deps_kick;
    }

    // compute shear strain increments
    // - two possible solutions
    xt::xtensor<double, 2> dgamma = 2.0 * (-epsxy + xt::sqrt(xt::pow(eps_new, 2.0) - xt::pow(epsxx, 2.0)));
    xt::xtensor<double, 2> dgamma_n = 2.0 * (-epsxy - xt::sqrt(xt::pow(eps_new, 2.0) - xt::pow(epsxx, 2.0)));
    // - discard irrelevant solutions
    dgamma_n = xt::where(dgamma_n <= 0.0, dgamma, dgamma_n);
    // - select lowest
    dgamma = xt::where(dgamma_n < dgamma, dgamma_n, dgamma);
    // - select minimal
    double dux = xt::amin(dgamma)();

    // add as affine deformation gradient to the system
    for (size_t n = 0; n < coor.shape(0); ++n) {
        u_new(n, 0) += dux * (coor(n, 1) - coor(0, 1));
    }
    sys.setU(u_new);

    // sanity check
    // ------------

    auto index = xt::unravel_index(xt::argmin(dgamma)(), dgamma.shape());
    size_t e = index[0];
    size_t q = index[1];

    eps = GM::Epsd(sys.plastic_Eps());

    auto idx_new = sys.plastic_CurrentIndex();

    if (std::abs(eps(e, q) - eps_new(e, q)) / eps_new(e, q) > 1e-4) {
        throw std::runtime_error("Strain not what it was supposed to be");
    }

    if (!kick && xt::any(xt::not_equal(idx, idx_new))) {
        throw std::runtime_error("Yielding took place where it shouldn't");
    }
}

inline auto localTriggerElement(System& sys, double deps_kick, size_t plastic_element)
{
    auto coor = sys.coor();
    auto dofs = sys.dofs();
    auto conn = sys.conn();
    auto plastic = sys.plastic();
    auto idx = sys.plastic_CurrentIndex();
    auto eps = GM::Epsd(sys.plastic_Eps());
    auto up = sys.vector().AsDofs_p(sys.u());

    FRICTIONQPOTFEM_ASSERT(plastic_element < plastic.size());

    // distance to yielding on the positive side
    auto epsy = sys.plastic_CurrentYieldRight();
    auto deps = epsy - eps;

    // find integration point closest to yielding
    auto e = plastic(plastic_element);
    auto q = xt::argmin(xt::view(deps, plastic_element, xt::all()))();

    // deviatoric strain at the selected quadrature-point
    xt::xtensor<double, 2> Eps = xt::view(sys.plastic_Eps(), plastic_element, q);
    xt::xtensor<double, 2> Epsd = GM::Deviatoric(Eps);

    // new equivalent deviatoric strain: yield strain + small strain kick
    double eps_new = epsy(plastic_element, q) + deps_kick / 2.0;

    // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
    double dgamma = 2.0 * (-Epsd(0, 1) + std::sqrt(std::pow(eps_new, 2.0) - std::pow(Epsd(0, 0), 2.0)));

    // apply increment in shear strain as a perturbation to the selected element
    // - nodes belonging to the current element, from connectivity
    auto elem = xt::view(conn, e, xt::all());
    // - displacement-DOFs
    auto udofs = sys.AsDofs(sys.u());
    // - update displacement-DOFs for the element
    for (size_t n = 0; n < conn.shape(1); ++n) {
        udofs(dofs(elem(n), 0)) += dgamma * (coor(elem(n), 1) - coor(elem(0), 1));
    }
    // - convert displacement-DOFs to (periodic) nodal displacement vector
    //   (N.B. storing to nodes directly does not ensure periodicity)
    sys.setU(sys.AsNode(udofs));

    eps = GM::Epsd(sys.plastic_Eps());
    auto idx_new = sys.plastic_CurrentIndex();
    auto up_new = sys.vector().AsDofs_p(sys.u());

    if (std::abs(eps(plastic_element, q) - eps_new) / eps_new > 1e-4) {
        throw std::runtime_error("Strain not what it was supposed to be");
    }

    if (!xt::any(xt::not_equal(idx, idx_new))) {
        throw std::runtime_error("Yielding didn't took place while it should have");
    }

    if (idx(plastic_element, q) == idx_new(plastic_element, q)) {
        throw std::runtime_error("Yielding didn't took place while it should have");
    }

    if (!xt::allclose(up, up_new)) {
        throw std::runtime_error("Fixed boundaries where moved");
    }

    return xt::xtensor<size_t, 1>{plastic_element, q};
}

inline auto localTriggerWeakestElement(System& sys, double deps_kick)
{
    auto eps = GM::Epsd(sys.plastic_Eps());
    auto epsy = sys.plastic_CurrentYieldRight();
    auto deps = epsy - eps;
    auto index = xt::unravel_index(xt::argmin(deps)(), deps.shape());
    return localTriggerElement(sys, deps_kick, index[0]);
}

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#endif
