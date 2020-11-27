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

    ret.push_back(fmt::format("qpot={0:d}.{1:d}.{2:d}",
        QPOT_VERSION_MAJOR,
        QPOT_VERSION_MINOR,
        QPOT_VERSION_PATCH));

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
    FRICTIONQPOTFEM_ASSERT(!m_set_M);
    this->setMatrix(m_M, val_elem);
    m_set_M = true;
    this->evalAllSet();
}

inline void System::setDampingMatrix(const xt::xtensor<double, 1>& val_elem)
{
    FRICTIONQPOTFEM_ASSERT(!m_set_D);
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

inline auto System::vector() const
{
    return m_vector;
}

inline auto System::quad() const
{
    return m_quad;
}

inline auto System::material() const
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

inline xt::xtensor<size_t, 2> System::plastic_CurrentIndex() const
{
    return xt::view(m_material.CurrentIndex(), xt::keep(m_elem_plas), xt::all());
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

    FRICTIONQPOTFEM_REQUIRE(false);
}

inline auto System::plastic_signOfSimpleShearPerturbation(double perturbation)
{
    auto u_0 = this->u();
    auto eps_0 = GM::Epsd(this->plastic_Eps());
    auto u_pert = this->u();

    for (size_t n = 0; n < m_nnode; ++n) {
        u_pert(n, 0) += perturbation * (m_coor(n, 1) - m_coor(0, 1));
    }

    this->setU(u_pert);
    auto eps_pert = GM::Epsd(this->plastic_Eps());
    this->setU(u_0);

    xt::xtensor<double, 2> sign = xt::sign(eps_pert - eps_0);
    return sign;
}

inline double System::addSimpleShearEventDriven(
    double deps_kick, bool kick, double direction, bool dry_run)
{
    FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());
    FRICTIONQPOTFEM_REQUIRE(direction == +1.0 || direction == -1.0);

    auto u_new = this->u();
    auto idx = this->plastic_CurrentIndex();
    auto eps = GM::Epsd(this->plastic_Eps());
    auto Epsd = GM::Deviatoric(this->plastic_Eps());
    auto epsxx = xt::view(Epsd, xt::all(), xt::all(), 0, 0);
    auto epsxy = xt::view(Epsd, xt::all(), xt::all(), 0, 1);

    // distance to yielding: "deps"
    // (event a positive kick can lead to a decreasing equivalent strain)
    xt::xtensor<double, 2> sign = this->plastic_signOfSimpleShearPerturbation(direction * deps_kick);
    xt::xtensor<double, 2> epsy_l = xt::abs(this->plastic_CurrentYieldLeft());
    xt::xtensor<double, 2> epsy_r = xt::abs(this->plastic_CurrentYieldRight());
    xt::xtensor<double, 2> epsy = xt::where(sign > 0, epsy_r, epsy_l);
    xt::xtensor<double, 2> deps = xt::abs(eps - epsy);

    // no kick & current strain sufficiently close the next yield strain: don't move
    if (!kick && xt::amin(deps)() < deps_kick / 2.0) {
        return 0.0;
    }

    // set yield strain close to next yield strain
    xt::xtensor<double, 2> eps_new = epsy + sign * (-deps_kick / 2.0);

    // or, apply a kick instead
    if (kick) {
        eps_new = eps + sign * deps_kick;
    }

    // compute shear strain increments
    // - two possible solutions
    //   (the factor two is needed as "dgamma" is the xy-component of the deformation gradient)
    xt::xtensor<double, 2> dgamma =
        2.0 * (-1.0 * direction * epsxy + xt::sqrt(xt::pow(eps_new, 2.0) - xt::pow(epsxx, 2.0)));
    xt::xtensor<double, 2> dgamma_n =
        2.0 * (-1.0 * direction * epsxy - xt::sqrt(xt::pow(eps_new, 2.0) - xt::pow(epsxx, 2.0)));
    // - discard irrelevant solutions
    dgamma_n = xt::where(dgamma_n <= 0.0, dgamma, dgamma_n);
    // - select lowest
    dgamma = xt::where(dgamma_n < dgamma, dgamma_n, dgamma);
    // - select minimal
    double dux = xt::amin(dgamma)();

    if (dry_run) {
        return direction * dux;
    }

    // add as affine deformation gradient to the system
    for (size_t n = 0; n < m_nnode; ++n) {
        u_new(n, 0) += direction * dux * (m_coor(n, 1) - m_coor(0, 1));
    }
    this->setU(u_new);

    // sanity check
    // ------------

    auto index = xt::unravel_index(xt::argmin(dgamma)(), dgamma.shape());
    size_t e = index[0];
    size_t q = index[1];

    eps = GM::Epsd(this->plastic_Eps());
    auto idx_new = this->plastic_CurrentIndex();

    FRICTIONQPOTFEM_REQUIRE(std::abs(eps(e, q) - eps_new(e, q)) / eps_new(e, q) < 1e-4);
    if (!kick) {
        FRICTIONQPOTFEM_REQUIRE(xt::all(xt::equal(idx, idx_new)));
    }

    return direction * dux;
}

inline double System::addSimpleShearToFixedStress(double target_stress, bool dry_run)
{
    FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());

    auto u_new = this->u();
    auto idx = this->plastic_CurrentIndex();
    auto dV = this->AsTensor<2>(this->dV());
    double G = m_material.G().data()[0];

    xt::xtensor<double, 2> Epsbar = xt::average(this->Eps(), dV, {0, 1});
    xt::xtensor<double, 2> Sigbar = xt::average(this->Sig(), dV, {0, 1});
    xt::xtensor<double, 2> Epsd = GM::Deviatoric(Epsbar);
    double epsxx = Epsd(0, 0);
    double epsxy = Epsd(0, 1);

    FRICTIONQPOTFEM_ASSERT(Sigbar(0, 1) >= 0);

    double eps = GM::Epsd(Epsbar)();
    double sig = GM::Sigd(Sigbar)();
    double direction = +1.0;
    if (target_stress < sig) {
        direction = -1.0;
    }

    double eps_new = eps + (target_stress - sig) / (2.0 * G);
    double dgamma =
        2.0 * (-1.0 * direction * epsxy + std::sqrt(std::pow(eps_new, 2.0) - std::pow(epsxx, 2.0)));

    if (dry_run) {
        return direction * dgamma;
    }

    for (size_t n = 0; n < m_nnode; ++n) {
        u_new(n, 0) += direction * dgamma * (m_coor(n, 1) - m_coor(0, 1));
    }
    this->setU(u_new);

    // sanity check
    // ------------

    Sigbar = xt::average(this->Sig(), dV, {0, 1});
    sig = GM::Sigd(Sigbar)();

    auto idx_new = this->plastic_CurrentIndex();

    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::equal(idx, idx_new)));
    FRICTIONQPOTFEM_REQUIRE(std::abs(target_stress - sig) / sig < 1e-4);

    return direction * dgamma;
}

inline double System::triggerElementWithLocalSimpleShear(
    double deps_kick,
    size_t plastic_element,
    bool trigger_weakest,
    double amplify)
{
    FRICTIONQPOTFEM_ASSERT(plastic_element < m_nelem_plas);

    auto idx = this->plastic_CurrentIndex();
    auto eps = GM::Epsd(this->plastic_Eps());
    auto up = m_vector.AsDofs_p(m_u);

    // distance to yielding on the positive side
    auto epsy = this->plastic_CurrentYieldRight();
    auto deps = epsy - eps;

    // find integration point closest to yielding
    size_t e = m_elem_plas(plastic_element);
    size_t q = xt::argmin(xt::view(deps, plastic_element, xt::all()))();
    if (!trigger_weakest) {
        q = xt::argmax(xt::view(deps, plastic_element, xt::all()))();
    }

    // deviatoric strain at the selected quadrature-point
    xt::xtensor<double, 2> Eps = xt::view(this->plastic_Eps(), plastic_element, q);
    xt::xtensor<double, 2> Epsd = GM::Deviatoric(Eps);

    // new equivalent deviatoric strain: yield strain + small strain kick
    double eps_new = epsy(plastic_element, q) + deps_kick / 2.0;

    // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
    double dgamma = 2.0 * (-Epsd(0, 1) + std::sqrt(std::pow(eps_new, 2.0) - std::pow(Epsd(0, 0), 2.0)));

    // apply increment in shear strain as a perturbation to the selected element
    // - nodes belonging to the current element, from connectivity
    auto elem = xt::view(m_conn, e, xt::all());
    // - displacement-DOFs
    auto udofs = m_vector.AsDofs(m_u);
    // - update displacement-DOFs for the element
    for (size_t n = 0; n < m_nne; ++n) {
        udofs(m_dofs(elem(n), 0)) += dgamma * amplify * (m_coor(elem(n), 1) - m_coor(elem(0), 1));
    }
    // - convert displacement-DOFs to (periodic) nodal displacement vector
    //   (N.B. storing to nodes directly does not ensure periodicity)
    this->setU(m_vector.AsNode(udofs));

    eps = GM::Epsd(this->plastic_Eps());
    auto idx_new = this->plastic_CurrentIndex();
    auto up_new = m_vector.AsDofs_p(m_u);

    FRICTIONQPOTFEM_REQUIRE(dgamma >= 0.0);
    FRICTIONQPOTFEM_REQUIRE(amplify >= 0.0);
    FRICTIONQPOTFEM_REQUIRE(std::abs(eps(plastic_element, q) - eps_new) / eps_new < 1e-4 || amplify != 1);
    FRICTIONQPOTFEM_REQUIRE(xt::any(xt::not_equal(idx, idx_new)));
    FRICTIONQPOTFEM_REQUIRE(idx(plastic_element, q) != idx_new(plastic_element, q));
    FRICTIONQPOTFEM_REQUIRE(xt::allclose(up, up_new));

    return dgamma;
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

inline auto HybridSystem::material_elastic() const
{
    return m_material_elas;
}

inline auto HybridSystem::material_plastic() const
{
    return m_material_plas;
}

inline xt::xtensor<double, 4> HybridSystem::Sig()
{
    if (m_eval_full) {
        this->computeStress();
        m_eval_full = false;
    }

    return m_Sig;
}

inline xt::xtensor<double, 4> HybridSystem::Eps()
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

inline xt::xtensor<double, 2> HybridSystem::plastic_ElementEnergyLandscapeForSimpleShear(
    const xt::xtensor<double, 1>& Delta_gamma,
    bool tilted)
{
    auto dV_plas = xt::view(m_quad.dV(), xt::keep(m_elem_plas), xt::all());
    double dV = dV_plas(0, 0);
    FRICTIONQPOTFEM_ASSERT(xt::allclose(dV_plas, dV));

    auto Eps = m_Eps_plas;
    auto dgamma = xt::diff(Delta_gamma);
    FRICTIONQPOTFEM_ASSERT(xt::all(dgamma >= 0.0));

    xt::xtensor<double, 2> ret = xt::empty<double>({m_nelem_plas, Delta_gamma.size()});
    xt::view(ret, xt::all(), 0) = xt::sum(m_material_plas.Energy() * dV, 1);

    for (size_t i = 0; i < dgamma.size(); ++i) {
        xt::view(Eps, xt::all(), xt::all(), 0, 1) += 0.5 * dgamma(i);
        xt::view(Eps, xt::all(), xt::all(), 1, 0) += 0.5 * dgamma(i);
        m_material_plas.setStrain(Eps);
        xt::view(ret, xt::all(), i + 1) = xt::sum(m_material_plas.Energy() * dV, 1);
    }

    if (tilted) {
        for (size_t e = 0; e < m_nelem_plas; ++e) {
            auto elem = xt::view(m_conn, m_elem_plas(e), xt::all());
            double h = m_coor(elem(3), 1) - m_coor(elem(0), 1);
            double f = m_fe_plas(e, 2, 0)
                     + m_fe_plas(e, 3, 0)
                     - m_fe_plas(e, 0, 0)
                     - m_fe_plas(e, 1, 0);
            xt::view(ret, e, xt::all()) -= (0.5 * h * f * Delta_gamma);
        }
    }

    ret /= dV * static_cast<double>(m_nip);

    m_material_plas.setStrain(m_Eps_plas);

    return ret;
}

// inline xt::xtensor<double, 2> HybridSystem::plastic_ElementEnergyBarrierForSimpleShear(bool tilted)
inline std::tuple<xt::xtensor<double, 2>, xt::xtensor<double, 2>> HybridSystem::plastic_ElementEnergyBarrierForSimpleShear(bool tilted)
{
    auto dV_plas = xt::view(m_quad.dV(), xt::keep(m_elem_plas), xt::all());
    double dV = dV_plas(0, 0);
    FRICTIONQPOTFEM_ASSERT(xt::allclose(dV_plas, dV));

    // TODO: input
    size_t niter = 20;

    static constexpr size_t nip = 4;
    FRICTIONQPOTFEM_ASSERT(m_nip == nip);

    // TODO: assert on h
    auto elem = xt::view(m_conn, m_elem_plas(0), xt::all());
    double h = m_coor(elem(3), 1) - m_coor(elem(0), 1);

    double inf = std::numeric_limits<double>::infinity();
    xt::xtensor<double, 2> ret = inf * xt::ones<double>({m_N, size_t(2)});
    xt::xtensor<double, 2> out_E = inf * xt::ones<double>({m_N, niter + size_t(1)});
    xt::xtensor<double, 2> out_gamma = inf * xt::ones<double>({m_N, niter + size_t(1)});

    std::array<GM::Cusp*, nip> models;
    xt::xtensor<double, 1> trial_dgamma = xt::empty<double>({nip});

    for (size_t e = 0; e < m_N; ++e) {

        double E0 = 0.0;
        double delta_gamma = 0.0;

        for (size_t q = 0; q < m_nip; ++q) {
            models[q] = m_material_plas.refCusp({e, q});
            E0 += models[q]->energy() * dV;
        }

        double E_n = E0;

        out_E(e, 0) = E0 / (dV * static_cast<double>(m_nip));
        out_gamma(e, 0) = 0.0;

        for (size_t i = 0; i < niter; ++i) {

            // for each integration point: compute the increment in strain to reach
            // the next minimum or the next cusp
            for (size_t q = 0; q < m_nip; ++q) {
                auto Eps = models[q]->Strain();
                double epsy_r = models[q]->currentYieldRight();
                double epsp = models[q]->epsp();
                double eps = GM::Epsd(Eps)();
                double eps_new = epsy_r;
                if (eps < epsp) {
                    eps_new = epsp;
                }
                eps_new += 1e-12; // TODO: figure out a way around this
                trial_dgamma(q) =
                    - Eps(0, 1)
                    + std::sqrt(std::pow(eps_new, 2.0) + std::pow(Eps(0, 1), 2.0) - std::pow(eps, 2.0));
            }

            // increment strain for integration points with the same value
            double E = 0.0;
            double dgamma = xt::amin(trial_dgamma)();
            delta_gamma += dgamma;

            for (size_t q = 0; q < m_nip; ++q) {
                auto Eps = models[q]->Strain();
                Eps(0, 1) += dgamma;
                Eps(1, 0) += dgamma;
                models[q]->setStrain(Eps);
                E += models[q]->energy() * dV;
            }

            if (tilted) {
                double f = m_fe_plas(e, 2, 0)
                         + m_fe_plas(e, 3, 0)
                         - m_fe_plas(e, 0, 0)
                         - m_fe_plas(e, 1, 0);
                E -= h * f * delta_gamma;
            }

            // energy barrier found: store the last known configuration, this will be the maximum
            if (E < E_n) {
                ret(e, 0) = delta_gamma - dgamma;
                ret(e, 1) = (E_n - E0) / (dV * static_cast<double>(m_nip));
                // break; // TODO: uncomment
            }

            out_E(e, i + 1) = E / (dV * static_cast<double>(m_nip));
            out_gamma(e, i + 1) = delta_gamma;

            // store 'history'
            E_n = E;
        }
    }

    m_material_plas.setStrain(m_Eps_plas);

    // return ret;
    return std::make_tuple(out_E, out_gamma);
}

// inline std::tuple<xt::xtensor<double, 1>, xt::xtensor<double, 1>>
// HybridSystem::plastic_ElementEnergyLandscapeForSimpleShearEventDriven(
//     size_t plastic_element,
//     bool tilted)
// {
//     double dv = quad.dV()(m_elem_plas(plastic), 0);



// }

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#endif
