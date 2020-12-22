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

    ret.push_back(fmt::format("gmattensor={0:d}.{1:d}.{2:d}",
        GMATTENSOR_VERSION_MAJOR,
        GMATTENSOR_VERSION_MINOR,
        GMATTENSOR_VERSION_PATCH));

    ret.push_back(fmt::format("qpot={0:d}.{1:d}.{2:d}",
        QPOT_VERSION_MAJOR,
        QPOT_VERSION_MINOR,
        QPOT_VERSION_PATCH));

    ret.push_back(fmt::format("xtensor={0:d}.{1:d}.{2:d}",
        XTENSOR_VERSION_MAJOR,
        XTENSOR_VERSION_MINOR,
        XTENSOR_VERSION_PATCH));

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
    this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

inline void System::initSystem(
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
    m_K = GF::MatrixPartitioned(m_conn, m_dofs, m_iip);

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

    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::not_equal(m_material.type(), GM::Type::Unset)));
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

inline double System::plastic_h() const
{
    auto bot = xt::view(m_conn, xt::keep(m_elem_plas), 0);
    auto top = xt::view(m_conn, xt::keep(m_elem_plas), 3);
    auto h_plas = xt::view(m_coor, xt::keep(top), 1) - xt::view(m_coor, xt::keep(bot), 1);
    double h = h_plas(0);
    FRICTIONQPOTFEM_ASSERT(xt::allclose(h_plas, h));
    return h;
}

inline double System::plastic_dV() const
{
    auto dV_plas = xt::view(m_quad.dV(), xt::keep(m_elem_plas), xt::all());
    double dV = dV_plas(0, 0);
    FRICTIONQPOTFEM_ASSERT(xt::allclose(dV_plas, dV));
    return dV;
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

inline auto System::stiffness() const
{
    return m_K;
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

inline auto System::plastic_signOfPerturbation(const xt::xtensor<double, 2>& delta_u)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(delta_u, {m_nnode, m_ndim}));

    auto u_0 = this->u();
    auto eps_0 = GM::Epsd(this->plastic_Eps());
    auto u_pert = this->u() + delta_u;
    this->setU(u_pert);
    auto eps_pert = GM::Epsd(this->plastic_Eps());
    this->setU(u_0);

    xt::xtensor<int, 2> sign = xt::sign(eps_pert - eps_0);
    return sign;
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

inline double System::addAffineSimpleShear(double delta_gamma)
{
    auto u_new = this->u();

    for (size_t n = 0; n < m_nnode; ++n) {
        u_new(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - m_coor(0, 1));
    }
    this->setU(u_new);

    return delta_gamma * 2.0;
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

inline xt::xtensor<size_t, 2> myargsort(xt::xtensor<double, 2>& A, size_t axis)
{
    if (A.shape(0) == 1) {
        xt::xtensor<double, 1> a = xt::view(A, 0, xt::all());
        xt::xtensor<size_t, 2> ret = xt::empty<size_t>(A.shape());
        xt::view(ret, 0, xt::all()) = xt::argsort(a, axis);
        return ret;
    }
    return xt::argsort(A, axis);
}

inline xt::xtensor<double, 2>
System::plastic_ElementYieldBarrierForSimpleShear(double deps_kick, size_t iquad)
{
    FRICTIONQPOTFEM_ASSERT(iquad < m_nip);

    auto eps = GM::Epsd(this->plastic_Eps());
    auto epsy = this->plastic_CurrentYieldRight();
    auto deps = xt::eval(epsy - eps);
    xt::xtensor<double, 2> ret = xt::empty<double>({m_N, size_t(2)});
    auto isort = myargsort(deps, 1);

    auto Eps = this->plastic_Eps();
    auto Epsd = GM::Deviatoric(Eps);

    for (size_t e = 0; e < m_N; ++e) {
        size_t q = isort(e, iquad);
        double eps_new = epsy(e, q) + deps_kick / 2.0;
        double gamma = Epsd(e, q, 0, 1);
        double epsd_xx = Epsd(e, q, 0, 0);
        ret(e, 0) = deps(e, q);
        ret(e, 1) = - gamma + std::sqrt(std::pow(eps_new, 2.0) - std::pow(epsd_xx, 2.0));
    }

    return ret;
}

inline HybridSystem::HybridSystem(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const xt::xtensor<size_t, 1>& elem_elastic,
    const xt::xtensor<size_t, 1>& elem_plastic)
{
    this->initHybridSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

inline void HybridSystem::initHybridSystem(
    const xt::xtensor<double, 2>& coor,
    const xt::xtensor<size_t, 2>& conn,
    const xt::xtensor<size_t, 2>& dofs,
    const xt::xtensor<size_t, 1>& iip,
    const xt::xtensor<size_t, 1>& elem_elastic,
    const xt::xtensor<size_t, 1>& elem_plastic)
{
    this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);

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
    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::not_equal(m_material_elas.type(), GM::Type::Unset)));
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
    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::not_equal(m_material_plas.type(), GM::Type::Unset)));
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

inline LocalTriggerFineLayerFull::LocalTriggerFineLayerFull(const System& sys)
{
    // Copy / allocate local variables

    auto m = sys.material();
    GM::Array<2> material(m.shape());
    material.setElastic(m.K(), m.G());
    FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(material.type(), GM::Type::Elastic)));

    auto vector = sys.vector();
    auto K = sys.stiffness();
    auto quad = sys.quad();
    GF::MatrixPartitionedSolver<> solver;

    m_elem_plas = sys.plastic();
    m_nelem_plas = m_elem_plas.size();
    m_nip = quad.nip();

    m_smin = xt::zeros<double>({m_nelem_plas, m_nip});
    m_pmin = xt::zeros<double>({m_nelem_plas, m_nip});
    m_Wmin = xt::zeros<double>({m_nelem_plas, m_nip});
    m_dgamma = xt::zeros<double>({m_nelem_plas, m_nip});
    m_dE = xt::zeros<double>({m_nelem_plas, m_nip});

    auto coor = sys.coor();
    auto conn = sys.conn();

    auto u = vector.AllocateNodevec(0.0);
    auto fint = vector.AllocateNodevec(0.0);
    auto fext = vector.AllocateNodevec(0.0);
    auto fres = vector.AllocateNodevec(0.0);

    auto ue = vector.AllocateElemvec(0.0);
    auto fe = vector.AllocateElemvec(0.0);

    auto Eps = quad.AllocateQtensor<2>(0.0);
    auto Sig = quad.AllocateQtensor<2>(0.0);
    std::copy(Eps.shape().cbegin(), Eps.shape().cend(), m_shape_T2.begin());

    m_dV = quad.dV();
    m_V = xt::sum(xt::view(m_dV, m_elem_plas(0)))();

    // Replicate mesh

    GF::Mesh::Quad4::FineLayer mesh(sys.coor(), sys.conn());

    auto elmap = mesh.roll(1);
    size_t nconfig = m_elem_plas(m_nelem_plas - 1) - elmap(m_elem_plas(m_nelem_plas - 1));
    size_t nroll = m_nelem_plas / nconfig;

    m_u_s.resize(nconfig);
    m_u_p.resize(nconfig);
    m_Eps_s.resize(nconfig);
    m_Eps_p.resize(nconfig);
    m_Sig_s.resize(nconfig);
    m_Sig_p.resize(nconfig);
    m_elemmap.resize(nroll);
    m_nodemap.resize(nroll);

    for (size_t roll = 0; roll < nroll; ++roll) {
        m_elemmap[roll] = mesh.roll(roll);
        m_nodemap[roll] = GF::Mesh::elemmap2nodemap(m_elemmap[roll], coor, conn);
    }

    for (size_t e = 0; e < nconfig; ++e) {

        // Simple shear perturbation

        xt::xtensor<double, 2> simple_shear = {
            {0.0, 1.0},
            {1.0, 0.0}};

        this->computePerturbation(e, simple_shear, u, Eps, Sig, K, solver, quad, vector, material);

        for (size_t q = 0; q < m_nip; ++q) {
            auto Epsd = GM::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            for (size_t roll = 0; roll < nroll; ++roll) {
                auto map = mesh.roll(roll);
                m_dgamma(e + map(m_elem_plas(e)) - m_elem_plas(e), q) = gamma;
            }
        }

        m_u_s[e] = u;
        m_Eps_s[e] = Eps;
        m_Sig_s[e] = Sig;

        // Pure shear perturbation

        xt::xtensor<double, 2> pure_shear = {
            {1.0, 0.0},
            {0.0, -1.0}};

        this->computePerturbation(e, pure_shear, u, Eps, Sig, K, solver, quad, vector, material);

        for (size_t q = 0; q < m_nip; ++q) {
            auto Epsd = GM::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double E = Epsd(0, 0);
            for (size_t roll = 0; roll < nroll; ++roll) {
                auto map = mesh.roll(roll);
                m_dE(e + map(m_elem_plas(e)) - m_elem_plas(e), q) = E;
            }
        }

        m_u_p[e] = u;
        m_Eps_p[e] = Eps;
        m_Sig_p[e] = Sig;
    }
}

inline void LocalTriggerFineLayerFull::computePerturbation(
    size_t trigger_plastic,
    const xt::xtensor<double, 2>& sig_star,
    xt::xtensor<double, 2>& u,
    xt::xtensor<double, 4>& Eps,
    xt::xtensor<double, 4>& Sig,
    GF::MatrixPartitioned& K,
    GF::MatrixPartitionedSolver<>& solver,
    const QD::Quadrature& quad,
    const GF::VectorPartitioned& vector,
    GM::Array<2>& material)
{
    size_t trigger_element = m_elem_plas(trigger_plastic);

    Sig.fill(0.0);
    u.fill(0.0);

    for (size_t q = 0; q < quad.nip(); ++q) {
        xt::view(Sig, trigger_element, q) = - sig_star;
    }

    auto fe = quad.Int_gradN_dot_tensor2_dV(Sig);
    auto fint = vector.AssembleNode(fe);
    auto fext = xt::zeros_like(fint);
    vector.copy_p(fint, fext);
    auto fres = xt::eval(fext - fint);

    solver.solve(K, fres, u);

    vector.asElement(u, fe);
    quad.symGradN_vector(fe, Eps);
    material.setStrain(Eps);
    material.stress(Sig);
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::u_s(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    size_t nconfig = m_u_s.size();
    size_t config = trigger_plastic % nconfig;
    size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
    return xt::view(m_u_s[config], xt::keep(m_nodemap[roll]));
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::u_p(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    size_t nconfig = m_u_p.size();
    size_t config = trigger_plastic % nconfig;
    size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
    return xt::view(m_u_p[config], xt::keep(m_nodemap[roll]));
}

inline xt::xtensor<double, 4> LocalTriggerFineLayerFull::Eps_s(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    size_t nconfig = m_u_s.size();
    size_t config = trigger_plastic % nconfig;
    size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
    return xt::view(m_Eps_s[config], xt::keep(m_elemmap[roll]));
}

inline xt::xtensor<double, 4> LocalTriggerFineLayerFull::Eps_p(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    size_t nconfig = m_u_p.size();
    size_t config = trigger_plastic % nconfig;
    size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
    return xt::view(m_Eps_p[config], xt::keep(m_elemmap[roll]));
}

inline xt::xtensor<double, 4> LocalTriggerFineLayerFull::Sig_s(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    size_t nconfig = m_u_s.size();
    size_t config = trigger_plastic % nconfig;
    size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
    return xt::view(m_Sig_s[config], xt::keep(m_elemmap[roll]));
}

inline xt::xtensor<double, 4> LocalTriggerFineLayerFull::Sig_p(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    size_t nconfig = m_u_p.size();
    size_t config = trigger_plastic % nconfig;
    size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
    return xt::view(m_Sig_p[config], xt::keep(m_elemmap[roll]));
}

inline xt::xtensor<double, 2>
LocalTriggerFineLayerFull::slice(const xt::xtensor<double, 2>& arg, size_t) const
{
    return arg;
}

inline xt::xtensor<double, 4>
LocalTriggerFineLayerFull::slice(const xt::xtensor<double, 4>& arg, size_t) const
{
    return arg;
}

inline void LocalTriggerFineLayerFull::setState(
    const xt::xtensor<double, 4>& Eps,
    const xt::xtensor<double, 4>& Sig,
    const xt::xtensor<double, 2>& epsy,
    size_t N)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(Eps, m_shape_T2));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(Sig, m_shape_T2));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(epsy, {m_nelem_plas, m_nip}));

    xt::xtensor<double, 2> S = xt::empty<double>({size_t(2), N});
    xt::xtensor<double, 2> P = xt::empty<double>({size_t(2), N});
    xt::xtensor<double, 2> W = xt::empty<double>({size_t(2), N});

    for (size_t e = 0; e < m_nelem_plas; ++e) {

        auto Eps_s = this->Eps_s(e);
        auto Eps_p = this->Eps_p(e);
        auto Sig_s = this->Sig_s(e);
        auto Sig_p = this->Sig_p(e);
        auto Sig_slice = this->slice(Sig, e);
        auto dV_slice = this->slice(m_dV, e);

        for (size_t q = 0; q < m_nip; ++q) {

            double dgamma = m_dgamma(e, q);
            double dE = m_dE(e, q);

            auto Epsd = GM::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            double E = Epsd(0, 0);
            double y = epsy(e, q);
            double a, b, c, D;

            // solve for "p = 0"
            a = SQR(dgamma);
            b = 2.0 * gamma * dgamma;
            c = SQR(gamma) + SQR(E) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double smax = (- b + std::sqrt(D)) / (2.0 * a);
            double smin = (- b - std::sqrt(D)) / (2.0 * a);

            // solve for "s = 0"
            a = SQR(dE);
            b = 2.0 * E * dE;
            c = SQR(E) + SQR(gamma) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double pmax = (- b + std::sqrt(D)) / (2.0 * a);
            double pmin = (- b - std::sqrt(D)) / (2.0 * a);

            size_t n = static_cast<size_t>(- smin / (smax - smin) * static_cast<double>(N));
            size_t m = N - n;

            for (size_t i = 0; i < 2; ++i) {
                xt::view(S, i, xt::range(0, n)) = xt::linspace<double>(smin, 0, n);
                xt::view(S, i, xt::range(n, N)) = xt::linspace<double>(smax / double(m), smax, m);
                P(i, 0) = 0.0;
                P(i, N - 1) = 0.0;
            }
            P(0, n - 1) = pmax;
            P(1, n - 1) = pmin;

            for (size_t j = 1; j < N - 1; ++j) {
                if (j == n - 1) {
                    continue;
                }
                a = SQR(dE);
                b = 2.0 * E * dE;
                c = SQR(E) + std::pow(gamma + S(0, j) * dgamma, 2.0) - SQR(y);
                D = SQR(b) - 4.0 * a * c;
                P(0, j) = (- b + std::sqrt(D)) / (2.0 * a);
                P(1, j) = (- b - std::sqrt(D)) / (2.0 * a);
            }

            for (size_t i = 0; i < P.shape(0); ++i) {
                for (size_t j = 0; j < P.shape(1); ++j) {
                    xt::xtensor<double, 4> sig = P(i, j) * Sig_p + S(i, j) * Sig_s + Sig_slice;
                    xt::xtensor<double, 4> deps = P(i, j) * Eps_p + S(i, j) * Eps_s;
                    xt::xtensor<double, 2> w = GT::A2s_ddot_B2s(sig, deps);
                    w *= dV_slice;
                    W(i, j) = xt::sum(w)();
                }
            }

            auto idx = xt::argmin(W)();
            m_smin(e, q) = S[idx];
            m_pmin(e, q) = P[idx];
            m_Wmin(e, q) = W[idx];
        }
    }
}

inline void LocalTriggerFineLayerFull::setStateMinimalSearch(
    const xt::xtensor<double, 4>& Eps,
    const xt::xtensor<double, 4>& Sig,
    const xt::xtensor<double, 2>& epsy)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(Eps, m_shape_T2));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(Sig, m_shape_T2));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(epsy, {m_nelem_plas, m_nip}));

    std::array<double, 8> S;
    std::array<double, 8> P;
    std::array<double, 8> W;

    for (size_t e = 0; e < m_nelem_plas; ++e) {

        auto Eps_s = this->Eps_s(e);
        auto Eps_p = this->Eps_p(e);
        auto Sig_s = this->Sig_s(e);
        auto Sig_p = this->Sig_p(e);
        auto Sig_slice = this->slice(Sig, e);
        auto dV_slice = this->slice(m_dV, e);

        for (size_t q = 0; q < m_nip; ++q) {

            double dgamma = m_dgamma(e, q);
            double dE = m_dE(e, q);

            auto Epsd = GM::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            double E = Epsd(0, 0);
            double y = epsy(e, q);
            double a, b, c, D;

            // solve for "p = 0"
            a = SQR(dgamma);
            b = 2.0 * gamma * dgamma;
            c = SQR(gamma) + SQR(E) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double smax = (- b + std::sqrt(D)) / (2.0 * a);
            double smin = (- b - std::sqrt(D)) / (2.0 * a);
            P[0] = 0.0;
            P[1] = 0.0;
            S[0] = smin;
            S[1] = smax;

            // solve for "s = 0"
            a = SQR(dE);
            b = 2.0 * E * dE;
            c = SQR(E) + SQR(gamma) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double pmax = (- b + std::sqrt(D)) / (2.0 * a);
            double pmin = (- b - std::sqrt(D)) / (2.0 * a);
            P[2] = pmin;
            P[3] = pmax;
            S[2] = 0.0;
            S[3] = 0.0;
            S[4] = smin / 2.0;
            S[5] = smin / 2.0;
            S[6] = smax / 2.0;
            S[7] = smax / 2.0;

            for (size_t i = 4; i < S.size(); ++i) {
                if (i % 2 != 0) {
                    continue;
                }
                a = SQR(dE);
                b = 2.0 * E * dE;
                c = SQR(E) + std::pow(gamma + S[i] * dgamma, 2.0) - SQR(y);
                D = SQR(b) - 4.0 * a * c;
                P[i] = (- b + std::sqrt(D)) / (2.0 * a);
                P[i + 1] = (- b - std::sqrt(D)) / (2.0 * a);

            }

            for (size_t i = 0; i < S.size(); ++i) {
                xt::xtensor<double, 4> sig = P[i] * Sig_p + S[i] * Sig_s + Sig_slice;
                xt::xtensor<double, 4> deps = P[i] * Eps_p + S[i] * Eps_s;
                xt::xtensor<double, 2> w = GT::A2s_ddot_B2s(sig, deps);
                w *= dV_slice;
                W[i] = xt::sum(w)();
            }

            auto idx = std::distance(W.begin(), std::min_element(W.begin(), W.end()));
            m_smin(e, q) = S[idx];
            m_pmin(e, q) = P[idx];
            m_Wmin(e, q) = W[idx];
        }
    }
}

inline void LocalTriggerFineLayerFull::setStateSimpleShear(
    const xt::xtensor<double, 4>& Eps,
    const xt::xtensor<double, 4>& Sig,
    const xt::xtensor<double, 2>& epsy)
{
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(Eps, m_shape_T2));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(Sig, m_shape_T2));
    FRICTIONQPOTFEM_ASSERT(xt::has_shape(epsy, {m_nelem_plas, m_nip}));

    std::array<double, 2> S;
    std::array<double, 2> W;

    for (size_t e = 0; e < m_nelem_plas; ++e) {

        auto Eps_s = this->Eps_s(e);
        auto Sig_s = this->Sig_s(e);
        auto Sig_slice = this->slice(Sig, e);
        auto dV_slice = this->slice(m_dV, e);

        for (size_t q = 0; q < m_nip; ++q) {

            double dgamma = m_dgamma(e, q);

            auto Epsd = GM::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            double E = Epsd(0, 0);
            double y = epsy(e, q);
            double a, b, c, D;

            // solve for "p = 0"
            a = SQR(dgamma);
            b = 2.0 * gamma * dgamma;
            c = SQR(gamma) + SQR(E) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            S[1] = (- b + std::sqrt(D)) / (2.0 * a);
            S[0] = (- b - std::sqrt(D)) / (2.0 * a);

            for (size_t i = 0; i < S.size(); ++i) {
                xt::xtensor<double, 4> sig = S[i] * Sig_s + Sig_slice;
                xt::xtensor<double, 4> deps = S[i] * Eps_s;
                xt::xtensor<double, 2> w = GT::A2s_ddot_B2s(sig, deps);
                w *= dV_slice;
                W[i] = xt::sum(w)();
            }

            size_t idx = 0;
            if (W[1] < W[0]) {
                idx = 1;
            }

            m_smin(e, q) = S[idx];
            m_pmin(e, q) = 0.0;
            m_Wmin(e, q) = W[idx];
        }
    }
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::barriers() const
{
    return m_Wmin / m_V;
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::p() const
{
    return m_pmin;
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::s() const
{
    return m_smin;
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::dgamma() const
{
    return m_dgamma;
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::dE() const
{
    return m_dE;
}

inline xt::xtensor<double, 2> LocalTriggerFineLayerFull::delta_u(size_t e, size_t q) const
{
    return m_pmin(e, q) * this->u_p(e) + m_smin(e, q) * this->u_s(e);
}

inline LocalTriggerFineLayer::LocalTriggerFineLayer(const System& sys, size_t roi) :
    LocalTriggerFineLayerFull::LocalTriggerFineLayerFull(sys)
{
    GF::Mesh::Quad4::FineLayer mesh(sys.coor(), sys.conn());

    m_elemslice.resize(m_nelem_plas);
    m_Eps_s_slice.resize(m_nelem_plas);
    m_Eps_p_slice.resize(m_nelem_plas);
    m_Sig_s_slice.resize(m_nelem_plas);
    m_Sig_p_slice.resize(m_nelem_plas);

    for (size_t e = 0; e < m_nelem_plas; ++e) {

        auto slice = xt::sort(mesh.elementgrid_around_ravel(m_elem_plas(e), roi));
        m_elemslice[e] = slice;

        auto Eps = LocalTriggerFineLayerFull::Eps_s(e);
        m_Eps_s_slice[e] = xt::view(Eps, xt::keep(slice));

        Eps = LocalTriggerFineLayerFull::Eps_p(e);
        m_Eps_p_slice[e] = xt::view(Eps, xt::keep(slice));

        auto Sig = LocalTriggerFineLayerFull::Sig_s(e);
        m_Sig_s_slice[e] = xt::view(Sig, xt::keep(slice));

        Sig = LocalTriggerFineLayerFull::Sig_p(e);
        m_Sig_p_slice[e] = xt::view(Sig, xt::keep(slice));
    }
}

inline xt::xtensor<double, 2>
LocalTriggerFineLayer::slice(const xt::xtensor<double, 2>& arg, size_t e) const
{
    return xt::view(arg, xt::keep(m_elemslice[e]));
}

inline xt::xtensor<double, 4>
LocalTriggerFineLayer::slice(const xt::xtensor<double, 4>& arg, size_t e) const
{
    return xt::view(arg, xt::keep(m_elemslice[e]));
}

inline xt::xtensor<double, 4> LocalTriggerFineLayer::Eps_s(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    return m_Eps_s_slice[trigger_plastic];
}

inline xt::xtensor<double, 4> LocalTriggerFineLayer::Eps_p(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    return m_Eps_p_slice[trigger_plastic];
}

inline xt::xtensor<double, 4> LocalTriggerFineLayer::Sig_s(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    return m_Sig_s_slice[trigger_plastic];
}

inline xt::xtensor<double, 4> LocalTriggerFineLayer::Sig_p(size_t trigger_plastic) const
{
    FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
    return m_Sig_p_slice[trigger_plastic];
}

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#endif
