/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_HPP
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_HPP

#include "UniformSingleLayer2d.h"

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

inline std::vector<std::string> version_dependencies()
{
    return Generic2d::version_dependencies();
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
    this->init(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

template <class C, class E, class L>
inline void System::init(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initHybridSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
}

inline std::string System::type() const
{
    return "FrictionQPotFEM.UniformSingleLayer2d.System";
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

inline xt::xtensor<double, 2> System::Energy()
{
    FRICTIONQPOTFEM_WARNING_PYTHON(
        "Deprecated: use this.material().Energy()."
        "Careful though: one has to call 'evalSystem()' every time after 'setU'");

    this->evalSystem();
    return m_material.Energy();
}

inline auto System::plastic_signOfPerturbation(const xt::xtensor<double, 2>& delta_u)
{
    FRICTIONQPOTFEM_WARNING_PYTHON(
        "Use plastic_SignDeltaEpsd(...) instead of plastic_signOfPerturbation(...)");

    return this->plastic_SignDeltaEpsd(delta_u);
}

inline auto System::plastic_signOfSimpleShearPerturbation(double perturbation)
{
    auto u_0 = this->u();
    auto eps_0 = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
    auto u_pert = this->u();

    for (size_t n = 0; n < m_nnode; ++n) {
        u_pert(n, 0) += perturbation * (m_coor(n, 1) - m_coor(0, 1));
    }

    this->setU(u_pert);
    auto eps_pert = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
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

inline double System::addAffineSimpleShearCentered(double delta_gamma)
{
    size_t ll = m_conn(m_elem_plas(0), 0);
    size_t ul = m_conn(m_elem_plas(0), 3);
    double y0 = (m_coor(ul, 1) + m_coor(ll, 1)) / 2.0;
    auto u_new = this->u();

    for (size_t n = 0; n < m_nnode; ++n) {
        u_new(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - y0);
    }
    this->setU(u_new);

    return delta_gamma * 2.0;
}

inline void System::initEventDrivenSimpleShear()
{
    FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());

    auto u = xt::zeros_like(m_u);

    for (size_t n = 0; n < m_nnode; ++n) {
        u(n, 0) += m_coor(n, 1) - m_coor(0, 1);
    }

    this->eventDriven_setDeltaU(u);
}

inline double
System::addSimpleShearEventDriven(double deps_kick, bool kick, double direction, bool dry_run)
{
    FRICTIONQPOTFEM_WARNING_PYTHON("Use initEventDrivenSimpleShear() + eventDrivenStep(...) "
                                   "instead of addSimpleShearEventDriven(...)");

    FRICTIONQPOTFEM_REQUIRE(direction > 0.0); // bugged, use new implementation

    FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());
    FRICTIONQPOTFEM_REQUIRE(direction == +1.0 || direction == -1.0);

    auto idx = this->plastic_CurrentIndex();
    auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
    auto Epsd = GMatTensor::Cartesian2d::Deviatoric(this->plastic_Eps());
    auto epsxx = xt::view(Epsd, xt::all(), xt::all(), 0, 0);
    auto epsxy = xt::view(Epsd, xt::all(), xt::all(), 0, 1);

    // distance to yielding: "deps"
    // (event a positive kick can lead to a decreasing equivalent strain)
    xt::xtensor<double, 2> sign =
        this->plastic_signOfSimpleShearPerturbation(direction * deps_kick);

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
        m_u(n, 0) += direction * dux * (m_coor(n, 1) - m_coor(0, 1));
    }
    this->updated_u();

    // sanity check
    // ------------

    auto index = xt::unravel_index(xt::argmin(dgamma)(), dgamma.shape());
    size_t e = index[0];
    size_t q = index[1];

    eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
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
    auto dV = m_quad.AsTensor<2>(this->dV());
    double G = m_material.G().data()[0];

    xt::xtensor<double, 2> Epsbar = xt::average(this->Eps(), dV, {0, 1});
    xt::xtensor<double, 2> Sigbar = xt::average(this->Sig(), dV, {0, 1});
    xt::xtensor<double, 2> Epsd = GMatTensor::Cartesian2d::Deviatoric(Epsbar);
    double epsxx = Epsd(0, 0);
    double epsxy = Epsd(0, 1);

    FRICTIONQPOTFEM_ASSERT(Sigbar(0, 1) >= 0);

    double eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)();
    double sig = GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)();
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
    sig = GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)();

    FRICTIONQPOTFEM_REQUIRE(std::abs(target_stress - sig) / sig < 1e-4);

    return direction * dgamma;
}

inline double System::addElasticSimpleShearToFixedStress(double target_stress, bool dry_run)
{
    auto idx = this->plastic_CurrentIndex();
    auto ret = this->addSimpleShearToFixedStress(target_stress, dry_run);
    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::equal(idx, this->plastic_CurrentIndex())));
    return ret;
}

inline double System::triggerElementWithLocalSimpleShear(
    double deps_kick,
    size_t plastic_element,
    bool trigger_weakest,
    double amplify)
{
    FRICTIONQPOTFEM_ASSERT(plastic_element < m_nelem_plas);

    auto idx = this->plastic_CurrentIndex();
    auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
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
    xt::xtensor<double, 2> Epsd = GMatTensor::Cartesian2d::Deviatoric(Eps);

    // new equivalent deviatoric strain: yield strain + small strain kick
    double eps_new = epsy(plastic_element, q) + deps_kick / 2.0;

    // convert to increment in shear strain (N.B. "dgamma = 2 * Epsd(0,1)")
    double dgamma =
        2.0 * (-Epsd(0, 1) + std::sqrt(std::pow(eps_new, 2.0) - std::pow(Epsd(0, 0), 2.0)));

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

    eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
    auto idx_new = this->plastic_CurrentIndex();
    auto up_new = m_vector.AsDofs_p(m_u);

    FRICTIONQPOTFEM_REQUIRE(dgamma >= 0.0);
    FRICTIONQPOTFEM_REQUIRE(amplify >= 0.0);
    FRICTIONQPOTFEM_REQUIRE(
        std::abs(eps(plastic_element, q) - eps_new) / eps_new < 1e-4 || amplify != 1);
    FRICTIONQPOTFEM_REQUIRE(xt::any(xt::not_equal(idx, idx_new)));
    FRICTIONQPOTFEM_REQUIRE(idx(plastic_element, q) != idx_new(plastic_element, q));
    FRICTIONQPOTFEM_REQUIRE(xt::allclose(up, up_new));

    return dgamma;
}

inline xt::xtensor<double, 2>
System::plastic_ElementYieldBarrierForSimpleShear(double deps_kick, size_t iquad)
{
    FRICTIONQPOTFEM_ASSERT(iquad < m_nip);

    auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
    auto epsy = this->plastic_CurrentYieldRight();
    auto deps = xt::eval(epsy - eps);
    xt::xtensor<double, 2> ret = xt::empty<double>({m_N, size_t(2)});
    auto isort = xt::argsort(deps, 1);

    auto Eps = this->plastic_Eps();
    auto Epsd = GMatTensor::Cartesian2d::Deviatoric(Eps);

    for (size_t e = 0; e < m_N; ++e) {
        size_t q = isort(e, iquad);
        double eps_new = epsy(e, q) + deps_kick / 2.0;
        double gamma = Epsd(e, q, 0, 1);
        double epsd_xx = Epsd(e, q, 0, 0);
        ret(e, 0) = deps(e, q);
        ret(e, 1) = -gamma + std::sqrt(std::pow(eps_new, 2.0) - std::pow(epsd_xx, 2.0));
    }

    return ret;
}

inline LocalTriggerFineLayerFull::LocalTriggerFineLayerFull(const System& sys)
{
    // Copy / allocate local variables

    auto m = sys.material();
    GMatElastoPlasticQPot::Cartesian2d::Array<2> material(m.shape());
    material.setElastic(m.K(), m.G());

    FRICTIONQPOTFEM_ASSERT(
        xt::all(xt::equal(material.type(), GMatElastoPlasticQPot::Cartesian2d::Type::Elastic)));

    auto vector = sys.vector();
    auto K = sys.stiffness();
    auto quad = sys.quad();
    GooseFEM::MatrixPartitionedSolver<> solver;

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

    auto u = vector.allocate_nodevec(0.0);
    auto fint = vector.allocate_nodevec(0.0);
    auto fext = vector.allocate_nodevec(0.0);
    auto fres = vector.allocate_nodevec(0.0);

    auto ue = vector.allocate_elemvec(0.0);
    auto fe = vector.allocate_elemvec(0.0);

    auto Eps = quad.allocate_qtensor<2>(0.0);
    auto Sig = quad.allocate_qtensor<2>(0.0);
    std::copy(Eps.shape().cbegin(), Eps.shape().cend(), m_shape_T2.begin());

    m_dV = quad.dV();
    m_V = xt::sum(xt::view(m_dV, m_elem_plas(0)))();

    // Replicate mesh

    GooseFEM::Mesh::Quad4::FineLayer mesh(sys.coor(), sys.conn());

    auto elmap = mesh.roll(1);
    size_t nconfig = m_elem_plas(m_nelem_plas - 1) - elmap(m_elem_plas(m_nelem_plas - 1));
    size_t nroll = m_nelem_plas / nconfig;

    for (size_t roll = 0; roll < nroll; ++roll) {
        m_elemmap.push_back(mesh.roll(roll));
        m_nodemap.push_back(GooseFEM::Mesh::elemmap2nodemap(m_elemmap[roll], coor, conn));
    }

    for (size_t e = 0; e < nconfig; ++e) {

        // Simple shear perturbation

        m_u_s.emplace_back(u.shape());
        m_Eps_s.emplace_back(Eps.shape());
        m_Sig_s.emplace_back(Sig.shape());

        xt::xtensor<double, 2> simple_shear = {{0.0, 1.0}, {1.0, 0.0}};

        this->computePerturbation(
            e, simple_shear, m_u_s[e], m_Eps_s[e], m_Sig_s[e], K, solver, quad, vector, material);

        for (size_t q = 0; q < m_nip; ++q) {
            auto Epsd = GMatTensor::Cartesian2d::Deviatoric(
                xt::eval(xt::view(m_Eps_s[e], m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            for (size_t roll = 0; roll < nroll; ++roll) {
                auto map = mesh.roll(roll);
                m_dgamma(e + map(m_elem_plas(e)) - m_elem_plas(e), q) = gamma;
            }
        }

        // Pure shear perturbation

        m_u_p.emplace_back(u.shape());
        m_Eps_p.emplace_back(Eps.shape());
        m_Sig_p.emplace_back(Sig.shape());

        xt::xtensor<double, 2> pure_shear = {{1.0, 0.0}, {0.0, -1.0}};

        this->computePerturbation(
            e, pure_shear, m_u_p[e], m_Eps_p[e], m_Sig_p[e], K, solver, quad, vector, material);

        for (size_t q = 0; q < m_nip; ++q) {
            auto Epsd = GMatTensor::Cartesian2d::Deviatoric(
                xt::eval(xt::view(m_Eps_p[e], m_elem_plas(e), q)));
            double E = Epsd(0, 0);
            for (size_t roll = 0; roll < nroll; ++roll) {
                auto map = mesh.roll(roll);
                m_dE(e + map(m_elem_plas(e)) - m_elem_plas(e), q) = E;
            }
        }
    }
}

inline void LocalTriggerFineLayerFull::computePerturbation(
    size_t trigger_plastic,
    const xt::xtensor<double, 2>& sig_star,
    xt::xtensor<double, 2>& u,
    xt::xtensor<double, 4>& Eps,
    xt::xtensor<double, 4>& Sig,
    GooseFEM::MatrixPartitioned& K,
    GooseFEM::MatrixPartitionedSolver<>& solver,
    const GooseFEM::Element::Quad4::Quadrature& quad,
    const GooseFEM::VectorPartitioned& vector,
    GMatElastoPlasticQPot::Cartesian2d::Array<2>& material)
{
    size_t trigger_element = m_elem_plas(trigger_plastic);

    Sig.fill(0.0);
    u.fill(0.0);

    for (size_t q = 0; q < quad.nip(); ++q) {
        xt::view(Sig, trigger_element, q) = -sig_star;
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

            auto Epsd =
                GMatTensor::Cartesian2d::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            double E = Epsd(0, 0);
            double y = epsy(e, q);
            double a, b, c, D;

            // solve for "p = 0"
            a = SQR(dgamma);
            b = 2.0 * gamma * dgamma;
            c = SQR(gamma) + SQR(E) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double smax = (-b + std::sqrt(D)) / (2.0 * a);
            double smin = (-b - std::sqrt(D)) / (2.0 * a);

            // solve for "s = 0"
            a = SQR(dE);
            b = 2.0 * E * dE;
            c = SQR(E) + SQR(gamma) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double pmax = (-b + std::sqrt(D)) / (2.0 * a);
            double pmin = (-b - std::sqrt(D)) / (2.0 * a);

            size_t n = static_cast<size_t>(-smin / (smax - smin) * static_cast<double>(N));
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
                P(0, j) = (-b + std::sqrt(D)) / (2.0 * a);
                P(1, j) = (-b - std::sqrt(D)) / (2.0 * a);
            }

            for (size_t i = 0; i < P.shape(0); ++i) {
                for (size_t j = 0; j < P.shape(1); ++j) {
                    xt::xtensor<double, 4> sig = P(i, j) * Sig_p + S(i, j) * Sig_s + Sig_slice;
                    xt::xtensor<double, 4> deps = P(i, j) * Eps_p + S(i, j) * Eps_s;
                    xt::xtensor<double, 2> w = GMatTensor::Cartesian2d::A2s_ddot_B2s(sig, deps);
                    w *= dV_slice;
                    W(i, j) = xt::sum(w)();
                }
            }

            auto idx = xt::argmin(W)();
            m_smin(e, q) = S.flat(idx);
            m_pmin(e, q) = P.flat(idx);
            m_Wmin(e, q) = W.flat(idx);
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

            auto Epsd =
                GMatTensor::Cartesian2d::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            double E = Epsd(0, 0);
            double y = epsy(e, q);
            double a, b, c, D;

            // solve for "p = 0"
            a = SQR(dgamma);
            b = 2.0 * gamma * dgamma;
            c = SQR(gamma) + SQR(E) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double smax = (-b + std::sqrt(D)) / (2.0 * a);
            double smin = (-b - std::sqrt(D)) / (2.0 * a);
            P[0] = 0.0;
            P[1] = 0.0;
            S[0] = smin;
            S[1] = smax;

            // solve for "s = 0"
            a = SQR(dE);
            b = 2.0 * E * dE;
            c = SQR(E) + SQR(gamma) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            double pmax = (-b + std::sqrt(D)) / (2.0 * a);
            double pmin = (-b - std::sqrt(D)) / (2.0 * a);
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
                P[i] = (-b + std::sqrt(D)) / (2.0 * a);
                P[i + 1] = (-b - std::sqrt(D)) / (2.0 * a);
            }

            for (size_t i = 0; i < S.size(); ++i) {
                xt::xtensor<double, 4> sig = P[i] * Sig_p + S[i] * Sig_s + Sig_slice;
                xt::xtensor<double, 4> deps = P[i] * Eps_p + S[i] * Eps_s;
                xt::xtensor<double, 2> w = GMatTensor::Cartesian2d::A2s_ddot_B2s(sig, deps);
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

            auto Epsd =
                GMatTensor::Cartesian2d::Deviatoric(xt::eval(xt::view(Eps, m_elem_plas(e), q)));
            double gamma = Epsd(0, 1);
            double E = Epsd(0, 0);
            double y = epsy(e, q);
            double a, b, c, D;

            // solve for "p = 0"
            a = SQR(dgamma);
            b = 2.0 * gamma * dgamma;
            c = SQR(gamma) + SQR(E) - SQR(y);
            D = SQR(b) - 4.0 * a * c;
            S[1] = (-b + std::sqrt(D)) / (2.0 * a);
            S[0] = (-b - std::sqrt(D)) / (2.0 * a);

            for (size_t i = 0; i < S.size(); ++i) {
                xt::xtensor<double, 4> sig = S[i] * Sig_s + Sig_slice;
                xt::xtensor<double, 4> deps = S[i] * Eps_s;
                xt::xtensor<double, 2> w = GMatTensor::Cartesian2d::A2s_ddot_B2s(sig, deps);
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

inline LocalTriggerFineLayer::LocalTriggerFineLayer(const System& sys, size_t roi)
    : LocalTriggerFineLayerFull::LocalTriggerFineLayerFull(sys)
{
    GooseFEM::Mesh::Quad4::FineLayer mesh(sys.coor(), sys.conn());

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
