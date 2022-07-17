/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H

#include "Generic2d.h"
#include "config.h"

/**
\return x^2
*/
#define SQR(x) ((x) * (x))

namespace FrictionQPotFEM {

/**
System in 2-d with:

-   A weak, middle, layer.
-   Uniform elasticity.
*/
namespace UniformSingleLayer2d {

/**
\copydoc Generic2d::version_dependencies()
*/
inline std::vector<std::string> version_dependencies()
{
    return Generic2d::version_dependencies();
}

/**
Class that uses GMatElastoPlasticQPot to evaluate stress everywhere
*/
class System : public Generic2d::System {

public:
    System() = default;

public:
    virtual ~System(){};

public:
    /**
    Define the geometry, including boundary conditions and element sets.

    \tparam C Type of nodal coordinates, e.g. `array_type::tensor<double, 2>`
    \tparam E Type of connectivity and DOFs, e.g. `array_type::tensor<size_t, 2>`
    \tparam L Type of node/element lists, e.g. `array_type::tensor<size_t, 1>`
    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem_elastic Elastic elements.
    \param elem_plastic Plastic elements.
    */
    template <class C, class E, class L>
    System(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elem_elastic,
        const L& elem_plastic)
    {
        this->init(coor, conn, dofs, iip, elem_elastic, elem_plastic);
    }

protected:
    /**
    Constructor alias, useful for derived classes.

    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem_elastic Elastic elements.
    \param elem_plastic Plastic elements.
    */
    template <class C, class E, class L>
    void init(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elem_elastic,
        const L& elem_plastic)
    {
        this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
    }

public:
    std::string type() const override
    {
        return "FrictionQPotFEM.UniformSingleLayer2d.System";
    }

public:
    /**
    Element height of all elements along the weak layer.
    \return Element height (scalar).
    */
    double typical_plastic_h() const
    {
        auto bot = xt::view(m_conn, xt::keep(m_elem_plas), 0);
        auto top = xt::view(m_conn, xt::keep(m_elem_plas), 3);
        auto h_plas = xt::view(m_coor, xt::keep(top), 1) - xt::view(m_coor, xt::keep(bot), 1);
        double h = h_plas(0);
        FRICTIONQPOTFEM_ASSERT(xt::allclose(h_plas, h));
        return h;
    }

public:
    /**
    Integration point volume of all elements along the weak layer.
    \return Integration point volume.
    */
    double typical_plastic_dV() const
    {
        auto dV_plas = xt::view(m_quad.dV(), xt::keep(m_elem_plas), xt::all());
        double dV = dV_plas(0, 0);
        FRICTIONQPOTFEM_ASSERT(xt::allclose(dV_plas, dV));
        return dV;
    }

public:
    /**
    Initialise event driven protocol for affine simple shear.
    */
    void initEventDrivenSimpleShear()
    {
        FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());
        this->eventDriven_setDeltaU(this->affineSimpleShear(0.5));
    }

public:
    /**
    Add simple shear until a target equivalent macroscopic stress has been reached.
    Depending of the target stress compared to the current equivalent macroscopic stress,
    the shear can be either to the left or to the right.

    \param target_stress
        Target stress (equivalent deviatoric value of the macroscopic stress tensor).

    \param dry_run
        If ``true`` do not apply displacement, do not check.

    \return
        xy-component of the deformation gradient that is applied to the system.
    */
    double addSimpleShearToFixedStress(double target_stress, bool dry_run = false)
    {
        FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());

        auto u_new = this->u();
        auto dV = m_quad.AsTensor<2>(this->dV());
        double G = m_material_plas.G().data()[0];

        array_type::tensor<double, 2> Epsbar = xt::average(this->Eps(), dV, {0, 1});
        array_type::tensor<double, 2> Sigbar = xt::average(this->Sig(), dV, {0, 1});
        array_type::tensor<double, 2> Epsd = GMatTensor::Cartesian2d::Deviatoric(Epsbar);
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
        double dgamma = 2.0 * (-1.0 * direction * epsxy +
                               std::sqrt(std::pow(eps_new, 2.0) - std::pow(epsxx, 2.0)));

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

public:
    /**
    \copydoc addSimpleShearToFixedStress(double, bool)

    \throw Throws if yielding is triggered before the stress was reached.
    */
    double addElasticSimpleShearToFixedStress(double target_stress, bool dry_run = false)
    {
        auto idx = this->plastic_CurrentIndex();
        auto ret = this->addSimpleShearToFixedStress(target_stress, dry_run);
        FRICTIONQPOTFEM_REQUIRE(xt::all(xt::equal(idx, this->plastic_CurrentIndex())));
        return ret;
    }

public:
    /**
    Apply local strain to the right to a specific plastic element.
    This 'triggers' one element while keeping the boundary conditions unchanged.
    Note that by applying shear to the element, yielding can also be triggered in
    the surrounding elements.

    \param deps_kick
        Size of the local stain kick to apply.

    \param plastic_element
        Which plastic element to trigger: System::plastic()(plastic_element).

    \param trigger_weakest
        If ``true``, trigger the weakest integration point.
        If ``false``, trigger the strongest.

    \param amplify
        Amplify the strain kick with a certain factor.

    \return
        xy-component of the deformation gradient that is applied to the system.
    */
    double triggerElementWithLocalSimpleShear(
        double deps_kick,
        size_t plastic_element,
        bool trigger_weakest = true,
        double amplify = 1.0)
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
        array_type::tensor<double, 2> Eps = xt::view(this->plastic_Eps(), plastic_element, q);
        array_type::tensor<double, 2> Epsd = GMatTensor::Cartesian2d::Deviatoric(Eps);

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
            udofs(m_dofs(elem(n), 0)) +=
                dgamma * amplify * (m_coor(elem(n), 1) - m_coor(elem(0), 1));
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

public:
    /**
    Read the distance to overcome the first cusp in the element.

    \param deps_kick
        Size of the local stain kick to apply.

    \param iquad
        Which integration point to check:
        - weakest: ``iquad = 0``
        - strongest: ``iquad = nip - 1``

    \return
        Array of shape ``[N, 2]`` with columns ``[delta_eps, delta_epsxy]``.
    */
    array_type::tensor<double, 2>
    plastic_ElementYieldBarrierForSimpleShear(double deps_kick = 0.0, size_t iquad = 0)
    {
        FRICTIONQPOTFEM_ASSERT(iquad < m_nip);

        auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
        auto epsy = this->plastic_CurrentYieldRight();
        auto deps = xt::eval(epsy - eps);
        array_type::tensor<double, 2> ret = xt::empty<double>({m_N, size_t(2)});
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
};

/**
Trigger element by a linear combination of simple shear and a pure shear perturbations.
The contribution of both perturbation is computed as the minimal
energy barrier needed to reach a yield surface for the triggered element, see:

- LocalTriggerFineLayerFull::setState
- LocalTriggerFineLayerFull::setStateMinimalSearch
- LocalTriggerFineLayerFull::setStateSimpleShear

The perturbations are established as the displacement field needed to reach mechanical equilibrium
when an eigen-stress is applied to the triggered element (assuming elasticity everywhere).
To get the two perturbations a sinple shear and pure shear eigen-stress are applied.

The configuration, including the definition of elasticity is read from the input System.
*/
class LocalTriggerFineLayerFull {
public:
    LocalTriggerFineLayerFull() = default;

    /**
    Constructor, reading the basic properties of the System, and computing the perturbation
    for all plastic elements.
    The perturbations of all elements are stored internally, making the computation of the
    energy barriers cheap.
    Note that this can use significant memory.

    \param sys The System (or derived class) to trigger.
    */
    LocalTriggerFineLayerFull(const System& sys)
    {
        // Copy / allocate local variables

        auto kappa = sys.K();
        auto G = sys.G();
        std::array<size_t, 2> shape = {kappa.shape(0), kappa.shape(1)};
        GMatElastoPlasticQPot::Cartesian2d::Array<2> material(shape);
        material.setElastic(kappa, G);

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

            array_type::tensor<double, 2> simple_shear = {{0.0, 1.0}, {1.0, 0.0}};

            this->computePerturbation(
                e,
                simple_shear,
                m_u_s[e],
                m_Eps_s[e],
                m_Sig_s[e],
                K,
                solver,
                quad,
                vector,
                material);

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

            array_type::tensor<double, 2> pure_shear = {{1.0, 0.0}, {0.0, -1.0}};

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

    virtual ~LocalTriggerFineLayerFull(){};

protected:
    /**
    Compute the displacement response to an eigen-stress applied to a plastic element.

    \param trigger_plastic Index of the plastic element.
    \param sig_star Eigen-stress applied to ``trigger_plastic``.
    \param u Output displacement field.
    \param Eps Output integration point strain corresponding to ``u``.
    \param Sig Output integration point stress corresponding to ``u``.
    \param K Stiffness matrix of the System.
    \param solver Diagonalisation of ``K``.
    \param quad Numerical quadrature of the System.
    \param vector Book-keeping of the System.
    \param material Material definition of the System.
    */
    void computePerturbation(
        size_t trigger_plastic,
        const array_type::tensor<double, 2>& sig_star,
        array_type::tensor<double, 2>& u,
        array_type::tensor<double, 4>& Eps,
        array_type::tensor<double, 4>& Sig,
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

public:
    /**
    Displacement field for the simple shear eigen-stress applied to a specific element.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(trigger_plastic).
    Function reads from memory, all computations are done in the constructor.

    \param trigger_plastic Index of the plastic element.
    \return Nodal displacement. Shape [System::m_nelem, System::m_nip].
    */
    array_type::tensor<double, 2> u_s(size_t trigger_plastic) const
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        size_t nconfig = m_u_s.size();
        size_t config = trigger_plastic % nconfig;
        size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
        return xt::view(m_u_s[config], xt::keep(m_nodemap[roll]));
    }

    /**
    Displacement field for the pure shear eigen-stress applied to a specific element.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(trigger_plastic).
    Function reads from memory, all computations are done in the constructor.

    \param trigger_plastic Index of the plastic element.
    \return Nodal displacement. Shape [System::m_nelem, System::m_nip].
    */
    array_type::tensor<double, 2> u_p(size_t trigger_plastic) const
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        size_t nconfig = m_u_p.size();
        size_t config = trigger_plastic % nconfig;
        size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
        return xt::view(m_u_p[config], xt::keep(m_nodemap[roll]));
    }

    /**
    Integration point strain tensors for LocalTriggerFineLayerFull::u_s.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(trigger_plastic).
    Function reads from memory, all computations are done in the constructor.

    \param trigger_plastic Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual array_type::tensor<double, 4> Eps_s(size_t trigger_plastic) const
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        size_t nconfig = m_u_s.size();
        size_t config = trigger_plastic % nconfig;
        size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
        return xt::view(m_Eps_s[config], xt::keep(m_elemmap[roll]));
    }

    /**
    Integration point strain tensors for LocalTriggerFineLayerFull::u_p.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(trigger_plastic).
    Function reads from memory, all computations are done in the constructor.

    \param trigger_plastic Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual array_type::tensor<double, 4> Eps_p(size_t trigger_plastic) const
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        size_t nconfig = m_u_p.size();
        size_t config = trigger_plastic % nconfig;
        size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
        return xt::view(m_Eps_p[config], xt::keep(m_elemmap[roll]));
    }

    /**
    Integration point stress tensors for LocalTriggerFineLayerFull::u_s.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(trigger_plastic).
    Function reads from memory, all computations are done in the constructor.

    \param trigger_plastic Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual array_type::tensor<double, 4> Sig_s(size_t trigger_plastic) const
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        size_t nconfig = m_u_s.size();
        size_t config = trigger_plastic % nconfig;
        size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
        return xt::view(m_Sig_s[config], xt::keep(m_elemmap[roll]));
    }

    /**
    Integration point stress tensors for LocalTriggerFineLayerFull::u_p.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(trigger_plastic).
    Function reads from memory, all computations are done in the constructor.

    \param trigger_plastic Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual array_type::tensor<double, 4> Sig_p(size_t trigger_plastic) const
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        size_t nconfig = m_u_p.size();
        size_t config = trigger_plastic % nconfig;
        size_t roll = (trigger_plastic - trigger_plastic % nconfig) / nconfig;
        return xt::view(m_Sig_p[config], xt::keep(m_elemmap[roll]));
    }

    /**
    Empty function, used by LocalTriggerFineLayer.

    \param arg Integration point scalar.
    \param e Index of the plastic element.
    \return A copy of ``arg``.
    */
    virtual array_type::tensor<double, 2>
    slice(const array_type::tensor<double, 2>& arg, size_t e) const
    {
        UNUSED(e);
        return arg;
    }

    /**
    Empty function, used by LocalTriggerFineLayer.

    \param arg Integration point tensor.
    \param e Index of the plastic element.
    \return A copy of ``arg``.
    */
    virtual array_type::tensor<double, 4>
    slice(const array_type::tensor<double, 4>& arg, size_t e) const
    {
        UNUSED(e);
        return arg;
    }

    /**
    Set current state and compute energy barriers to reach the specified yield surface
    (for all plastic elements).
    The yield surface is discretised in ``N`` steps.

    \param Eps Integration point strain, see System::Eps.
    \param Sig Integration point stress, see System::Sig.
    \param epsy Next yield strains, see System::plastic_CurrentYieldRight.
    \param N Number of steps in which to discretise the yield surface.
    */
    void setState(
        const array_type::tensor<double, 4>& Eps,
        const array_type::tensor<double, 4>& Sig,
        const array_type::tensor<double, 2>& epsy,
        size_t N = 100)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(Eps, m_shape_T2));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(Sig, m_shape_T2));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(epsy, {m_nelem_plas, m_nip}));

        array_type::tensor<double, 2> S = xt::empty<double>({size_t(2), N});
        array_type::tensor<double, 2> P = xt::empty<double>({size_t(2), N});
        array_type::tensor<double, 2> W = xt::empty<double>({size_t(2), N});

        for (size_t i = 0; i < S.shape(0); ++i) {
            S(i, 0) = 0.0;
            P(i, 0) = 0.0;
            W(i, 0) = 0.0;
        }

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
                    xt::view(S, i, xt::range(n, N)) =
                        xt::linspace<double>(smax / double(m), smax, m);
                    P(i, 0) = 0.0;
                    P(i, N - 1) = 0.0;
                }
                if (n > 0) {
                    P(0, n - 1) = pmax;
                    P(1, n - 1) = pmin;
                }

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
                        array_type::tensor<double, 4> sig =
                            P(i, j) * Sig_p + S(i, j) * Sig_s + Sig_slice;

                        array_type::tensor<double, 4> deps = P(i, j) * Eps_p + S(i, j) * Eps_s;

                        array_type::tensor<double, 2> w =
                            GMatTensor::Cartesian2d::A2s_ddot_B2s(sig, deps);

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

    /**
    Set current state and compute energy barriers to reach the specified yield surface
    (for all plastic elements).
    The yield surface is discretised in only ``8`` steps.

    \param Eps Integration point strain, see System::Eps.
    \param Sig Integration point stress, see System::Sig.
    \param epsy Next yield strains, see System::plastic_CurrentYieldRight.
    */
    void setStateMinimalSearch(
        const array_type::tensor<double, 4>& Eps,
        const array_type::tensor<double, 4>& Sig,
        const array_type::tensor<double, 2>& epsy)
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
                    array_type::tensor<double, 4> sig = P[i] * Sig_p + S[i] * Sig_s + Sig_slice;
                    array_type::tensor<double, 4> deps = P[i] * Eps_p + S[i] * Eps_s;
                    array_type::tensor<double, 2> w =
                        GMatTensor::Cartesian2d::A2s_ddot_B2s(sig, deps);
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

    /**
    Set current state and compute energy barriers to reach the specified yield surface,
    for a purely simple shear perturbation (for all plastic elements)

    \param Eps Integration point strain, see System::Eps.
    \param Sig Integration point stress, see System::Sig.
    \param epsy Next yield strains, see System::plastic_CurrentYieldRight.
    */
    void setStateSimpleShear(
        const array_type::tensor<double, 4>& Eps,
        const array_type::tensor<double, 4>& Sig,
        const array_type::tensor<double, 2>& epsy)
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
                    array_type::tensor<double, 4> sig = S[i] * Sig_s + Sig_slice;
                    array_type::tensor<double, 4> deps = S[i] * Eps_s;
                    array_type::tensor<double, 2> w =
                        GMatTensor::Cartesian2d::A2s_ddot_B2s(sig, deps);
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

    /**
    Get all energy barriers, as energy density.
    Shape of output: [LocalTriggerFineLayerFull::nelem_elas, LocalTriggerFineLayerFull::nip].
    Function reads from memory, all computations are done in the construction and
    LocalTriggerFineLayerFull::setState (or one of its approximations).

    \return Energy barriers.
    */
    array_type::tensor<double, 2> barriers() const
    {
        return m_Wmin / m_V;
    }

    /**
    The energy barrier in LocalTriggerFineLayerFull::barriers is reached with a displacement
    LocalTriggerFineLayerFull::delta_u =
    ``p`` * LocalTriggerFineLayerFull::u_p + ``s`` * LocalTriggerFineLayerFull::u_s.
    This function returns the value of ``p``.
    Shape of output: [LocalTriggerFineLayerFull::nelem_elas, LocalTriggerFineLayerFull::nip].
    Function reads from memory, all computations are done in the construction and
    LocalTriggerFineLayerFull::setState (or one of its approximations).

    \return Pure shear contribution.
    */
    const array_type::tensor<double, 2>& p() const
    {
        return m_pmin;
    }

    /**
    The energy barrier in LocalTriggerFineLayerFull::barriers is reached with a displacement
    LocalTriggerFineLayerFull::delta_u =
    ``p`` * LocalTriggerFineLayerFull::u_p + ``s`` * LocalTriggerFineLayerFull::u_s.
    This function returns the value of ``s``.
    Shape of output: [LocalTriggerFineLayerFull::nelem_elas, LocalTriggerFineLayerFull::nip].
    Function reads from memory, all computations are done in the construction and
    LocalTriggerFineLayerFull::setState (or one of its approximations).

    \return Simple shear contribution.
    */
    const array_type::tensor<double, 2>& s() const
    {
        return m_smin;
    }

    /**
    Simple shear mode for all integration points of the triggered element, for all elements.
    The output is thus::

        dgamma(e, q) = Eps_s(plastic(e), q).

    \return Shape [System::m_elem_plas, System::m_nip].
    */
    const array_type::tensor<double, 2>& dgamma() const
    {
        return m_dgamma;
    }

    /**
    Pure shear mode for all integration points of the triggered element, for all elements.
    The output is thus::

        dE(e, q) = Deviatoric(Eps_p)(plastic(e), q).

    \return Shape [System::m_elem_plas, System::m_nip].
    */
    const array_type::tensor<double, 2>& dE() const
    {
        return m_dE;
    }

    /**
    The energy barrier in LocalTriggerFineLayerFull::barriers is reached with this
    displacement field.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(e).
    Function reads from memory, all computations are done in the construction and
    LocalTriggerFineLayerFull::setState (or one of its approximations).

    \param e Index of the plastic element.
    \param q Index of the integration point.
    \return Nodal displacement. Shape [System::m_nelem, System::m_nip].
    */
    array_type::tensor<double, 2> delta_u(size_t e, size_t q) const
    {
        return m_pmin(e, q) * this->u_p(e) + m_smin(e, q) * this->u_s(e);
    }

protected:
    size_t m_nip; ///< Number of integration points.
    size_t m_nelem_plas; ///< Number of plastic elements.
    array_type::tensor<size_t, 1> m_elem_plas; ///< Plastic elements.

    /**
    Perturbation for each plastic element.
    The idea is to store/compute the minimal number of perturbations as possible,
    and use a periodic "roll" to reconstruct the perturbations everywhere.
    Because of the construction of the "FineLayer"-mesh, one roll of the mesh will not
    correspond to one roll of the middle layer, therefore a few percolations are needed.
    */
    std::vector<array_type::tensor<double, 2>>
        m_u_s; ///< Disp. field for simple shear perturbation.
    std::vector<array_type::tensor<double, 2>>
        m_u_p; ///< Displacement field for pure shear perturbation.
    std::vector<array_type::tensor<double, 4>>
        m_Eps_s; ///< Strain field for simple shear perturbation.
    std::vector<array_type::tensor<double, 4>>
        m_Eps_p; ///< Strain field for pure shear perturbation.
    std::vector<array_type::tensor<double, 4>>
        m_Sig_s; ///< Stress field for simple shear perturbation.
    std::vector<array_type::tensor<double, 4>>
        m_Sig_p; ///< Stress field for pure shear perturbation.
    std::vector<array_type::tensor<size_t, 1>> m_nodemap; ///< Node-map for the roll.
    std::vector<array_type::tensor<size_t, 1>> m_elemmap; ///< Element-map for the roll.

    array_type::tensor<double, 2> m_dV; ///< Integration point volume.
    double m_V; ///< Volume of a plastic element: assumed homogeneous!
    std::array<size_t, 4> m_shape_T2; ///< Shape of an integration point tensor.

    array_type::tensor<double, 2> m_smin; ///< value of "s" at minimal work "W" [nip, N]
    array_type::tensor<double, 2> m_pmin; ///< value of "p" at minimal work "W" [nip, N]
    array_type::tensor<double, 2> m_Wmin; ///< value of minimal work "W" [nip, N]

    /**
    Strain change in the element for each plastic element::

        == Eps_s(plastic(e), q, 0, 1) [nip, N]
    */
    array_type::tensor<double, 2> m_dgamma;
    array_type::tensor<double, 2> m_dE; ///< == Eps_p(plastic(e), q, 0, 0) [nip, N]
};

/**
Similar to LocalTriggerFineLayerFull, with the difference that only a (small) group of elements
around the triggered element is considered to compute the energy barriers.
The should speed-up the evaluation of the energy barriers significantly.
*/
class LocalTriggerFineLayer : public LocalTriggerFineLayerFull {
public:
    LocalTriggerFineLayer() = default;

    /**
    Constructor.

    \param sys
        System.

    \param roi
        Edge size of the square box encapsulating the triggered element.
        See GooseFEM::Mesh::Quad4::FineLayer::elementgrid_around_ravel.
    */
    LocalTriggerFineLayer(const System& sys, size_t roi = 5)
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

    /**
    Select values in the region of interest around a plastic element.

    \param arg Integration point scalar.
    \param e Index of the plastic element.
    \return Slice of ``arg`` for the region of interest around ``e``.
    */
    array_type::tensor<double, 2>
    slice(const array_type::tensor<double, 2>& arg, size_t e) const override
    {
        return xt::view(arg, xt::keep(m_elemslice[e]));
    }

    /**
    Select values in the region of interest around a plastic element.

    \param arg Integration point tensor.
    \param e Index of the plastic element.
    \return Slice of ``arg`` for the region of interest around ``e``.
    */
    array_type::tensor<double, 4>
    slice(const array_type::tensor<double, 4>& arg, size_t e) const override
    {
        return xt::view(arg, xt::keep(m_elemslice[e]));
    }

    array_type::tensor<double, 4> Eps_s(size_t trigger_plastic) const override
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        return m_Eps_s_slice[trigger_plastic];
    }

    array_type::tensor<double, 4> Eps_p(size_t trigger_plastic) const override
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        return m_Eps_p_slice[trigger_plastic];
    }

    array_type::tensor<double, 4> Sig_s(size_t trigger_plastic) const override
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        return m_Sig_s_slice[trigger_plastic];
    }

    array_type::tensor<double, 4> Sig_p(size_t trigger_plastic) const override
    {
        FRICTIONQPOTFEM_ASSERT(trigger_plastic < m_nelem_plas);
        return m_Sig_p_slice[trigger_plastic];
    }

protected:
    std::vector<array_type::tensor<size_t, 1>>
        m_elemslice; ///< Region-of-interest (ROI) per plastic element.
    std::vector<array_type::tensor<double, 4>>
        m_Eps_s_slice; ///< LocalTriggerFineLayerFull::m_Eps_s for the ROI only.
    std::vector<array_type::tensor<double, 4>>
        m_Eps_p_slice; ///< LocalTriggerFineLayerFull::m_Eps_p for the ROI only.
    std::vector<array_type::tensor<double, 4>>
        m_Sig_s_slice; ///< LocalTriggerFineLayerFull::m_Sig_s for the ROI only.
    std::vector<array_type::tensor<double, 4>>
        m_Sig_p_slice; ///< LocalTriggerFineLayerFull::m_Sig_p for the ROI only.
};

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#endif
