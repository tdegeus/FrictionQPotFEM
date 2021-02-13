/**
\file UniformSingleLayer2d.h

\brief
System in 2-d with:

-   A weak, middle, layer.
-   Uniform elasticity.

Implementation in UniformSingleLayer2d.hpp

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H

#include "config.h"

#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GMatTensor/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <GooseFEM/Matrix.h>
#include <GooseFEM/MatrixPartitioned.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xset_operation.hpp>
#include <fmt/core.h>

/**
\return x^2
*/
#define SQR(x) ((x) * (x))

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;
namespace GT = GMatTensor::Cartesian2d;

/**
Return versions of returned libraries.

\return List of strings with version information.
*/
inline std::vector<std::string> versionInfo();

/**
Class that uses GMatElastoPlasticQPot to evaluate stress everywhere
*/
class System {

public:

    System() = default;

    virtual ~System() {};

    /**
    Define the geometry, including boundary conditions and element sets.

    \param coor Nodal coordinates.
    \param conn Connectivity
    \param dofs DOFs per node
    \param iip DOFs whose displacement is fixed
    \param elem_elastic Elastic elements
    \param elem_plastic Plastic elements
    */
    System(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    /**
    Set mass matrix, based on certain density (taken uniform per element).

    \param rho_elem Density per element.
    */
    void setMassMatrix(const xt::xtensor<double, 1>& rho_elem);

    /**
    Set damping matrix, based on certain density (taken uniform per element).

    \param alpha_elem Damping per element.
    */
    void setDampingMatrix(const xt::xtensor<double, 1>& alpha_elem);

    /**
    Set material parameters for the elastic elements
    (taken uniform per element, ordering the same as in the constructor).

    \param K_elem Bulk modulus per element.
    \param G_elem Bulk modulus per element.
    */
    virtual void setElastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem);

    /**
    Set material parameters for the plastic elements
    (taken uniform per element, ordering the same as in the constructor).

    \param K_elem Bulk modulus per element.
    \param G_elem Bulk modulus per element.
    \param epsy_elem Yield history per element.
    */
    virtual void setPlastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem,
        const xt::xtensor<double, 2>& epsy_elem);

    /**
    Check if elasticity is homogeneous.

    \return ``true`` is elasticity is homogeneous (``false`` otherwise).
    */
    bool isHomogeneousElastic() const;

    /**
    Set time step. Using for example in System::timeStep and System::minimise.
    */
    void setDt(double dt);

    /**
    Set nodal displacements.
    Internally, System::computeForceMaterial is called to update forces deriving from
    elasticity (System::m_fmaterial).
    */
    void setU(const xt::xtensor<double, 2>& u);

    /**
    Set nodal velocities.
    */
    void setV(const xt::xtensor<double, 2>& v);

    /**
    Set nodal accelerations.
    */
    void setA(const xt::xtensor<double, 2>& a);

    /**
    Set nodal velocities and accelerations equal to zero.
    Call this function after an energy minimisation (taken care of in System::minimise).
    */
    void quench();

    /**
    List of elastic elements.

    \return List of element numbers.
    */
    auto elastic() const;

    /**
    List of plastic elements.

    \return List of element numbers.
    */
    auto plastic() const;

    /**
    Connectivity.

    \return Connectivity.
    */
    auto conn() const;

    /**
    Nodal coordinates.

    \return Nodal coordinates.
    */
    auto coor() const;

    /**
    DOFs per node.

    \return DOFs per node.
    */
    auto dofs() const;

    /**
    Element height of all elements along the weak layer.

    \return Element height (scalar).
    */
    double plastic_h() const;

    /**
    Integration point volume of all elements along the weak layer.

    \return Integration point volume.
    */
    double plastic_dV() const;

    /**
    Nodal displacements.

    \return Nodal displacements.
    */
    auto u() const;

    /**
    Nodal velocities.

    \return Nodal velocities.
    */
    auto v() const;

    /**
    Nodal accelerations.

    \return Nodal accelerations.
    */
    auto a() const;

    /**
    Force deriving from elasticity.

    \return Force deriving from elasticity.
    */
    auto fmaterial() const;

    /**
    Norm of the relative residual force (the external force at the top/bottom boundaries is
    used for normalisation).

    \return Relative residual.
    */
    double residual() const;

    /**
    Current time.

    \return Current time.
    */
    double t() const;

    /**
    Integration point volume (of each integration point)

    \return Integration point volume (System::m_quad::dV).
    */
    auto dV() const;

    /**
    Elastic stiffness matrix.

    \return Stiffness matrix (System::m_K).
    */
    auto stiffness() const;

    /**
    GooseFEM vector definition.
    Takes care of bookkeeping.

    \return GooseFEM::VectorPartitioned (System::m_vector)
    */
    auto vector() const;

    /**
    GooseFEM quadrature definition.
    Takes case of interpolation, and taking gradient and integrating.

    \return GooseFEM::Element::Quad4::Quadrature (System::m_quad)
    */
    auto quad() const;

    /**
    GMatElastoPlasticQPot Array definition.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (System::m_material).
    */
    auto material() const;

    /**
    Stress tensor of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    virtual xt::xtensor<double, 4> Sig();

    /**
    Strain tensor of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    virtual xt::xtensor<double, 4> Eps();

    /**
    Stress tensor of integration points of plastic elements only, see System::plastic.

    \return Integration point tensor. Shape: ``[m_plastic.size(), nip, 2, 2]``.
    */
    virtual xt::xtensor<double, 4> plastic_Sig() const;

    /**
    Strain tensor of integration points of plastic elements only, see System::plastic.

    \return Integration point tensor. Shape: ``[m_plastic.size(), nip, 2, 2]``.
    */
    virtual xt::xtensor<double, 4> plastic_Eps() const;

    /**
    Current yield strain left (in the negative equivalent strain direction).

    \return Integration point scalar. Shape: ``[m_plastic.size(), nip]``.
    */
    virtual xt::xtensor<double, 2> plastic_CurrentYieldLeft() const;

    /**
    Current yield strain right (in the positive equivalent strain direction).

    \return Integration point scalar. Shape: ``[m_plastic.size(), nip]``.
    */
    virtual xt::xtensor<double, 2> plastic_CurrentYieldRight() const;

    /**
    Yield strain at an offset to the current yield strain left
    (in the negative equivalent strain direction).
    If ``offset = 0`` the result is the same result as the basic System::plastic_CurrentYieldLeft.

    \param offset Offset (number of yield strains).
    \return Integration point scalar. Shape: ``[m_plastic.size(), nip]``.
    */
    virtual xt::xtensor<double, 2> plastic_CurrentYieldLeft(size_t offset) const;

    /**
    Yield strain at an offset to the current yield strain right
    (in the positive equivalent strain direction).
    If ``offset = 0`` the result is the same result as the basic System::plastic_CurrentYieldRight.

    \param offset Offset (number of yield strains).
    \return Integration point scalar. Shape: ``[m_plastic.size(), nip]``.
    */
    virtual xt::xtensor<double, 2> plastic_CurrentYieldRight(size_t offset) const;

    /**
    Current index in the landscape.

    \return Integration point scalar. Shape: ``[m_plastic.size(), nip]``.
    */
    virtual xt::xtensor<size_t, 2> plastic_CurrentIndex() const;

    /**
    Plastic strain.

    \return Integration point scalar. Shape: ``[m_plastic.size(), nip]``.
    */
    virtual xt::xtensor<double, 2> plastic_Epsp() const;

    /**
    Make a time-step: apply velocity-Verlet integration.
    Forces are computed where needed as follows:

    -   After updating the displacement System::computeForceMaterial is evaluated to update
        System::m_fmaterial.

    -   After update the velocity, System::m_fdamp is computed directly using the
        damping matrix System::m_D
    */
    void timeStep();

    /**
    Minimise energy: run System::timeStep until a mechanical equilibrium has been reached.

    \param tol
        Relative force tolerance for equilibrium. See System::residual for definition.

    \param niter_tol
        Enforce the residual check for ``niter_tol`` consecutive increments.

    \param max_iter
        Maximum number of iterations. Throws ``std::runtime_error`` otherwise.

    \return The number of iterations.
    */
    size_t minimise(double tol = 1e-5, size_t niter_tol = 20, size_t max_iter = 1000000);

    /**
    Get the sign of the equivalent strain increment upon a displacement perturbation,
    for each integration point of each plastic element.

    \param delta_u displacement perturbation.
    \return Integration point scalar. Shape: ``[m_plastic.size(), nip]``.
    */
    auto plastic_signOfPerturbation(const xt::xtensor<double, 2>& delta_u);

    /**
    Add affine simple shear (may be negative to subtract affine simple shear).
    The displacement of the bottom boundary is zero, while it is maximal for the top boundary.
    The input is the strain value, the shear component of deformation gradient is twice that.

    \param delta_gamma Affine strain to add.
    \return xy-component of the deformation gradient that is applied to the system.
    */
    double addAffineSimpleShear(double delta_gamma);

    /**
    Symmetrised version of System::addAffineSimpleShear.
    Similar to System::addAffineSimpleShear with the difference that the displacement is zero
    exactly in the middle, while the displacement at the bottom and the top boundary is maximal
    (with a negative displacement for the bottom boundary).

    \param delta_gamma Affine strain to add.
    \return xy-component of the deformation gradient that is applied to the system.
    */
    double addAffineSimpleShearCentered(double delta_gamma);

    /**
    Add event driven simple shear step.

    \param deps
        Size of the local stain kick to apply.

    \param kick
        If ``false``, increment displacements to ``deps / 2`` of yielding again.
        If ``true``, increment displacements by a affine simple shear increment ``deps``.

    \param direction
        If ``+1``: apply shear to the right. If ``-1`` applying shear to the left.

    \param dry_run
        If ``true`` do not apply displacement, do not check.

    \return
        xy-component of the deformation gradient that is applied to the system.
    */
    double addSimpleShearEventDriven(
        double deps,
        bool kick,
        double direction = +1.0,
        bool dry_run = false);

    /**
    Add simple shear until a target equivalent macroscopic stress has been reached.
    Depending of the target stress compared to the current equivalent macroscopic stress,
    the shear can be either to the left or to the right.
    Throws if yielding is triggered before the stress was reached.

    \param stress
        Target stress (equivalent deviatoric value of the macroscopic stress tensor).

    \param dry_run
        If ``true`` do not apply displacement, do not check.

    \return
        xy-component of the deformation gradient that is applied to the system.
    */
    double addSimpleShearToFixedStress(
        double stress,
        bool dry_run = false);

    /**
    Apply local strain to the right to a specific plastic element.
    This 'triggers' one element while keeping the boundary conditions unchanged.
    Note that by applying shear to the element, yielding can also be triggered in
    the surrounding elements.

    \param deps
        Size of the local stain kick to apply.

    \param plastic_element
        Which plastic element to trigger: System::plastic()(plastic_element).

    \param trigger_weakest_integration_point
        If ``true``, trigger the weakest integration point.
        If ``false``, trigger the strongest.

    \param amplify
        Amplify the strain kick with a certain factor.

    \return
        xy-component of the deformation gradient that is applied to the system.
    */
    double triggerElementWithLocalSimpleShear(
        double deps,
        size_t plastic_element,
        bool trigger_weakest_integration_point = true,
        double amplify = 1.0);

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
    xt::xtensor<double, 2> plastic_ElementYieldBarrierForSimpleShear(
        double deps_kick = 0.0,
        size_t iquad = 0);

protected:

    xt::xtensor<size_t, 2> m_conn; ///< Connectivity.
    xt::xtensor<double, 2> m_coor; ///< Nodal coordinates.
    xt::xtensor<size_t, 2> m_dofs; ///< DOFs.
    xt::xtensor<size_t, 1> m_iip; ///< Fixed DOFs.
    size_t m_N; ///< Number of plastic elements. Alias of System::nelem_plas.
    size_t m_nelem; ///< Number of element.
    size_t m_nelem_elas; ///< Number of elastic elements.
    size_t m_nelem_plas; ///< Number of plastic elements.
    size_t m_nne; ///< Number of nodes per element.
    size_t m_nnode; ///< Number of nodes.
    size_t m_ndim; ///< Number of spatial dimensions.
    size_t m_nip; ///< Number of integration points.
    xt::xtensor<size_t, 1> m_elem_elas; ///< Elastic elements.
    xt::xtensor<size_t, 1> m_elem_plas; ///< Plastic elements.
    QD::Quadrature m_quad; ///< Numerical quadrature.
    GF::VectorPartitioned m_vector; ///< Convert vectors between 'nodevec', 'elemvec', ...
    GF::MatrixDiagonalPartitioned m_M; ///< Mass matrix (diagonal)
    GF::MatrixDiagonal m_D; ///< Damping matrix (diagonal)
    GM::Array<2> m_material; ///< Material definition.
    xt::xtensor<double, 2> m_u; ///< Nodal displacements.
    xt::xtensor<double, 2> m_v; ///< Nodal velocities.
    xt::xtensor<double, 2> m_a; ///< Nodal accelerations.
    xt::xtensor<double, 2> m_v_n; ///< Nodal velocities last time-step.
    xt::xtensor<double, 2> m_a_n; ///< Nodal accelerations last time-step.
    xt::xtensor<double, 3> m_ue; ///< Element vector (used for displacements).
    xt::xtensor<double, 3> m_fe; ///< Element vector (used for displacements).
    xt::xtensor<double, 2> m_fmaterial; ///< Nodal force, deriving from elasticity.
    xt::xtensor<double, 2> m_fdamp; ///< Nodal force, deriving from damping.
    xt::xtensor<double, 2> m_fint; ///< Nodal force: total internal force.
    xt::xtensor<double, 2> m_fext; ///< Nodal force: total external force (reaction force)
    xt::xtensor<double, 2> m_fres; ///< Nodal force: residual force.
    xt::xtensor<double, 4> m_Eps; ///< Integration point tensor: strain.
    xt::xtensor<double, 4> m_Sig; ///< Integration point tensor: stress.
    GF::MatrixPartitioned m_K; ///< Stiffness matrix.
    GF::MatrixPartitionedSolver<> m_solve; ///< Solver to solve ``m_K \ m_fres``
    double m_t = 0.0; ///< Current time.
    double m_dt = 0.0; ///< Time-step.
    bool m_allset = false; ///< Internal allocation check.
    bool m_set_M = false; ///< Internal allocation check: mass matrix was written.
    bool m_set_D = false; ///< Internal allocation check: damping matrix was written.
    bool m_set_elas = false; ///< Internal allocation check: elastic elements were written.
    bool m_set_plas = false; ///< Internal allocation check: plastic elements were written.

protected:

    // Constructor alias
    void initSystem(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    // Check the material definition and initialise strain.
    void initMaterial();

    // Check if all prerequisites are satisfied.
    void evalAllSet();

    // Compute strain and stress tensors.
    void computeStress();

    /**
    Update System::m_fmaterial based on the current displacement field System::m_u.
    This implies taking the gradient of the stress tensor, System::m_Sig,
    computed using System::computeStress.

    Internal rule: System::computeForceMaterial is always evaluated after an update of System::m_u.
    This is taken care off by calling System::setU, and never updating System::m_u directly.
    */
    virtual void computeForceMaterial();

    // Get the sign of the equivalent strain increment upon a displacement perturbation,
    // for each integration point of each plastic element.
    auto plastic_signOfSimpleShearPerturbation(double perturbation);

};

// -----------------------------------------------------------------------
// Use speed-up by evaluating the elastic response with a stiffness matrix
// -----------------------------------------------------------------------

class HybridSystem : public System {

public:

    HybridSystem() = default;

    HybridSystem(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    // Set material parameters for the elastic elements
    // (taken uniform per element, ordering the same as in the constructor).
    void setElastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem) override;

    // Set material parameters for the plastic elements
    // (taken uniform per element, ordering the same as in the constructor).
    void setPlastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem,
        const xt::xtensor<double, 2>& epsy_elem) override;

    // Get the underlying "GMatElastoPlasticQPot::Array<2>"
    auto material_elastic() const;
    auto material_plastic() const;

    // Extract stress and strain.
    // Note: involves re-evaluating the stress and strain,
    // as they are only known in the plastic elements.
    xt::xtensor<double, 4> Sig() override;
    xt::xtensor<double, 4> Eps() override;

    // Extract for the plastic elements only.
    xt::xtensor<double, 4> plastic_Sig() const override; // stress tensor
    xt::xtensor<double, 4> plastic_Eps() const override; // strain tensor
    xt::xtensor<double, 2> plastic_CurrentYieldLeft() const override; // yield strain 'left'
    xt::xtensor<double, 2> plastic_CurrentYieldRight() const override; // yield strain 'right'
    xt::xtensor<double, 2> plastic_CurrentYieldLeft(size_t offset) const override;
    xt::xtensor<double, 2> plastic_CurrentYieldRight(size_t offset) const override;
    xt::xtensor<size_t, 2> plastic_CurrentIndex() const override; // current index in the landscape
    xt::xtensor<double, 2> plastic_Epsp() const override; // plastic strain

protected:

    // mesh parameters
    xt::xtensor<size_t, 2> m_conn_elas;
    xt::xtensor<size_t, 2> m_conn_plas;

    // numerical quadrature
    QD::Quadrature m_quad_elas;
    QD::Quadrature m_quad_plas;

    // convert vectors between 'nodevec', 'elemvec', ...
    GF::VectorPartitioned m_vector_elas;
    GF::VectorPartitioned m_vector_plas;

    // material definition
    GM::Array<2> m_material_elas;
    GM::Array<2> m_material_plas;

    // element vectors
    xt::xtensor<double, 3> m_ue_plas;
    xt::xtensor<double, 3> m_fe_plas;

    // nodal forces
    xt::xtensor<double, 2> m_felas;
    xt::xtensor<double, 2> m_fplas;

    // integration point tensors
    xt::xtensor<double, 4> m_Eps_elas;
    xt::xtensor<double, 4> m_Eps_plas;
    xt::xtensor<double, 4> m_Sig_elas;
    xt::xtensor<double, 4> m_Sig_plas;

    // stiffness matrix
    GF::Matrix m_K_elas;

    // keep track of need to recompute
    bool m_eval_full = true;

protected:

    // Constructor alias
    void initHybridSystem(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    // Evaluate "m_fmaterial": computes strain and stress in the plastic elements only.
    // Contrary to "System::computeForceMaterial" does not call "computeStress",
    // therefore separate overrides of "Sig" and "Eps" are needed.
    void computeForceMaterial() override;

};

// -------------------------------------------------
// Trigger by simple shear + pure shear perturbation
// -------------------------------------------------

class LocalTriggerFineLayerFull
{
public:
    LocalTriggerFineLayerFull() = default;
    LocalTriggerFineLayerFull(const System& sys);
    virtual ~LocalTriggerFineLayerFull() {};

    // set state, compute energy barriers for all integration points,
    // discretise the yield-surface in "ntest"-steps
    void setState(
        const xt::xtensor<double, 4>& Eps,
        const xt::xtensor<double, 4>& Sig,
        const xt::xtensor<double, 2>& epsy,
        size_t ntest = 100);

    // set state, compute energy barriers for all integration points,
    // discretise the yield-surface using a minimal number of tests
    void setStateMinimalSearch(
        const xt::xtensor<double, 4>& Eps,
        const xt::xtensor<double, 4>& Sig,
        const xt::xtensor<double, 2>& epsy);

    // set state, compute energy barriers for simple shear perturbation
    void setStateSimpleShear(
        const xt::xtensor<double, 4>& Eps,
        const xt::xtensor<double, 4>& Sig,
        const xt::xtensor<double, 2>& epsy);

    // return all energy barriers [nelem_elas, nip], as energy density
    xt::xtensor<double, 2> barriers() const;
    xt::xtensor<double, 2> p() const; // correspond "p"
    xt::xtensor<double, 2> s() const; // correspond "s"

    // return the displacement corresponding to the energy barrier
    xt::xtensor<double, 2> delta_u(size_t plastic_element, size_t q) const;

    // return perturbation
    xt::xtensor<double, 2> u_s(size_t plastic_element) const;
    xt::xtensor<double, 2> u_p(size_t plastic_element) const;
    virtual xt::xtensor<double, 4> Eps_s(size_t plastic_element) const;
    virtual xt::xtensor<double, 4> Eps_p(size_t plastic_element) const;
    virtual xt::xtensor<double, 4> Sig_s(size_t plastic_element) const;
    virtual xt::xtensor<double, 4> Sig_p(size_t plastic_element) const;
    xt::xtensor<double, 2> dgamma() const;
    xt::xtensor<double, 2> dE() const;

    // remap quantities (does nothing here, but used by derived class)
    virtual xt::xtensor<double, 2> slice(const xt::xtensor<double, 2>& arg, size_t e) const;
    virtual xt::xtensor<double, 4> slice(const xt::xtensor<double, 4>& arg, size_t e) const;

protected:
    void computePerturbation(
        size_t plastic_element,
        const xt::xtensor<double, 2>& sig_star, // stress perturbation at "plastic_element"
        xt::xtensor<double, 2>& u,      // output equilibrium displacement
        xt::xtensor<double, 4>& Eps,    // output equilibrium strain`
        xt::xtensor<double, 4>& Sig,    // output equilibrium stress
        GF::MatrixPartitioned& K,
        GF::MatrixPartitionedSolver<>& solver,
        const QD::Quadrature& quad,
        const GF::VectorPartitioned& vector,
        GM::Array<2>& material);

protected:
    // Basic info.
    size_t m_nip;
    size_t m_nelem_plas;
    xt::xtensor<size_t, 1> m_elem_plas;

    // Perturbation for each plastic element.
    // The idea is to store/compute the minimal number of perturbations as possible,
    // and use a periodic "roll" to reconstruct the perturbations everywhere.
    // Because of the construction of the "FineLayer"-mesh, one roll of the mesh will not
    // correspond to one roll of the middle layer, therefore a few percolations are needed.
    std::vector<xt::xtensor<double, 2>> m_u_s;
    std::vector<xt::xtensor<double, 2>> m_u_p;
    std::vector<xt::xtensor<double, 4>> m_Eps_s;
    std::vector<xt::xtensor<double, 4>> m_Eps_p;
    std::vector<xt::xtensor<double, 4>> m_Sig_s;
    std::vector<xt::xtensor<double, 4>> m_Sig_p;
    std::vector<xt::xtensor<double, 1>> m_nodemap;
    std::vector<xt::xtensor<double, 1>> m_elemmap;

    // Integration point values.
    xt::xtensor<double, 2> m_dV;
    double m_V;
    std::array<size_t, 4> m_shape_T2;

    // Perturbation of minimal work.
    xt::xtensor<double, 2> m_smin; // value of "s" at minimal work "W" [nip, N]
    xt::xtensor<double, 2> m_pmin; // value of "p" at minimal work "W" [nip, N]
    xt::xtensor<double, 2> m_Wmin; // value of minimal work "W" [nip, N]

    // Strain change in the element for each plastic element.
    xt::xtensor<double, 2> m_dgamma; // == Eps_s(plastic(e), q, 0, 1) [nip, N]
    xt::xtensor<double, 2> m_dE; // == Eps_p(plastic(e), q, 0, 0) [nip, N]
};

// -------------------------------------------------
// Trigger by simple shear + pure shear perturbation
// -------------------------------------------------

class LocalTriggerFineLayer : public LocalTriggerFineLayerFull
{
public:
    LocalTriggerFineLayer() = default;
    LocalTriggerFineLayer(const System& sys, size_t region_of_interest = 5);

    xt::xtensor<double, 4> Eps_s(size_t plastic_element) const override;
    xt::xtensor<double, 4> Eps_p(size_t plastic_element) const override;
    xt::xtensor<double, 4> Sig_s(size_t plastic_element) const override;
    xt::xtensor<double, 4> Sig_p(size_t plastic_element) const override;

    xt::xtensor<double, 2> slice(const xt::xtensor<double, 2>& arg, size_t e) const override;
    xt::xtensor<double, 4> slice(const xt::xtensor<double, 4>& arg, size_t e) const override;

protected:
    std::vector<xt::xtensor<size_t, 1>> m_elemslice;

    std::vector<xt::xtensor<double, 4>> m_Eps_s_slice;
    std::vector<xt::xtensor<double, 4>> m_Eps_p_slice;
    std::vector<xt::xtensor<double, 4>> m_Sig_s_slice;
    std::vector<xt::xtensor<double, 4>> m_Sig_p_slice;
};

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#include "UniformSingleLayer2d.hpp"

#endif
