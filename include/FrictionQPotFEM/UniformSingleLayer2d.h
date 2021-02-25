/**
System in 2-d with:

-   A weak, middle, layer.
-   Uniform elasticity.

Implementation in UniformSingleLayer2d.hpp

\file UniformSingleLayer2d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H

#include "config.h"
#include "version.h"

#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GMatTensor/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <GooseFEM/Matrix.h>
#include <GooseFEM/MatrixPartitioned.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xset_operation.hpp>

/**
\return x^2
*/
#define SQR(x) ((x) * (x))

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

/**
Return versions of this library and of all of its dependencies.
The output is a list of strings, e.g.::

    "frictionqpotfem=0.7.1",
    "goosefem=0.7.0",
    ...

\return List of strings.
*/
inline std::vector<std::string> version_dependencies();

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
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem_elastic Elastic elements.
    \param elem_plastic Plastic elements.
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
    Internally, this updates System::m_fmaterial by calling System::computeForceMaterial.
    */
    void setU(const xt::xtensor<double, 2>& u);

    /**
    Set nodal velocities.
    Internally, this updates System::m_fdamp.
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
    List of elastic elements. Shape: [System::m_nelem_elas].

    \return List of element numbers.
    */
    auto elastic() const;

    /**
    List of plastic elements. Shape: [System::m_nelem_plas].

    \return List of element numbers.
    */
    auto plastic() const;

    /**
    Connectivity. Shape: [System::m_nelem, System::m_nne].

    \return Connectivity.
    */
    auto conn() const;

    /**
    Nodal coordinates. Shape: [System::m_nnode, System::m_ndim].

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
    Mass matrix, see System::m_M.

    \return Mass matrix (diagonal, partitioned).
    */
    auto mass() const;

    /**
    Damping matrix, see System::m_D.

    \return Damping matrix (diagonal).
    */
    auto damping() const;

    /**
    Force deriving from elasticity.

    \return Nodal force. Shape ``[nnode, ndim]``    .
    */
    auto fmaterial() const;

    /**
    Force deriving from damping.

    \return Nodal force. Shape ``[nnode, ndim]``    .
    */
    auto fdamp() const;

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
    Elastic energy of each integration point.
    Note: this function is put here by convenience as ``this.material().Energy()`` gave problems.

    \return Integration point scalar. Shape: ``[nelem, nip]``.
    */
    virtual xt::xtensor<double, 2> Energy();

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

    xt::xtensor<size_t, 2> m_conn; ///< Connectivity. See System::conn.
    xt::xtensor<double, 2> m_coor; ///< Nodal coordinates. See System::coor.
    xt::xtensor<size_t, 2> m_dofs; ///< DOFs. Shape: [System::m_nnode, System::m_ndim].
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
    GooseFEM::Element::Quad4::Quadrature m_quad; ///< Numerical quadrature.
    GooseFEM::VectorPartitioned m_vector; ///< Convert vectors between 'nodevec', 'elemvec', ...
    GooseFEM::MatrixDiagonalPartitioned m_M; ///< Mass matrix (diagonal)
    GooseFEM::MatrixDiagonal m_D; ///< Damping matrix (diagonal)
    GMatElastoPlasticQPot::Cartesian2d::Array<2> m_material; ///< Material definition.
    xt::xtensor<double, 2> m_u; ///< Nodal displacements.
    xt::xtensor<double, 2> m_v; ///< Nodal velocities.
    xt::xtensor<double, 2> m_a; ///< Nodal accelerations.
    xt::xtensor<double, 2> m_v_n; ///< Nodal velocities last time-step.
    xt::xtensor<double, 2> m_a_n; ///< Nodal accelerations last time-step.
    xt::xtensor<double, 3> m_ue; ///< Element vector (used for displacements).
    xt::xtensor<double, 3> m_fe; ///< Element vector (used for forces).
    xt::xtensor<double, 2> m_fmaterial; ///< Nodal force, deriving from elasticity.
    xt::xtensor<double, 2> m_fdamp; ///< Nodal force, deriving from damping.
    xt::xtensor<double, 2> m_fint; ///< Nodal force: total internal force.
    xt::xtensor<double, 2> m_fext; ///< Nodal force: total external force (reaction force)
    xt::xtensor<double, 2> m_fres; ///< Nodal force: residual force.
    xt::xtensor<double, 4> m_Eps; ///< Integration point tensor: strain.
    xt::xtensor<double, 4> m_Sig; ///< Integration point tensor: stress.
    GooseFEM::MatrixPartitioned m_K; ///< Stiffness matrix.
    GooseFEM::MatrixPartitionedSolver<> m_solve; ///< Solver to solve ``m_K \ m_fres``
    double m_t = 0.0; ///< Current time.
    double m_dt = 0.0; ///< Time-step.
    bool m_allset = false; ///< Internal allocation check.
    bool m_set_M = false; ///< Internal allocation check: mass matrix was written.
    bool m_set_D = false; ///< Internal allocation check: damping matrix was written.
    bool m_set_elas = false; ///< Internal allocation check: elastic elements were written.
    bool m_set_plas = false; ///< Internal allocation check: plastic elements were written.

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
    void initSystem(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    /**
    If all material points are specified: initialise strain and set stiffness matrix.
    */
    void initMaterial();

    /**
    Set System::m_allset = ``true`` if all prerequisites are satisfied.
    */
    void evalAllSet();

    /**
    Compute strain and stress tensors.
    Uses System::m_u to update System::m_Sig and System::m_Eps.
    */
    void computeStress();

    /**
    Update System::m_fmaterial based on the current displacement field System::m_u.
    This implies taking the gradient of the stress tensor, System::m_Sig,
    computed using System::computeStress.

    Internal rule: This function is always evaluated after an update of System::m_u.
    This is taken care off by calling System::setU, and never updating System::m_u directly.
    */
    virtual void computeForceMaterial();

    /**
    Get the sign of the equivalent strain increment upon a displacement perturbation,
    for each integration point of each plastic element.

    \param perturbation xy-component of the deformation gradient to use to perturb.
    \return Sign of the perturbation. Shape: [System::m_nelem, System::m_nip].
    */
    auto plastic_signOfSimpleShearPerturbation(double perturbation);

};

/**
Same functionality as System, but with potential speed-ups.
This class may be faster in most cases, as internally the elastic and plastic
elements are separated.
The force deriving from elasticity, System::m_fmaterial, is now computed as
HybridSystem::m_material_elas + HybridSystem::m_material_plas.

Thereby:
-   Elastic: forces, HybridSystem::m_material_elas, are evaluated using a stiffness matrix.
    This avoids the evaluation of stress and strain.
    This of course implies that their evaluation requires them to be computed.
    Thus HybridSystem::Sig and HybridSystem::Eps require a computation, whereas
    System::Sig and System::Eps are for free.
-   Plastic: methods as HybridSystem::plastic_Sig, HybridSystem::plastic_Epsp, etc.
    do not require slicing (which is needed in System::plastic_Sig, etc.).
*/
class HybridSystem : public System {

public:

    HybridSystem() = default;

    /**
    Define the geometry, including boundary conditions and element sets.

    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem_elastic Elastic elements.
    \param elem_plastic Plastic elements.
    */
    HybridSystem(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    void setElastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem) override;

    void setPlastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem,
        const xt::xtensor<double, 2>& epsy_elem) override;

    /**
    GMatElastoPlasticQPot Array definition for the elastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (System::m_material_elas).
    */
    auto material_elastic() const;

    /**
    GMatElastoPlasticQPot Array definition for the plastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (System::m_material_plas).
    */
    auto material_plastic() const;

    /**
    Elastic energy of each integration point.
    Note: this function is put here by convenience as ``this.material().Energy()`` gave problems.
    Note: involves re-evaluating the stress and strain,
    as they are only known in the plastic elements.
    No re-evaluation is needed if the method is class directly after HybridSystem::Eps
    or HybridSystem::Sig.

    \return Integration point scalar. Shape: ``[nelem, nip]``.
    */
    xt::xtensor<double, 2> Energy() override;

    /**
    Stress tensor of each integration point.
    Note: involves re-evaluating the stress and strain,
    as they are only known in the plastic elements.
    No re-evaluation is needed if the method is class directly after HybridSystem::Eps.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    xt::xtensor<double, 4> Sig() override;

    /**
    Strain tensor of each integration point.
    Note: involves re-evaluating the stress and strain,
    as they are only known in the plastic elements.
    No re-evaluation is needed if the method is class directly after HybridSystem::Sig.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    xt::xtensor<double, 4> Eps() override;

    xt::xtensor<double, 4> plastic_Sig() const override;
    xt::xtensor<double, 4> plastic_Eps() const override;
    xt::xtensor<double, 2> plastic_CurrentYieldLeft() const override;
    xt::xtensor<double, 2> plastic_CurrentYieldRight() const override;
    xt::xtensor<double, 2> plastic_CurrentYieldLeft(size_t offset) const override;
    xt::xtensor<double, 2> plastic_CurrentYieldRight(size_t offset) const override;
    xt::xtensor<size_t, 2> plastic_CurrentIndex() const override;
    xt::xtensor<double, 2> plastic_Epsp() const override;

protected:

    xt::xtensor<size_t, 2> m_conn_elas; ///< Slice of System::m_conn for elastic elements.
    xt::xtensor<size_t, 2> m_conn_plas; ///< Slice of System::m_conn for plastic elements.
    GooseFEM::Element::Quad4::Quadrature m_quad_elas; ///< Numerical quadrature for elastic elements.
    GooseFEM::Element::Quad4::Quadrature m_quad_plas; ///< Numerical quadrature for plastic elements.
    GooseFEM::VectorPartitioned m_vector_elas; ///< Convert vectors for elastic elements.
    GooseFEM::VectorPartitioned m_vector_plas; ///< Convert vectors for plastic elements.
    GMatElastoPlasticQPot::Cartesian2d::Array<2> m_material_elas; ///< Material definition for elastic elements.
    GMatElastoPlasticQPot::Cartesian2d::Array<2> m_material_plas; ///< Material definition for plastic elements.
    xt::xtensor<double, 3> m_ue_plas; ///< Element vector for elastic elements (used for displacements).
    xt::xtensor<double, 3> m_fe_plas; ///< Element vector for plastic elements (used for forces).
    xt::xtensor<double, 2> m_felas; ///< Nodal force, deriving from elasticity of elastic elements.
    xt::xtensor<double, 2> m_fplas; ///< Nodal force, deriving from elasticity of plastic elements.
    xt::xtensor<double, 4> m_Eps_elas; ///< Integration point tensor: strain for elastic elements.
    xt::xtensor<double, 4> m_Eps_plas; ///< Integration point tensor: strain for plastic elements.
    xt::xtensor<double, 4> m_Sig_elas; ///< Integration point tensor: stress for elastic elements.
    xt::xtensor<double, 4> m_Sig_plas; ///< Integration point tensor: stress for plastic elements.
    GooseFEM::Matrix m_K_elas; ///< Stiffness matrix for elastic elements.
    bool m_eval_full = true; ///< Keep track of the need to recompute full variables as System::m_Sig.

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
    void initHybridSystem(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    /**
    Update System::m_fmaterial based on the current displacement field System::m_u.
    Contrary to System::computeForceMaterial does not call HybridSystem::computeStress,
    therefore separate computation (and overrides) of
    HybridSystem::Sig and HybridSystem::Eps are needed.

    Internal rule: This function is always evaluated after an update of System::m_u.
    This is taken care off by calling System::setU, and never updating System::m_u directly.
    */
    void computeForceMaterial() override;

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
class LocalTriggerFineLayerFull
{
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
    LocalTriggerFineLayerFull(const System& sys);

    virtual ~LocalTriggerFineLayerFull() {};

    /**
    Set current state and compute energy barriers to reach the specified yield surface
    (for all plastic elements).
    The yield surface is discretised in ``ntest`` steps.

    \param Eps Integration point strain, see System::Eps.
    \param Sig Integration point stress, see System::Sig.
    \param epsy Next yield strains, see System::plastic_CurrentYieldRight.
    \param ntest Number of steps in which to discretise the yield surface.
    */
    void setState(
        const xt::xtensor<double, 4>& Eps,
        const xt::xtensor<double, 4>& Sig,
        const xt::xtensor<double, 2>& epsy,
        size_t ntest = 100);

    /**
    Set current state and compute energy barriers to reach the specified yield surface
    (for all plastic elements).
    The yield surface is discretised in only ``8`` steps.

    \param Eps Integration point strain, see System::Eps.
    \param Sig Integration point stress, see System::Sig.
    \param epsy Next yield strains, see System::plastic_CurrentYieldRight.
    */
    void setStateMinimalSearch(
        const xt::xtensor<double, 4>& Eps,
        const xt::xtensor<double, 4>& Sig,
        const xt::xtensor<double, 2>& epsy);

    /**
    Set current state and compute energy barriers to reach the specified yield surface,
    for a purely simple shear perturbation (for all plastic elements)

    \param Eps Integration point strain, see System::Eps.
    \param Sig Integration point stress, see System::Sig.
    \param epsy Next yield strains, see System::plastic_CurrentYieldRight.
    */
    void setStateSimpleShear(
        const xt::xtensor<double, 4>& Eps,
        const xt::xtensor<double, 4>& Sig,
        const xt::xtensor<double, 2>& epsy);

    /**
    Get all energy barriers, as energy density.
    Shape of output: [LocalTriggerFineLayerFull::nelem_elas, LocalTriggerFineLayerFull::nip].
    Function reads from memory, all computations are done in the construction and
    LocalTriggerFineLayerFull::setState (or one of its approximations).

    \return Energy barriers.
    */
    xt::xtensor<double, 2> barriers() const;

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
    xt::xtensor<double, 2> p() const;

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
    xt::xtensor<double, 2> s() const;

    /**
    The energy barrier in LocalTriggerFineLayerFull::barriers is reached with this
    displacement field.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(plastic_element).
    Function reads from memory, all computations are done in the construction and
    LocalTriggerFineLayerFull::setState (or one of its approximations).

    \param plastic_element Index of the plastic element.
    \param q Index of the integration point.
    \return Nodal displacement. Shape [System::m_nelem, System::m_nip].
    */
    xt::xtensor<double, 2> delta_u(size_t plastic_element, size_t q) const;

    /**
    Displacement field for the simple shear eigen-stress applied to a specific element.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(plastic_element).
    Function reads from memory, all computations are done in the constructor.

    \param plastic_element Index of the plastic element.
    \return Nodal displacement. Shape [System::m_nelem, System::m_nip].
    */
    xt::xtensor<double, 2> u_s(size_t plastic_element) const;

    /**
    Displacement field for the pure shear eigen-stress applied to a specific element.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(plastic_element).
    Function reads from memory, all computations are done in the constructor.

    \param plastic_element Index of the plastic element.
    \return Nodal displacement. Shape [System::m_nelem, System::m_nip].
    */
    xt::xtensor<double, 2> u_p(size_t plastic_element) const;

    /**
    Integration point strain tensors for LocalTriggerFineLayerFull::u_s.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(plastic_element).
    Function reads from memory, all computations are done in the constructor.

    \param plastic_element Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual xt::xtensor<double, 4> Eps_s(size_t plastic_element) const;

    /**
    Integration point strain tensors for LocalTriggerFineLayerFull::u_p.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(plastic_element).
    Function reads from memory, all computations are done in the constructor.

    \param plastic_element Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual xt::xtensor<double, 4> Eps_p(size_t plastic_element) const;

    /**
    Integration point stress tensors for LocalTriggerFineLayerFull::u_s.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(plastic_element).
    Function reads from memory, all computations are done in the constructor.

    \param plastic_element Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual xt::xtensor<double, 4> Sig_s(size_t plastic_element) const;

    /**
    Integration point stress tensors for LocalTriggerFineLayerFull::u_p.
    This function takes the index of the plastic element; the real element number
    is obtained by LocalTriggerFineLayerFull::m_elem_plas(plastic_element).
    Function reads from memory, all computations are done in the constructor.

    \param plastic_element Index of the plastic element.
    \return Integration point tensor. Shape [System::m_nelem, System::m_nip, 2, 2].
    */
    virtual xt::xtensor<double, 4> Sig_p(size_t plastic_element) const;

    /**
    Simple shear mode for all integration points of the triggered element, for all elements.
    The output is thus::

        dgamma(e, q) = Eps_s(plastic(e), q).

    \return Shape [System::m_elem_plas, System::m_nip].
    */
    xt::xtensor<double, 2> dgamma() const;

    /**
    Pure shear mode for all integration points of the triggered element, for all elements.
    The output is thus::

        dE(e, q) = Deviatoric(Eps_p)(plastic(e), q).

    \return Shape [System::m_elem_plas, System::m_nip].
    */
    xt::xtensor<double, 2> dE() const;

    /**
    Empty function, used by LocalTriggerFineLayer.

    \param arg Integration point scalar.
    \param e Index of the plastic element.
    \return A copy of ``arg``.
    */
    virtual xt::xtensor<double, 2> slice(const xt::xtensor<double, 2>& arg, size_t e) const;

    /**
    Empty function, used by LocalTriggerFineLayer.

    \param arg Integration point tensor.
    \param e Index of the plastic element.
    \return A copy of ``arg``.
    */
    virtual xt::xtensor<double, 4> slice(const xt::xtensor<double, 4>& arg, size_t e) const;

protected:

    /**
    Compute the displacement response to an eigen-stress applied to a plastic element.

    \param plastic_element Index of the plastic element.
    \param sig_star Eigen-stress applied to ``plastic_element``.
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
        size_t plastic_element,
        const xt::xtensor<double, 2>& sig_star,
        xt::xtensor<double, 2>& u,
        xt::xtensor<double, 4>& Eps,
        xt::xtensor<double, 4>& Sig,
        GooseFEM::MatrixPartitioned& K,
        GooseFEM::MatrixPartitionedSolver<>& solver,
        const GooseFEM::Element::Quad4::Quadrature& quad,
        const GooseFEM::VectorPartitioned& vector,
        GMatElastoPlasticQPot::Cartesian2d::Array<2>& material);

protected:

    size_t m_nip; ///< Number of integration points.
    size_t m_nelem_plas; ///< Number of plastic elements.
    xt::xtensor<size_t, 1> m_elem_plas; ///< Plastic elements.

    /**
    Perturbation for each plastic element.
    The idea is to store/compute the minimal number of perturbations as possible,
    and use a periodic "roll" to reconstruct the perturbations everywhere.
    Because of the construction of the "FineLayer"-mesh, one roll of the mesh will not
    correspond to one roll of the middle layer, therefore a few percolations are needed.
    */
    std::vector<xt::xtensor<double, 2>> m_u_s; ///< Displacement field for simple shear perturbation.
    std::vector<xt::xtensor<double, 2>> m_u_p; ///< Displacement field for pure shear perturbation.
    std::vector<xt::xtensor<double, 4>> m_Eps_s; ///< Strain field for simple shear perturbation.
    std::vector<xt::xtensor<double, 4>> m_Eps_p; ///< Strain field for pure shear perturbation.
    std::vector<xt::xtensor<double, 4>> m_Sig_s; ///< Stress field for simple shear perturbation.
    std::vector<xt::xtensor<double, 4>> m_Sig_p; ///< Stress field for pure shear perturbation.
    std::vector<xt::xtensor<double, 1>> m_nodemap; ///< Node-map for the roll.
    std::vector<xt::xtensor<double, 1>> m_elemmap; ///< Element-map for the roll.

    xt::xtensor<double, 2> m_dV; ///< Integration point volume.
    double m_V; ///< Volume of a plastic element: assumed homogeneous!
    std::array<size_t, 4> m_shape_T2; ///< Shape of an integration point tensor.

    xt::xtensor<double, 2> m_smin; ///< value of "s" at minimal work "W" [nip, N]
    xt::xtensor<double, 2> m_pmin; ///< value of "p" at minimal work "W" [nip, N]
    xt::xtensor<double, 2> m_Wmin; ///< value of minimal work "W" [nip, N]

    /**
    Strain change in the element for each plastic element::

        == Eps_s(plastic(e), q, 0, 1) [nip, N]
    */
    xt::xtensor<double, 2> m_dgamma;
    xt::xtensor<double, 2> m_dE; ///< == Eps_p(plastic(e), q, 0, 0) [nip, N]
};

/**
Similar to LocalTriggerFineLayerFull, with the difference that only a (small) group of elements
around the triggered element is considered to compute the energy barriers.
The should speed-up the evaluation of the energy barriers significantly.
*/
class LocalTriggerFineLayer : public LocalTriggerFineLayerFull
{
public:

    LocalTriggerFineLayer() = default;

    /**
    Constructor.

    \param sys
        System.

    \param region_of_interest
        Edge size of the square box encapsulating the triggered element.
        See GooseFEM::Mesh::Quad4::FineLayer::elementgrid_around_ravel.
    */
    LocalTriggerFineLayer(const System& sys, size_t region_of_interest = 5);

    xt::xtensor<double, 4> Eps_s(size_t plastic_element) const override;
    xt::xtensor<double, 4> Eps_p(size_t plastic_element) const override;
    xt::xtensor<double, 4> Sig_s(size_t plastic_element) const override;
    xt::xtensor<double, 4> Sig_p(size_t plastic_element) const override;

    /**
    Select values in the region of interest around a plastic element.

    \param arg Integration point scalar.
    \param e Index of the plastic element.
    \return Slice of ``arg`` for the region of interest around ``e``.
    */
    xt::xtensor<double, 2> slice(const xt::xtensor<double, 2>& arg, size_t e) const override;

    /**
    Select values in the region of interest around a plastic element.

    \param arg Integration point tensor.
    \param e Index of the plastic element.
    \return Slice of ``arg`` for the region of interest around ``e``.
    */
    xt::xtensor<double, 4> slice(const xt::xtensor<double, 4>& arg, size_t e) const override;

protected:
    std::vector<xt::xtensor<size_t, 1>> m_elemslice; ///< Region-of-interest per plastic element.
    std::vector<xt::xtensor<double, 4>> m_Eps_s_slice; ///< LocalTriggerFineLayerFull::m_Eps_s for the ROI only.
    std::vector<xt::xtensor<double, 4>> m_Eps_p_slice; ///< LocalTriggerFineLayerFull::m_Eps_p for the ROI only.
    std::vector<xt::xtensor<double, 4>> m_Sig_s_slice; ///< LocalTriggerFineLayerFull::m_Sig_s for the ROI only.
    std::vector<xt::xtensor<double, 4>> m_Sig_p_slice; ///< LocalTriggerFineLayerFull::m_Sig_p for the ROI only.
};

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#include "UniformSingleLayer2d.hpp"

#endif
