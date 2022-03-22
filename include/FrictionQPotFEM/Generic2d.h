/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_GENERIC2D_H
#define FRICTIONQPOTFEM_GENERIC2D_H

#include "config.h"
#include "version.h"

#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GMatElastoPlasticQPot/version.h>
#include <GMatTensor/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <GooseFEM/Matrix.h>
#include <GooseFEM/MatrixPartitioned.h>
#include <algorithm>
#include <string>
#include <xtensor/xnorm.hpp>
#include <xtensor/xset_operation.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>

namespace FrictionQPotFEM {

/**
Generic system of elastic and plastic elements.
For the moment this not part of the public API and can be subjected to changes.
*/
namespace Generic2d {

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
System with elastic elements and plastic elements (GMatElastoPlasticQPot).

For efficiency, the nodal forces for the elastic elements are evaluated using the tangent.
This means that getting stresses and strains in those elements is not for free.
Therefore, there are separate methods to get stresses and strains only in the plastic elements
(as they are readily available as they are needed for the force computation).
*/
class System {

public:
    System() = default;

    virtual ~System(){};

    /**
    Define the geometry, including boundary conditions and element sets.

    \tparam C Type of nodal coordinates, e.g. `xt::xtensor<double, 2>`
    \tparam E Type of connectivity and DOFs, e.g. `xt::xtensor<size_t, 2>`
    \tparam L Type of node/element lists, e.g. `xt::xtensor<size_t, 1>`
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
        const L& elem_plastic);

    /**
    Return the linear system size (in number of blocks).
    */
    virtual size_t N() const;

    /**
    Return the type of system.
    */
    virtual std::string type() const;

    /**
    Set mass matrix, based on certain density (taken uniform per element).

    \tparam T e.g. `xt::xtensor<double, 1>`.
    \param rho_elem Density per element.
    */
    template <class T>
    void setMassMatrix(const T& rho_elem);

    /**
    Set the value of the damping at the interface.
    Note that you can specify either setEta() or setDampingMatrix() or both.

    \param eta Damping parameter
    */
    void setEta(double eta);

    /**
    Set damping matrix, based on certain density (taken uniform per element).
    Note that you can specify either setEta() or setDampingMatrix() or both.

    \param alpha_elem Damping per element.
    */
    template <class T>
    void setDampingMatrix(const T& alpha_elem);

    /**
    Set material parameters for the elastic elements
    (taken uniform per element, ordering the same as in the constructor).

    \param K_elem Bulk modulus per element.
    \param G_elem Bulk modulus per element.
    */
    virtual void
    setElastic(const xt::xtensor<double, 1>& K_elem, const xt::xtensor<double, 1>& G_elem);

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
    Get the current yield strains per plastic element.
    Note that in this system the yield strains history is always the same for all the integration
    points in the system.
    \return [plastic().size, n]
    */
    xt::xtensor<double, 2> epsy() const;

    /**
    Reset yield strains (to avoid re-construction).
    \param epsy_elem Yield history per element.
    */
    virtual void reset_epsy(const xt::xtensor<double, 2>& epsy_elem);

    /**
    Check if elasticity is homogeneous.

    \return ``true`` is elasticity is homogeneous (``false`` otherwise).
    */
    bool isHomogeneousElastic() const;

    /**
    Set time.
    */
    void setT(double t);

    /**
    Set time step. Using for example in System::timeStep and System::minimise.
    */
    void setDt(double dt);

    /**
    Set nodal displacements.
    Internally, this updates the relevant forces using updated_u().

    \param u ``[nnode, ndim]``.
    */
    template <class T>
    void setU(const T& u);

    /**
    Set nodal velocities.
    Internally, this updates the relevant forces using updated_v().

    \param v ``[nnode, ndim]``.
    */
    template <class T>
    void setV(const T& v);

    /**
    Set nodal accelerations.

    \param a ``[nnode, ndim]``.
    */
    template <class T>
    void setA(const T& a);

    /**
    Set external force.
    Note: the external force on the DOFs whose displacement are prescribed are response forces
    computed during timeStep(). Internally on the system of unknown DOFs is solved, so any
    change to the response forces is ignored.

    \param fext ``[nnode, ndim]``.
    */
    template <class T>
    void setFext(const T& fext);

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
    auto& mass() const;

    /**
    Damping matrix, see System::m_D.

    \return Damping matrix (diagonal).
    */
    auto& damping() const;

    /**
    External force.
    Note: the external force on the DOFs whose displacement are prescribed are response forces
    computed during timeStep().

    \return Nodal force. Shape ``[nnode, ndim]``    .
    */
    auto fext() const;

    /**
    Internal force.
    Note: computed during timeStep().

    \return Nodal force. Shape ``[nnode, ndim]``    .
    */
    auto fint() const;

    /**
    Force deriving from elasticity.

    \return Nodal force. Shape ``[nnode, ndim]``    .
    */
    auto fmaterial() const;

    /**
    Force deriving from damping.
    This force is the sum of damping at the interface plus that of background damping
    (or just one of both if just one is specified).

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
    GooseFEM vector definition.
    Takes care of bookkeeping.

    \return GooseFEM::VectorPartitioned (System::m_vector)
    */
    const GooseFEM::VectorPartitioned& vector() const;

    /**
    GooseFEM quadrature definition.
    Takes case of interpolation, and taking gradient and integrating.

    \return GooseFEM::Element::Quad4::Quadrature (System::m_quad)
    */
    const GooseFEM::Element::Quad4::Quadrature& quad() const;

    /**
    GMatElastoPlasticQPot Array definition for the elastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (#m_material_elas).
    */
    const GMatElastoPlasticQPot::Cartesian2d::Array<2>& material_elastic() const;

    /**
    GMatElastoPlasticQPot Array definition for the plastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (#m_material_plas).
    */
    const GMatElastoPlasticQPot::Cartesian2d::Array<2>& material_plastic() const;

    /**
    Bulk modulus per integration point.

    \return Integration point scalar. Shape: ``[nelem, nip]``.
    */
    xt::xtensor<double, 2> K() const;

    /**
    Shear modulus per integration point.

    \return Integration point scalar. Shape: ``[nelem, nip]``.
    */
    xt::xtensor<double, 2> G() const;

    /**
    Stress tensor of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    xt::xtensor<double, 4> Sig();

    /**
    Strain tensor of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    xt::xtensor<double, 4> Eps();

    /**
    Stiffness tensor of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2, 2, 2]``.
    */
    GooseFEM::MatrixPartitioned stiffness() const;

    /**
    Stress tensor of integration points of plastic elements only, see System::plastic.

    \return Integration point tensor. Shape: [plastic().size(), nip, 2, 2].
    */
    xt::xtensor<double, 4> plastic_Sig() const;

    /**
    Strain tensor of integration points of plastic elements only, see System::plastic.

    \return Integration point tensor. Shape: [plastic().size(), nip, 2, 2].
    */
    xt::xtensor<double, 4> plastic_Eps() const;

    /**
    Strain tensor of of a specific plastic element.

    \param e_plastic Plastic element (real element number = plastic()[e]).
    \param q Integration point (real element number = plastic()[e]).
    \return Integration point tensor. Shape: [2, 2].
    */
    xt::xtensor<double, 2> plastic_Eps(size_t e_plastic, size_t q) const;

    /**
    Current yield strain left (in the negative equivalent strain direction).

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    xt::xtensor<double, 2> plastic_CurrentYieldLeft() const;

    /**
    Current yield strain right (in the positive equivalent strain direction).

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    xt::xtensor<double, 2> plastic_CurrentYieldRight() const;

    /**
    Yield strain at an offset to the current yield strain left
    (in the negative equivalent strain direction).
    If ``offset = 0`` the result is the same result as the basic System::plastic_CurrentYieldLeft.

    \param offset Offset (number of yield strains).
    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    xt::xtensor<double, 2> plastic_CurrentYieldLeft(size_t offset) const;

    /**
    Yield strain at an offset to the current yield strain right
    (in the positive equivalent strain direction).
    If ``offset = 0`` the result is the same result as the basic System::plastic_CurrentYieldRight.

    \param offset Offset (number of yield strains).
    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    xt::xtensor<double, 2> plastic_CurrentYieldRight(size_t offset) const;

    /**
    Current index in the landscape.

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    xt::xtensor<size_t, 2> plastic_CurrentIndex() const;

    /**
    Plastic strain.

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    xt::xtensor<double, 2> plastic_Epsp() const;

    /**
    Check that the current yield-index is at least `n` away from the end.
    \param n Margin.
    \return `true` if the current yield-index is at least `n` away from the end.
    */
    bool boundcheck_right(size_t n) const;

    /**
    Set purely elastic and linear response to specific boundary conditions.
    Since this response is linear and elastic it can be scaled freely to transverse a
    fully elastic interval at once, without running any dynamics,
    and run an event driven code using eventDrivenStep().

    \param delta_u Nodal displacement field.
    \param autoscale Scale such that the perturbation is small enough.
    \return Value with which the input perturbation is scaled, see also eventDriven_deltaU().
    */
    template <class T>
    double eventDriven_setDeltaU(const T& delta_u, bool autoscale = true);

    /**
    Get displacement perturbation used for event driven code, see eventDriven_setDeltaU().
    \return Nodal displacements.
    */
    auto eventDriven_deltaU() const;

    /**
    Add event driven step for the boundary conditions that correspond to the displacement
    perturbation set in eventDriven_setDeltaU().

    \todo
        The current implementation is suited for unloading, but only until the first yield strain.
        An improved implementation is needed to deal with the symmetry of the equivalent strain.
        A `FRICTIONQPOTFEM_WIP` assertion is made.

    \param deps
        Size of the local stain kick to apply.

    \param kick
        If ``false``, increment displacements to ``deps / 2`` of yielding again.
        If ``true``, increment displacements by a affine simple shear increment ``deps``.

    \param direction
        If ``+1``: apply shear to the right. If ``-1`` applying shear to the left.

    \param yield_element
        If `true` and `kick == true` then the element closest to yielding is selected (as normal),
        but of that element the displacement update is the maximum of the element, such that all
        integration points of the element are forced to yield.

    \param iterative
        If `true` the step is iteratively searched.
        This is more costly by recommended, if the perturbation is non-analytical
        (and contains numerical errors).

    \return
        Factor with which the displacement perturbation, see eventDriven_deltaU(), is scaled.
    */
    virtual double eventDrivenStep(
        double deps,
        bool kick,
        int direction = 1,
        bool yield_element = false,
        bool iterative = false);

    /**
    Make a time-step: apply velocity-Verlet integration.
    Forces are computed where needed using:
    updated_u(), updated_v(), and computeInternalExternalResidualForce().
    */
    void timeStep();

    /**
    Make a number of time steps.
    \param n Number of steps to make.
    */
    void timeSteps(size_t n);

    /**
    Make a number of time steps (or stop early if mechanical equilibrium was reached).

    \param n (Maximum) Number of steps to make.
    \param tol Relative force tolerance for equilibrium. See System::residual for definition.
    \param niter_tol Enforce the residual check for ``niter_tol`` consecutive increments.

    \return
        -   Number of iterations: `== n`
        -   `0`: if stopped when the residual is reached
            (and the number of iterations was ``< n``).
    */
    size_t timeSteps_residualcheck(size_t n, double tol = 1e-5, size_t niter_tol = 20);

    /**
    \copydoc timeSteps(size_t)

    This function stops if the yield-index in any of the plastic elements is close the end.
    In that case the function returns zero, in all other cases the function returns a
    positive number.

    \param nmargin Number of potentials to leave as margin.
    */
    size_t timeSteps_boundcheck(size_t n, size_t nmargin = 5);

    /**
    Make a number of steps with the following protocol.
    (1) Add a displacement \f$ \underline{v} \Delta t \f$ to each of the nodes.
    (2) Make a timeStep().

    \param n Number of steps to make.
    \param v Nodal velocity to add ``[nnode, ndim]``.
    */
    template <class T>
    void flowSteps(size_t n, const T& v);

    /**
    \copydoc flowSteps(size_t, const T&)

    This function stops if the yield-index in any of the plastic elements is close the end.
    In that case the function returns zero, in all other cases the function returns a
    positive number.

    \param nmargin
        Number of potentials to leave as margin.
    */
    template <class T>
    size_t flowSteps_boundcheck(size_t n, const T& v, size_t nmargin = 5);

    /**
    Perform a series of time-steps until:
    -   the next plastic event,
    -   equilibrium, or
    -   a maximum number of iterations.

    \param tol Relative force tolerance for equilibrium. See System::residual for definition.
    \param niter_tol Enforce the residual check for ``niter_tol`` consecutive increments.
    \param max_iter Maximum number of iterations.

    \return
        -   *Number of iterations*:
            If stopped by a plastic event or the maximum number of iterations.
        -   `0`:
            If stopped when the residual is reached
            (so no plastic event occurred and the number of iterations was lower than the maximum).
    */
    size_t
    timeStepsUntilEvent(double tol = 1e-5, size_t niter_tol = 20, size_t max_iter = 10000000);

    /**
    Minimise energy: run System::timeStep until a mechanical equilibrium has been reached.

    \param tol Relative force tolerance for equilibrium. See System::residual for definition.
    \param niter_tol Enforce the residual check for ``niter_tol`` consecutive increments.
    \param max_iter Maximum number of iterations. Throws ``std::runtime_error`` otherwise.

    \return The number of iterations.
    */
    size_t minimise(double tol = 1e-5, size_t niter_tol = 20, size_t max_iter = 10000000);

    /**
    \copydoc System::minimise(double, size_t, size_t)

    This function stops if the yield-index in any of the plastic elements is close the end.
    In that case the function returns zero (in all other cases it returns a positive number).

    \param nmargin
        Number of potentials to leave as margin.
    */
    size_t minimise_boundcheck(
        size_t nmargin = 5,
        double tol = 1e-5,
        size_t niter_tol = 20,
        size_t max_iter = 10000000);

    /**
    \copydoc System::minimise(double, size_t, size_t)

    This function stops when a certain number of blocks has yielded at least once.
    In that case the function returns zero (in all other cases it returns a positive number).

    \note ``A_truncate`` and ``S_truncate`` are defined on the first integration point.

    \param A_truncate
        Truncate if ``A_truncate`` blocks have yielded at least once.

    \param S_truncate
        Truncate if the number of times blocks yielded is equal to ``S_truncate``.
        **Warning** This option is reserved for future use, but for the moment does nothing.
    */
    size_t minimise_truncate(
        size_t A_truncate = 0,
        size_t S_truncate = 0,
        double tol = 1e-5,
        size_t niter_tol = 20,
        size_t max_iter = 10000000);

    /**
    \copydoc System::minimise_truncate(size_t, size_t, double, size_t, size_t)

    This overload can be used to specify a reference state when manually triggering.
    If triggering is done before calling this function, already one block yielded,
    making `A_truncate` and `S_truncate` inaccurate.

    \param idx_n Reference potential index of the first integration point.
    */
    template <class T>
    size_t minimise_truncate(
        const T& idx_n,
        size_t A_truncate = 0,
        size_t S_truncate = 0,
        double tol = 1e-5,
        size_t niter_tol = 20,
        size_t max_iter = 10000000);

    /**
    Get the displacement field that corresponds to an affine simple shear of a certain strain.
    The displacement of the bottom boundary is zero, while it is maximal for the top boundary.

    \param delta_gamma Strain to add (the shear component of deformation gradient is twice that).
    \return Nodal displacements.
    */
    xt::xtensor<double, 2> affineSimpleShear(double delta_gamma) const;

    /**
    Get the displacement field that corresponds to an affine simple shear of a certain strain.
    Similar to affineSimpleShear() with the difference that the displacement is zero
    exactly in the middle, while the displacement at the bottom and the top boundary is maximal
    (with a negative displacement for the bottom boundary).

    \param delta_gamma Strain to add (the shear component of deformation gradient is twice that).
    \return Nodal displacements.
    */
    xt::xtensor<double, 2> affineSimpleShearCentered(double delta_gamma) const;

protected:
    xt::xtensor<size_t, 2> m_conn; ///< Connectivity, see conn().
    xt::xtensor<size_t, 2> m_conn_elas; ///< Slice of #m_conn for elastic elements.
    xt::xtensor<size_t, 2> m_conn_plas; ///< Slice of #m_conn for plastic elements.
    xt::xtensor<double, 2> m_coor; ///< Nodal coordinates, see coor().
    xt::xtensor<size_t, 2> m_dofs; ///< DOFs, shape: [#m_nnode, #m_ndim].
    xt::xtensor<size_t, 1> m_iip; ///< Fixed DOFs.
    size_t m_N; ///< Number of plastic elements, alias of #m_nelem_plas.
    size_t m_nelem; ///< Number of elements.
    size_t m_nelem_elas; ///< Number of elastic elements.
    size_t m_nelem_plas; ///< Number of plastic elements.
    size_t m_nne; ///< Number of nodes per element.
    size_t m_nnode; ///< Number of nodes.
    size_t m_ndim; ///< Number of spatial dimensions.
    size_t m_nip; ///< Number of integration points.
    xt::xtensor<size_t, 1> m_elem_elas; ///< Elastic elements.
    xt::xtensor<size_t, 1> m_elem_plas; ///< Plastic elements.
    GooseFEM::Element::Quad4::Quadrature m_quad; ///< Quadrature for all elements.
    GooseFEM::Element::Quad4::Quadrature m_quad_elas; ///< #m_quad for elastic elements only.
    GooseFEM::Element::Quad4::Quadrature m_quad_plas; ///< #m_quad for plastic elements only.
    GooseFEM::VectorPartitioned m_vector; ///< Convert vectors between 'nodevec', 'elemvec', ....
    GooseFEM::VectorPartitioned m_vector_elas; ///< #m_vector for elastic elements only.
    GooseFEM::VectorPartitioned m_vector_plas; ///< #m_vector for plastic elements only.
    GooseFEM::MatrixDiagonalPartitioned m_M; ///< Mass matrix (diagonal).
    GooseFEM::MatrixDiagonal m_D; ///< Damping matrix (diagonal).
    GMatElastoPlasticQPot::Cartesian2d::Array<2> m_material_elas; ///< Material for elastic el.
    GMatElastoPlasticQPot::Cartesian2d::Array<2> m_material_plas; ///< Material for plastic el.

    /**
    Nodal displacements.
    \warning To make sure that the right forces are computed at the right time,
    always call updated_u() after manually updating #m_u (setU() automatically take care of this).
    */
    xt::xtensor<double, 2> m_u;

    /**
    Nodal velocities.
    \warning To make sure that the right forces are computed at the right time,
    always call updated_v() after manually updating #m_v (setV() automatically take care of this).
    */
    xt::xtensor<double, 2> m_v;

    xt::xtensor<double, 2> m_a; ///< Nodal accelerations.
    xt::xtensor<double, 2> m_v_n; ///< Nodal velocities last time-step.
    xt::xtensor<double, 2> m_a_n; ///< Nodal accelerations last time-step.
    xt::xtensor<double, 3> m_ue; ///< Element vector (used for displacements).
    xt::xtensor<double, 3> m_fe; ///< Element vector (used for forces).
    xt::xtensor<double, 3> m_ue_elas; ///< El. vector for elastic elements (used for displacements).
    xt::xtensor<double, 3> m_fe_elas; ///< El. vector for plastic elements (used for forces).
    xt::xtensor<double, 3> m_ue_plas; ///< El. vector for elastic elements (used for displacements).
    xt::xtensor<double, 3> m_fe_plas; ///< El. vector for plastic elements (used for forces).
    xt::xtensor<double, 2> m_fmaterial; ///< Nodal force, deriving from elasticity.
    xt::xtensor<double, 2> m_felas; ///< Nodal force, deriving from elasticity of elastic elements.
    xt::xtensor<double, 2> m_fplas; ///< Nodal force, deriving from elasticity of plastic elements.
    xt::xtensor<double, 2> m_fdamp; ///< Nodal force, deriving from damping.
    xt::xtensor<double, 2> m_fvisco; ///< Nodal force, deriving from damping at the interface
    xt::xtensor<double, 2> m_ftmp; ///< Temporary for internal use.
    xt::xtensor<double, 2> m_fint; ///< Nodal force: total internal force.
    xt::xtensor<double, 2> m_fext; ///< Nodal force: total external force (reaction force)
    xt::xtensor<double, 2> m_fres; ///< Nodal force: residual force.
    xt::xtensor<double, 4> m_Eps; ///< Integration point tensor: strain.
    xt::xtensor<double, 4> m_Sig; ///< Integration point tensor: stress.
    xt::xtensor<double, 4> m_Eps_elas; ///< Integration point tensor: strain for elastic elements.
    xt::xtensor<double, 4> m_Eps_plas; ///< Integration point tensor: strain for plastic elements.
    xt::xtensor<double, 4> m_Sig_elas; ///< Integration point tensor: stress for elastic elements.
    xt::xtensor<double, 4> m_Sig_plas; ///< Integration point tensor: stress for plastic elements.
    xt::xtensor<double, 4> m_Epsdot_plas; ///< Integration point tensor: strain-rate for plastic el.
    GooseFEM::Matrix m_K_elas; ///< Stiffness matrix for elastic elements only.
    double m_t = 0.0; ///< Current time.
    double m_dt = 0.0; ///< Time-step.
    double m_eta = 0.0; ///< Damping at the interface
    bool m_allset = false; ///< Internal allocation check.
    bool m_set_M = false; ///< Internal allocation check: mass matrix was written.
    bool m_set_D = false; ///< Internal allocation check: damping matrix was written.
    bool m_set_visco = false; ///< Internal allocation check: interfacial damping specified.
    bool m_set_elas = false; ///< Internal allocation check: elastic elements were written.
    bool m_set_plas = false; ///< Internal allocation check: plastic elements were written.
    bool m_eval_full = true; ///< Keep track of the need to recompute full stress/strain.
    xt::xtensor<double, 2> m_pert_u; ///< See eventDriven_setDeltaU()
    xt::xtensor<double, 4> m_pert_Epsd_plastic; ///< Strain deviator for #m_pert_u.

protected:
    /**
    Constructor alias, useful for derived classes.

    \tparam C Type of nodal coordinates, e.g. `xt::xtensor<double, 2>`
    \tparam E Type of connectivity and DOFs, e.g. `xt::xtensor<size_t, 2>`
    \tparam L Type of node/element lists, e.g. `xt::xtensor<size_t, 1>`
    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem_elastic Elastic elements.
    \param elem_plastic Plastic elements.
    */
    template <class C, class E, class L>
    void initSystem(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elem_elastic,
        const L& elem_plastic);

    /**
    Set m_allset = ``true`` if all prerequisites are satisfied.
    */
    void evalAllSet();

    /**
    Compute strain and stress tensors.
    Uses m_u to update m_Sig and m_Eps.
    */
    void computeFullStress();

    /**
    Evaluate relevant forces when m_u is updated.
    */
    virtual void updated_u();

    /**
    Evaluate relevant forces when m_v is updated.
    */
    virtual void updated_v();

    /**
    Update m_fmaterial based on the current displacement field m_u.
    This implies taking the gradient of the stress tensor, m_Sig,
    computed using computeFullStress.

    Internal rule: This function is always evaluated after an update of m_u.
    This is taken care off by calling setU, and never updating m_u directly.
    */
    virtual void computeForceMaterial();

    /**
    Compute:
    -   m_fint = m_fmaterial + m_fdamp
    -   m_fext[iip] = m_fint[iip]
    -   m_fres = m_fext - m_fint

    Internal rule: all relevant forces are expected to be updated before this function is called.
    */
    virtual void computeInternalExternalResidualForce();
};

} // namespace Generic2d
} // namespace FrictionQPotFEM

#include "Generic2d.hpp"

#endif
