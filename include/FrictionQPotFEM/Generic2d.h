/**
Generic system of elastic and plastic elements.
Implementation in Generic2d.hpp.

\file Generic2d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef FRICTIONQPOTFEM_GENERIC2D_H
#define FRICTIONQPOTFEM_GENERIC2D_H

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
Class that uses GMatElastoPlasticQPot to evaluate stress everywhere.
*/
class System {

public:

    System() = default;

    virtual ~System() {};

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
    Set mass matrix, based on certain density (taken uniform per element).

    \tparam T e.g. `xt::xtensor<double, 1>`.
    \param rho_elem Density per element.
    */
    template <class T>
    void setMassMatrix(const T& rho_elem);

    /**
    Set damping matrix, based on certain density (taken uniform per element).

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
    const GooseFEM::MatrixPartitioned& stiffness() const;

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
    GMatElastoPlasticQPot Array definition.

    \return GMatElastoPlasticQPot::Cartesian2d::Array <2> (System::m_material).
    */
    const GMatElastoPlasticQPot::Cartesian2d::Array<2>& material() const;

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
    Forces are computed where needed using:
    updated_u(), updated_v(), and computeInternalExternalResidualForce().
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
    If all material points are specified: initialise strain and set stiffness matrix.
    */
    void initMaterial();

    /**
    Set m_allset = ``true`` if all prerequisites are satisfied.
    */
    void evalAllSet();

    /**
    Compute strain and stress tensors.
    Uses m_u to update m_Sig and m_Eps.
    */
    void computeStress();

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
    computed using computeStress.

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

\warning
As described, some variables of System will not be updated on the fly but have to be evaluated
before use (that is exactly where the speed-up comes from).
The overrides of Sig() and Eps() automatically take care of this and can be called without
any consideration (except of course it involves some computations).
For the following cases, one has to call evalSystem() after updating #u in order to get access
to the current state:
-   material()
*/
class HybridSystem : public System {

public:

    HybridSystem() = default;

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
    HybridSystem(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elem_elastic,
        const L& elem_plastic);

    void setElastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem) override;

    void setPlastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem,
        const xt::xtensor<double, 2>& epsy_elem) override;

    /**
    Evaluate the full System.
    Call this function after setU() if you want to use material().
    Note that in theory the same goes for Sig() and Eps(), but those call evalSystem() internally.
    */
    void evalSystem();

    /**
    GMatElastoPlasticQPot Array definition for the elastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (System::m_material_elas).
    */
    const GMatElastoPlasticQPot::Cartesian2d::Array<2>& material_elastic() const;

    /**
    GMatElastoPlasticQPot Array definition for the plastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (System::m_material_plas).
    */
    const GMatElastoPlasticQPot::Cartesian2d::Array<2>& material_plastic() const;

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
    void initHybridSystem(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elem_elastic,
        const L& elem_plastic);

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

} // namespace Generic2d
} // namespace FrictionQPotFEM

#include "Generic2d.hpp"

#endif
