/**
See FrictionQPotFEM::UniformSingleLayer2d.
Implementation in UniformSingleLayer2d.hpp.

\file UniformSingleLayer2d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H

#include "config.h"
#include "version.h"
#include "Generic2d.h"

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
class System : public Generic2d::HybridSystem {

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
    Elastic energy of each integration point.
    Note: this function is put here by convenience as ``this.material().Energy()`` gave problems.

    \return Integration point scalar. Shape: ``[nelem, nip]``.
    */
    [[ deprecated ]]
    virtual xt::xtensor<double, 2> Energy();

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
    \copydoc addSimpleShearToFixedStress(double, bool)

    \throw Throws if yielding is triggered before the stress was reached.
    */
    double addElasticSimpleShearToFixedStress(
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

    /**
    Get the sign of the equivalent strain increment upon a displacement perturbation,
    for each integration point of each plastic element.

    \param perturbation xy-component of the deformation gradient to use to perturb.
    \return Sign of the perturbation. Shape: [System::m_nelem, System::m_nip].
    */
    auto plastic_signOfSimpleShearPerturbation(double perturbation);

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
        const L& elem_plastic);

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
    std::vector<xt::xtensor<size_t, 1>> m_nodemap; ///< Node-map for the roll.
    std::vector<xt::xtensor<size_t, 1>> m_elemmap; ///< Element-map for the roll.

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
