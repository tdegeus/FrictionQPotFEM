/*

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

namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;
namespace GT = GMatTensor::Cartesian2d;

#define SQR(x) ((x) * (x)) // x^2

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

// -------------------------------------
// Return versions of returned libraries
// -------------------------------------

inline std::vector<std::string> versionInfo();

// -------------------------------------------------------
// Use GMatElastoPlasticQPot to evaluate stress everywhere
// -------------------------------------------------------

class System {

public:

    System() = default;
    virtual ~System() {};

    // Define the geometry, include boundary conditions and element sets.
    System(
        const xt::xtensor<double, 2>& coor, // nodal coordinates
        const xt::xtensor<size_t, 2>& conn, // connectivity
        const xt::xtensor<size_t, 2>& dofs, // DOFs per node
        const xt::xtensor<size_t, 1>& iip,  // DOFs whose displacement is fixed
        const xt::xtensor<size_t, 1>& elem_elastic,  // elastic elements
        const xt::xtensor<size_t, 1>& elem_plastic); // plastic elements

    // Set mass and damping matrix, based on certain density (taken uniform per element).
    void setMassMatrix(const xt::xtensor<double, 1>& rho_elem);
    void setDampingMatrix(const xt::xtensor<double, 1>& alpha_elem);

    // Set material parameters for the elastic elements
    // (taken uniform per element, ordering the same as in the constructor).
    virtual void setElastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem);

    // Set material parameters for the plastic elements
    // (taken uniform per element, ordering the same as in the constructor).
    virtual void setPlastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem,
        const xt::xtensor<double, 2>& epsy_elem);

    // Check if elasticity is homogeneous.
    bool isHomogeneousElastic() const;

    // Set time step.
    void setDt(double dt);

    // Set nodal quantities.
    void setU(const xt::xtensor<double, 2>& u); // displacements
    void setV(const xt::xtensor<double, 2>& v); // velocities
    void setA(const xt::xtensor<double, 2>& a); // accelerations

    // Set nodal velocities and accelerations equal to zero.
    // Call this function after an energy minimisation (taken care of in "minimise").
    void quench();

    // Extract elastic/plastic elements.
    auto elastic() const;
    auto plastic() const;

    // Extract mesh
    auto conn() const; // connectivity
    auto coor() const; // nodal coordinates
    auto dofs() const; // DOFs per node

    // Basic properties of the layer
    double plastic_h() const; // element height
    double plastic_dV() const; // integration point volume

    // Extract nodal quantities.
    auto u() const; // displacements
    auto v() const; // velocities
    auto a() const; // accelerations
    auto fmaterial() const; // material resistance

    // Extract residual (internal forces normalised by the external forces).
    double residual() const;

    // Extract current time.
    double t() const;

    // Extract integration volume.
    auto dV() const;

    // Get the "GooseFEM::VectorPartitioned" and the "GooseFEM::Element::Quad4::Quadrature"
    auto stiffness() const;
    auto vector() const;
    auto quad() const;

    // Get the underlying "GMatElastoPlasticQPot::Array<2>"
    auto material() const;

    // Extract stress and strain tensors.
    virtual xt::xtensor<double, 4> Sig();
    virtual xt::xtensor<double, 4> Eps();

    // Extract for the plastic elements only (per integration point).
    virtual xt::xtensor<double, 4> plastic_Sig() const; // stress tensor
    virtual xt::xtensor<double, 4> plastic_Eps() const; // strain tensor
    virtual xt::xtensor<double, 2> plastic_CurrentYieldLeft() const; // yield strain 'left'
    virtual xt::xtensor<double, 2> plastic_CurrentYieldRight() const; // yield strain 'right'
    virtual xt::xtensor<double, 2> plastic_CurrentYieldLeft(size_t offset) const;
    virtual xt::xtensor<double, 2> plastic_CurrentYieldRight(size_t offset) const;
    virtual xt::xtensor<size_t, 2> plastic_CurrentIndex() const; // current index in the landscape
    virtual xt::xtensor<double, 2> plastic_Epsp() const; // plastic strain

    // Make a time-step: apply velocity-Verlet integration.
    void timeStep();

    // Minimise energy: run "timeStep" until a mechanical equilibrium has been reached.
    // Returns the number of iterations.
    size_t minimise(double tol = 1e-5, size_t niter_tol = 20, size_t max_iter = 1000000);

    // Get the sign of the equivalent strain increment upon a displacement perturbation,
    // for each integration point of each plastic element.
    auto plastic_signOfPerturbation(const xt::xtensor<double, 2>& delta_u);

    // Add affine simple shear (may be negative to subtract affine simple shear).
    // The displacement of the bottom boundary is zero, while it is maximal for the top boundary.
    // The input is the strain value, the shear component deformation gradient is twice that.
    // Return deformation gradient that is applied to the system.
    double addAffineSimpleShear(double delta_gamma);

    // Similar to "addAffineSimpleShear" with the difference that the displacement is zero
    // exactly in the middle, while the displacement at the bottom and the top boundary is maximal
    // (with a negative displacement for the bottom boundary).
    double addAffineSimpleShearCentered(double delta_gamma);

    // Add event driven simple shear step.
    // Return deformation gradient that is applied to the system.
    double addSimpleShearEventDriven(
        double deps, // size of the local stain kick to apply
        bool kick, // "kick = false": increment displacements to "deps / 2" of yielding again
        double direction = +1.0, // "direction = +1": apply shear to the right
        bool dry_run = false); // "dry_run = true": do not apply displacement, do not check

    // Add simple shear until a target equivalent macroscopic stress has been reached.
    // Depending of the target stress compared to the current equivalent macroscopic stress,
    // the shear can be either to the left or to the right.
    // Throws if yielding is triggered before the stress was reached.
    // Return deformation gradient that is applied to the system.
    double addSimpleShearToFixedStress(
        double target_equivalent_macroscopic_stress,
        bool dry_run = false); // "dry_run = true": do not apply displacement, do not check

    // Apply local strain to the right to a specific plastic element.
    // This 'triggers' one element while keeping the boundary conditions unchanged.
    // Note that by applying shear to the element, yielding can also be triggered in
    // the surrounding elements.
    // Return deformation gradient that is applied to the element.
    double triggerElementWithLocalSimpleShear(
        double deps, // size of the local stain kick to apply
        size_t plastic_element, // which plastic element to trigger: sys.plastic()(plastic_element)
        bool trigger_weakest_integration_point = true, // trigger weakest or strongest int. point
        double amplify = 1.0); // amplify the strain kick with a certain factor

    // Read the distance to overcome the first cusp in the element.
    // Returns array of shape (N, 2) with columns (delta_eps, delta_epsxy).
    xt::xtensor<double, 2> plastic_ElementYieldBarrierForSimpleShear(
        double deps_kick = 0.0,
        size_t iquad = 0); // iquad = 0 -> weakest, iquad = nip - 1 -> strongest

protected:

    // mesh parameters
    xt::xtensor<size_t, 2> m_conn;
    xt::xtensor<double, 2> m_coor;
    xt::xtensor<size_t, 2> m_dofs;
    xt::xtensor<size_t, 1> m_iip;

    // mesh dimensions
    size_t m_N; // == nelem_plas
    size_t m_nelem;
    size_t m_nelem_elas;
    size_t m_nelem_plas;
    size_t m_nne;
    size_t m_nnode;
    size_t m_ndim;
    size_t m_nip;

    // element sets
    xt::xtensor<size_t, 1> m_elem_elas;
    xt::xtensor<size_t, 1> m_elem_plas;

    // numerical quadrature
    QD::Quadrature m_quad;

    // convert vectors between 'nodevec', 'elemvec', ...
    GF::VectorPartitioned m_vector;

    // mass matrix
    GF::MatrixDiagonalPartitioned m_M;

    // damping matrix
    GF::MatrixDiagonal m_D;

    // material definition
    GM::Array<2> m_material;

    // nodal displacements, velocities, and accelerations (current and last time-step)
    xt::xtensor<double, 2> m_u;
    xt::xtensor<double, 2> m_v;
    xt::xtensor<double, 2> m_a;
    xt::xtensor<double, 2> m_v_n;
    xt::xtensor<double, 2> m_a_n;

    // element vectors
    xt::xtensor<double, 3> m_ue;
    xt::xtensor<double, 3> m_fe;

    // nodal forces
    xt::xtensor<double, 2> m_fmaterial;
    xt::xtensor<double, 2> m_fdamp;
    xt::xtensor<double, 2> m_fint;
    xt::xtensor<double, 2> m_fext;
    xt::xtensor<double, 2> m_fres;

    // integration point tensors
    xt::xtensor<double, 4> m_Eps;
    xt::xtensor<double, 4> m_Sig;

    // stiffness matrix
    GF::MatrixPartitioned m_K;
    GF::MatrixPartitionedSolver<> m_solve;

    // time
    double m_t = 0.0;
    double m_dt = 0.0;

    // check
    bool m_allset = false;
    bool m_set_M = false;
    bool m_set_D = false;
    bool m_set_elas = false;
    bool m_set_plas = false;

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

    // Evaluate "m_fmaterial". Calls "computeStress".
    // Internal rule: "computeForceMaterial" is always evaluated after an update of "m_u".
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
