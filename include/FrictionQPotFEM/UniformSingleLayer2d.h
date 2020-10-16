/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM

*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_H

#include "config.h"

#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <GooseFEM/Matrix.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xset_operation.hpp>

namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

// -------------------------------------------------------
// Use GMatElastoPlasticQPot to evaluate stress everywhere
// -------------------------------------------------------

class System {

public:

    System() = default;

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

    // Set nodal displacements.
    void setU(const xt::xtensor<double, 2>& u);

    // Set nodal velocities and accelerations equal to zero.
    // Call this function after an energy minimisation (taken care of in "minimise").
    void quench();

    // Extract elastic/plastic elements.
    auto elastic() const;
    auto plastic() const;

    // Extract nodal quantities.
    auto coor() const; // coordinates
    auto u() const; // displacements
    auto fmaterial() const; // material resistance

    // Extract residual (internal forces normalised by the external forces).
    double residual() const;

    // Extract current time.
    double t() const;

    // Extract integration volume.
    auto dV() const;

    // Convert "qtensor" to "qscalar" (see GooseFEM).
    template <size_t rank, class T> auto AsTensor(const T& arg) const;

    // Extract stress and strain tensors.
    auto Sig() const;
    auto Eps() const;

    // Extract for the plastic elements only.
    virtual xt::xtensor<double, 4> plastic_Sig() const; // stress tensor
    virtual xt::xtensor<double, 4> plastic_Eps() const; // strain tensor
    virtual xt::xtensor<double, 2> plastic_CurrentYieldLeft() const; // yield strain 'left'
    virtual xt::xtensor<double, 2> plastic_CurrentYieldRight() const; // yield strain 'right'
    virtual xt::xtensor<size_t, 2> plastic_CurrentIndex() const; // current index in the landscape

    // Make a time-step.
    void timeStep();

    // Minimise energy:
    // Returns the number of iterations.
    size_t minimise(double tol = 1e-5, size_t niter_tol = 20, size_t max_iter = 100000);

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

    // Initialise geometry (called by constructor).
    void initGeometry(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    // Function to unify the implementations of "setMassMatrix" and "setDampingMatrix".
    template <class T>
    void setMatrix(T& matrix, const xt::xtensor<double, 1>& val_elem);

    // Check the material definition and initialise strain.
    void initMaterial();

    // Check if all prerequisites are satisfied.
    void evalAllSet();

    // Compute strain and stress tensors.
    void computeStress();

    // Evaluate "m_fmaterial". Calls "computeStress".
    // Internal rule: "computeForceMaterial" is always evaluated after an update of "m_u".
    virtual void computeForceMaterial();

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

    // Extract stress and strain.
    // Note: involves re-evaluating the stress and strain,
    // as they are only known in the plastic elements.
    auto Sig();
    auto Eps();

    // Extract for the plastic elements only.
    xt::xtensor<double, 4> plastic_Sig() const override; // stress tensor
    xt::xtensor<double, 4> plastic_Eps() const override; // strain tensor
    xt::xtensor<double, 2> plastic_CurrentYieldLeft() const override; // yield strain 'left'
    xt::xtensor<double, 2> plastic_CurrentYieldRight() const override; // yield strain 'right'
    xt::xtensor<size_t, 2> plastic_CurrentIndex() const override; // current index in the landscape

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

    // Evaluate "m_fmaterial": computes strain and stress in the plastic elements only.
    // Contrary to "System::computeForceMaterial" does not call "computeStress",
    // therefore separate overrides of "Sig" and "Eps" are needed.
    void computeForceMaterial() override;

};

// ----------------
// Event drive step
// ----------------

inline void addEventDrivenShear(System& sys, double deps, bool kick);

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#include "UniformSingleLayer2d.hpp"

#endif