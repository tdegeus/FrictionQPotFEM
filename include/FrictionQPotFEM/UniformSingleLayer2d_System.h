/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM

*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_SYSTEM_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_SYSTEM_H

#include "config.h"

#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GooseFEM/GooseFEM.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xset_operation.hpp>

namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

class System {

public:

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
    void setElastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem);

    // Set material parameters for the plastic elements
    // (taken uniform per element, ordering the same as in the constructor).
    void setPlastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem,
        const xt::xtensor<double, 2>& epsy_elem);

    // Set/overwrite various parameters
    void setDt(double dt); // time step
    void setU(const xt::xtensor<double, 2>& u); // nodal displacements
    void quench(); // set "v" and "a" equal to zero

    // Extract variables.
    auto nelem() const;
    auto forceMaterial() const;
    double residual() const;
    double t() const;
    auto u() const;
    auto dV() const;
    template <size_t rank, class T> auto AsTensor(const T& arg) const;

    // Compute strain and stress (everywhere).
    void computeStress();

    // Make a time-step.
    void timeStep();

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

private:

    template <class T>
    void setMatrix(T& matrix, const xt::xtensor<double, 1>& val_elem);

    void initMaterial();

    void evalAllSet();

};

} // namespace UniformSingleLayer2d
} // namespace FrictionQPotFEM

#include "UniformSingleLayer2d_System.hpp"

#endif
