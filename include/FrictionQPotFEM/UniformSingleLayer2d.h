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

namespace GF = GooseFEM;
namespace QD = GooseFEM::Element::Quad4;
namespace GM = GMatElastoPlasticQPot::Cartesian2d;

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d {

class HybridSystem {

public:

    HybridSystem() = default;

    void setGeometry(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 1>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const xt::xtensor<size_t, 1>& elem_elastic,
        const xt::xtensor<size_t, 1>& elem_plastic);

    void setMassMatrix(const xt::xtensor<double, 1>& rho_elem);
    void setDampingMatrix(const xt::xtensor<double, 1>& alpha_elem);

    void setElastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem);

    void setPlastic(
        const xt::xtensor<double, 1>& K_elem,
        const xt::xtensor<double, 1>& G_elem,
        const xt::xtensor<double, 2>& epsy_elem);

    void setDt(double dt);

    auto nelem() const;

    void computeStress();
    void computeStressPlastic();

    void timeStep();

private:

    // mesh parameters
    xt::xtensor<size_t, 2> m_conn;
    xt::xtensor<size_t, 2> m_conn_elas;
    xt::xtensor<size_t, 2> m_conn_plas;
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
    QD::Quadrature m_quad_elas;
    QD::Quadrature m_quad_plas;

    // convert vectors between 'nodevec', 'elemvec', ...
    GF::VectorPartitioned m_vector;
    GF::VectorPartitioned m_vector_elas;
    GF::VectorPartitioned m_vector_plas;

    // mass matrix
    GF::MatrixDiagonalPartitioned m_M;

    // damping matrix
    GF::MatrixDiagonal m_D;

    // material definition
    GM::Array<2> m_material;
    GM::Array<2> m_material_elas;
    GM::Array<2> m_material_plas;

    // nodal displacements, velocities, and accelerations (current and last time-step)
    xt::xtensor<double, 2> m_u;
    xt::xtensor<double, 2> m_v;
    xt::xtensor<double, 2> m_a;
    xt::xtensor<double, 2> m_v_n;
    xt::xtensor<double, 2> m_a_n;

    // element vectors
    xt::xtensor<double, 3> m_ue;
    xt::xtensor<double, 3> m_fe;
    xt::xtensor<double, 3> m_ue_plas;
    xt::xtensor<double, 3> m_fe_plas;

    // nodal forces
    xt::xtensor<double, 2> m_felas;
    xt::xtensor<double, 2> m_fplas;
    xt::xtensor<double, 2> m_fdamp;
    xt::xtensor<double, 2> m_fint;
    xt::xtensor<double, 2> m_fext;
    xt::xtensor<double, 2> m_fres;

    // integration point tensors
    xt::xtensor<double, 4> m_Eps;
    xt::xtensor<double, 4> m_Eps_elas;
    xt::xtensor<double, 4> m_Eps_plas;
    xt::xtensor<double, 4> m_Sig;
    xt::xtensor<double, 4> m_Sig_elas;
    xt::xtensor<double, 4> m_Sig_plas;

    // stiffness matrix
    GF::Matrix m_K_elas;

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

#include "UniformSingleLayer2d.hpp"

#endif
