/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_GENERIC2D_H
#define FRICTIONQPOTFEM_GENERIC2D_H

#include <algorithm>
#include <string>

#include "config.h"

#include <xtensor/xnorm.hpp>
#include <xtensor/xset_operation.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xtensor.hpp>

#include <GooseFEM/GooseFEM.h>
#include <GooseFEM/Matrix.h>
#include <GooseFEM/MatrixPartitioned.h>

#include <GMatElastoPlasticQPot/Cartesian2d.h>
#include <GMatElastoPlasticQPot/version.h>

#include <GMatTensor/Cartesian2d.h>
#include <GMatTensor/version.h>

namespace FrictionQPotFEM {

/**
Convert array of yield strains stored per element [nelem, n]:

1.  Pre-prepend elastic: epsy[e, :] -> [-epsy[e, 0], epsy[e, :]]
2.  Broadcast to be the same for all quadrature points, converting the shape to [nelem, nip, n].

\param arg Yield strains per element.
\param nip Number of integration points.
\return Broadcast yield strains.
*/
inline array_type::tensor<double, 3> epsy_initelastic_toquad(
    const array_type::tensor<double, 2>& arg,
    size_t nip = GooseFEM::Element::Quad4::Gauss::nip())
{
    array_type::tensor<double, 3> ret = xt::empty<double>({arg.shape(0), nip, arg.shape(1) + 1});

    for (size_t e = 0; e < arg.shape(0); ++e) {
        for (size_t q = 0; q < nip; ++q) {
            std::copy(&arg(e, 0), &arg(e, 0) + arg.shape(1), &ret(e, q, 1));
            ret(e, q, 0) = -arg(e, 0);
        }
    }

    return ret;
}

/**
Broadcast array of moduli stored per element [nelem] to be the same for all quadrature points,
converting the shape to [nelem, nip].

\param arg Moduli per element.
\param nip Number of integration points.
\return Broadcast moduli.
*/
inline array_type::tensor<double, 2> moduli_toquad(
    const array_type::tensor<double, 1>& arg,
    size_t nip = GooseFEM::Element::Quad4::Gauss::nip())
{
    array_type::tensor<double, 2> ret = xt::empty<double>({arg.shape(0), nip});

    for (size_t e = 0; e < arg.shape(0); ++e) {
        for (size_t q = 0; q < nip; ++q) {
            ret(e, q) = arg(e);
        }
    }

    return ret;
}

/**
Extract uniform value (throw if not uniform):

-   If `all(arg[0] == arg)` return `arg[0]`.
-   Otherwise throw.

\param arg Values.
\return Uniform value.
*/
template <class T>
inline typename T::value_type getuniform(const T& arg)
{
    if (xt::allclose(arg.flat(0), arg)) {
        return arg.flat(0);
    }

    throw std::runtime_error("Values not uniform");
}

/**
Generic system of elastic and plastic elements.
For the moment this not part of the public API and can be subjected to changes.
*/
namespace Generic2d {

namespace detail {

bool is_same(double a, double b)
{
    return std::nextafter(a, std::numeric_limits<double>::lowest()) <= b &&
           std::nextafter(a, std::numeric_limits<double>::max()) >= b;
}

} // namespace detail

/**
Return versions of this library and of all of its dependencies.
The output is a list of strings, e.g.::

    "frictionqpotfem=0.7.1",
    "goosefem=0.7.0",
    ...

\return List of strings.
*/
inline std::vector<std::string> version_dependencies()
{
    return GMatTensor::version_dependencies();
}

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
    \param elastic_elem Elastic elements.
    \param elastic_K Bulk modulus per quad. point of each elastic element, see setElastic().
    \param elastic_G Shear modulus per quad. point of each elastic element, see setElastic().
    \param plastic_elem Plastic elements.
    \param plastic_K Bulk modulus per quad. point of each plastic element, see Plastic().
    \param plastic_G Shear modulus per quad. point of each plastic element, see Plastic().
    \param plastic_epsy Yield strain per quad. point of each plastic element, see Plastic().
    \param dt Time step, set setDt().
    \param rho Mass density, see setMassMatrix().
    \param alpha Background damping density, see setDampingMatrix().
    \param eta Damping at the interface (homogeneous), see setEta().
    */
    template <class C, class E, class L, class M, class Y>
    System(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elastic_elem,
        const M& elastic_K,
        const M& elastic_G,
        const L& plastic_elem,
        const M& plastic_K,
        const M& plastic_G,
        const Y& plastic_epsy,
        double dt,
        double rho,
        double alpha,
        double eta)
    {
        this->initSystem(
            coor,
            conn,
            dofs,
            iip,
            elastic_elem,
            elastic_K,
            elastic_G,
            plastic_elem,
            plastic_K,
            plastic_G,
            plastic_epsy,
            dt,
            rho,
            alpha,
            eta);
    }

protected:
    /**
    \cond
    */
    template <class C, class E, class L, class M, class Y>
    void initSystem(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elastic_elem,
        const M& elastic_K,
        const M& elastic_G,
        const L& plastic_elem,
        const M& plastic_K,
        const M& plastic_G,
        const Y& plastic_epsy,
        double dt,
        double rho,
        double alpha,
        double eta)
    {
        m_inc = 0;
        m_full_outdated = true;

        m_coor = coor;
        m_elem_elas = elastic_elem;
        m_elem_plas = plastic_elem;

        array_type::tensor<size_t, 2> conn_elas = xt::view(conn, xt::keep(m_elem_elas), xt::all());
        array_type::tensor<size_t, 2> conn_plas = xt::view(conn, xt::keep(m_elem_plas), xt::all());

        m_nnode = m_coor.shape(0);
        m_ndim = m_coor.shape(1);
        m_nelem = conn.shape(0);
        m_nne = conn.shape(1);

        m_nelem_elas = m_elem_elas.size();
        m_nelem_plas = m_elem_plas.size();
        m_N = m_nelem_plas;

#ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
        // check that "elem_plastic" and "elem_plastic" together span all elements
        array_type::tensor<size_t, 1> elem = xt::concatenate(xt::xtuple(m_elem_elas, m_elem_plas));
        FRICTIONQPOTFEM_ASSERT(xt::sort(elem) == xt::arange<size_t>(m_nelem));
        // check that all "iip" or in "dofs"
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::isin(iip, dofs)));
#endif

        m_vector = GooseFEM::VectorPartitioned(conn, dofs, iip);
        m_vector_elas = GooseFEM::VectorPartitioned(conn_elas, dofs, iip);
        m_vector_plas = GooseFEM::VectorPartitioned(conn_plas, dofs, iip);

        m_quad = GooseFEM::Element::Quad4::Quadrature(m_vector.AsElement(m_coor));
        m_quad_elas = GooseFEM::Element::Quad4::Quadrature(m_vector_elas.AsElement(m_coor));
        m_quad_plas = GooseFEM::Element::Quad4::Quadrature(m_vector_plas.AsElement(m_coor));
        m_nip = m_quad.nip();

        m_u = m_vector.allocate_nodevec(0.0);
        m_v = m_vector.allocate_nodevec(0.0);
        m_a = m_vector.allocate_nodevec(0.0);
        m_v_n = m_vector.allocate_nodevec(0.0);
        m_a_n = m_vector.allocate_nodevec(0.0);

        m_ue = m_vector.allocate_elemvec(0.0);
        m_fe = m_vector.allocate_elemvec(0.0);
        m_ue_elas = m_vector_elas.allocate_elemvec(0.0);
        m_fe_elas = m_vector_elas.allocate_elemvec(0.0);
        m_ue_plas = m_vector_plas.allocate_elemvec(0.0);
        m_fe_plas = m_vector_plas.allocate_elemvec(0.0);

        m_fmaterial = m_vector.allocate_nodevec(0.0);
        m_felas = m_vector.allocate_nodevec(0.0);
        m_fplas = m_vector.allocate_nodevec(0.0);
        m_fdamp = m_vector.allocate_nodevec(0.0);
        m_fvisco = m_vector.allocate_nodevec(0.0);
        m_ftmp = m_vector.allocate_nodevec(0.0);
        m_fint = m_vector.allocate_nodevec(0.0);
        m_fext = m_vector.allocate_nodevec(0.0);
        m_fres = m_vector.allocate_nodevec(0.0);

        m_Eps = m_quad.allocate_qtensor<2>(0.0);
        m_Sig = m_quad.allocate_qtensor<2>(0.0);
        m_Epsdot_plas = m_quad_plas.allocate_qtensor<2>(0.0);

        m_M = GooseFEM::MatrixDiagonalPartitioned(conn, dofs, iip);
        m_D = GooseFEM::MatrixDiagonal(conn, dofs);

        m_K_elas = GooseFEM::Matrix(conn_elas, dofs);

        // allocated to strain-free state, matching #m_u
        m_elas = GMatElastoPlasticQPot::Cartesian2d::Elastic<2>(elastic_K, elastic_G);
        m_plas = GMatElastoPlasticQPot::Cartesian2d::Cusp<2>(plastic_K, plastic_G, plastic_epsy);

        m_K_elas.assemble(m_quad_elas.Int_gradN_dot_tensor4_dot_gradNT_dV(m_elas.C()));

        this->setDt(dt);
        this->setRho(rho);
        this->setAlpha(alpha);
        this->setEta(eta);
    }
    /**
    \endcond
    */

public:
    /**
    Return the linear system size (in number of blocks).
    */
    virtual size_t N() const
    {
        return m_nelem_plas;
    }

public:
    /**
    Return the type of system.
    */
    virtual std::string type() const
    {
        return "FrictionQPotFEM.Generic2d.System";
    }

public:
    /**
    Mass density.
    The output is non-zero only if the density is homogeneous.
    I.e. a zero value does not mean that there is no mass.
    \return Float
    */
    double rho() const
    {
        return m_rho;
    }

public:
    /**
    Overwrite the mass density to a homogeneous quantity.
    To use a non-homogeneous density use setMassMatrix().
    \param rho Mass density.
    */
    void setRho(double rho)
    {
        return this->setMassMatrix(xt::eval(rho * xt::ones<double>({m_nelem})));
    }

public:
    /**
    Overwrite mass matrix, based on certain density that is uniform per element.
    This function allows for more heterogeneity than the constructor.
    To use a homogeneous system, use setRho().
    \tparam T e.g. `array_type::tensor<double, 1>`.
    \param val_elem Density per element.
    */
    template <class T>
    void setMassMatrix(const T& val_elem)
    {
        FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);

        if (xt::allclose(val_elem, val_elem(0))) {
            m_rho = val_elem(0);
        }
        else {
            m_rho = 0.0;
        }

        GooseFEM::Element::Quad4::Quadrature nodalQuad(
            m_vector.AsElement(m_coor),
            GooseFEM::Element::Quad4::Nodal::xi(),
            GooseFEM::Element::Quad4::Nodal::w());

        array_type::tensor<double, 2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});
        for (size_t q = 0; q < nodalQuad.nip(); ++q) {
            xt::view(val_quad, xt::all(), q) = val_elem;
        }

        m_M.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    }

public:
    /**
    Overwrite the value of the damping at the interface.
    Note that you can specify either setEta() or setDampingMatrix() or both.
    \param eta Damping parameter
    */
    void setEta(double eta)
    {
        if (detail::is_same(eta, 0.0)) {
            m_set_visco = false;
            m_eta = 0.0;
        }
        else {
            m_set_visco = true;
            m_eta = eta;
        }
    }

public:
    /**
    Get the damping at the interface.
    \return Float
    */
    double eta() const
    {
        return m_eta;
    }

public:
    /**
    Overwrite background damping density (proportional to the velocity),
    To use a non-homogeneous density use setDampingMatrix().
    \param alpha Damping parameter.
    */
    void setAlpha(double alpha)
    {
        return this->setDampingMatrix(xt::eval(alpha * xt::ones<double>({m_nelem})));
    }

public:
    /**
    Background damping density.
    The output is non-zero only if the density is homogeneous.
    I.e. a zero value does not mean that there is no damping.
    \return Float
    */
    double alpha() const
    {
        return m_alpha;
    }

public:
    /**
    Overwrite damping matrix, based on certain density (taken uniform per element).
    This function allows for more heterogeneity than the constructor.
    Note that you can specify either setEta() or setDampingMatrix() or both.
    To use a homogeneous system, use setAlpha().
    \param val_elem Damping per element.
    */
    template <class T>
    void setDampingMatrix(const T& val_elem)
    {
        FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);
        m_set_D = true;

        if (xt::allclose(val_elem, val_elem(0))) {
            m_alpha = val_elem(0);
            if (detail::is_same(m_alpha, 0.0)) {
                m_set_D = false;
            }
        }
        else {
            m_alpha = 0.0;
        }

        GooseFEM::Element::Quad4::Quadrature nodalQuad(
            m_vector.AsElement(m_coor),
            GooseFEM::Element::Quad4::Nodal::xi(),
            GooseFEM::Element::Quad4::Nodal::w());

        array_type::tensor<double, 2> val_quad = xt::empty<double>({m_nelem, nodalQuad.nip()});
        for (size_t q = 0; q < nodalQuad.nip(); ++q) {
            xt::view(val_quad, xt::all(), q) = val_elem;
        }

        m_D.assemble(nodalQuad.Int_N_scalar_NT_dV(val_quad));
    }

public:
    /**
    Check if elasticity is homogeneous.
    \return `true` is elasticity is homogeneous (`false` otherwise).
    */
    bool isHomogeneousElastic() const
    {
        auto k = this->K();
        auto g = this->G();

        return xt::allclose(k, k.data()[0]) && xt::allclose(g, g.data()[0]);
    }

public:
    /**
    Get time step.
    */
    double dt() const
    {
        return m_dt;
    }

    /**
    Overwrite time step. Using for example in System::timeStep and System::minimise.
    */
    void setDt(double dt)
    {
        m_dt = dt;
    }

public:
    /**
    Overwrite nodal displacements.
    Internally, this updates the relevant forces using updated_u().
    \param u `[nnode, ndim]`.
    */
    template <class T>
    void setU(const T& u)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(u, {m_nnode, m_ndim}));
        xt::noalias(m_u) = u;
        m_full_outdated = true;
        this->updated_u();
    }

public:
    /**
    Overwrite nodal velocities.
    Internally, this updates the relevant forces using updated_v().
    \param v `[nnode, ndim]`.
    */
    template <class T>
    void setV(const T& v)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, {m_nnode, m_ndim}));
        xt::noalias(m_v) = v;
        this->updated_v();
    }

public:
    /**
    Overwrite nodal accelerations.

    \param a `[nnode, ndim]`.
    */
    template <class T>
    void setA(const T& a)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(a, {m_nnode, m_ndim}));
        xt::noalias(m_a) = a;
    }

public:
    /**
    Evaluate relevant forces when #m_u is updated.
    Under normal circumstances, this function is called automatically when needed.
    */
    virtual void updated_u()
    {
        this->computeForceMaterial();
    }

    /**
    Evaluate relevant forces when #m_v is updated.
    Under normal circumstances, this function is called automatically when needed.
    */
    void updated_v()
    {
        if (m_set_D) {
            m_D.dot(m_v, m_fdamp);
        }

        // m_ue_plas, m_fe_plas, m_ftmp are temporaries that can be reused
        if (m_set_visco) {
            m_vector_plas.asElement(m_v, m_ue_plas);
            m_quad_plas.symGradN_vector(m_ue_plas, m_Epsdot_plas);
            m_quad_plas.int_gradN_dot_tensor2_dV(m_Epsdot_plas, m_fe_plas);
            if (!m_set_D) {
                m_vector_plas.assembleNode(m_fe_plas, m_fdamp);
                m_fdamp *= m_eta;
            }
            else {
                m_vector_plas.assembleNode(m_fe_plas, m_ftmp);
                m_ftmp *= m_eta;
                m_fdamp += m_ftmp;
            }
        }
    }

public:
    /**
    Overwrite external force.
    Note: the external force on the DOFs whose displacement are prescribed are response forces
    computed during timeStep(). Internally on the system of unknown DOFs is solved, so any
    change to the response forces is ignored.
    \param fext `[nnode, ndim]`.
    */
    template <class T>
    void setFext(const T& fext)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(fext, {m_nnode, m_ndim}));
        xt::noalias(m_fext) = fext;
    }

public:
    /**
    List of elastic elements. Shape: [System::m_nelem_elas].
    \return List of element numbers.
    */
    const auto& elastic_elem() const
    {
        return m_elem_elas;
    }

public:
    /**
    List of plastic elements. Shape: [System::m_nelem_plas].
    \return List of element numbers.
    */
    const auto& plastic_elem() const
    {
        return m_elem_plas;
    }

public:
    /**
    Connectivity. Shape: [System::m_nelem, System::m_nne].
    \return Connectivity.
    */
    const auto& conn() const
    {
        return m_vector.conn();
    }

public:
    /**
    Nodal coordinates. Shape: [System::m_nnode, System::m_ndim].
    \return Nodal coordinates.
    */
    const auto& coor() const
    {
        return m_coor;
    }

public:
    /**
    DOFs per node.
    \return DOFs per node.
    */
    const auto& dofs() const
    {
        return m_vector.dofs();
    }

public:
    /**
    Nodal displacements.
    \return Nodal displacements.
    */
    const auto& u() const
    {
        return m_u;
    }

public:
    /**
    Nodal velocities.
    \return Nodal velocities.
    */
    const auto& v() const
    {
        return m_v;
    }

public:
    /**
    Nodal accelerations.
    \return Nodal accelerations.
    */
    const auto& a() const
    {
        return m_a;
    }

public:
    /**
    Mass matrix, see System::m_M.
    \return Mass matrix (diagonal, partitioned).
    */
    auto& mass() const
    {
        return m_M;
    }

public:
    /**
    Damping matrix, see setAlpha() and setDampingMatrix().
    Note that there can be second source of damping, see setEta().
    \return Damping matrix (diagonal).
    */
    auto& damping() const
    {
        return m_D;
    }

public:
    /**
    External force.
    \return Nodal force. Shape `[nnode, ndim]`    .
    */
    const auto& fext()
    {
        this->computeInternalExternalResidualForce();
        return m_fext;
    }

public:
    /**
    Internal force.
    \return Nodal force. Shape `[nnode, ndim]`    .
    */
    const auto& fint()
    {
        this->computeInternalExternalResidualForce();
        return m_fint;
    }

public:
    /**
    Force deriving from elasticity.
    \return Nodal force. Shape `[nnode, ndim]`    .
    */
    const auto& fmaterial() const
    {
        return m_fmaterial;
    }

public:
    /**
    Force deriving from damping.
    This force is the sum of damping at the interface plus that of background damping
    (or just one of both if just one is specified).
    \return Nodal force. Shape `[nnode, ndim]`    .
    */
    const auto& fdamp() const
    {
        return m_fdamp;
    }

public:
    /**
    Norm of the relative residual force (the external force at the top/bottom boundaries is
    used for normalisation).
    \return Relative residual.
    */
    double residual() const
    {
        double r_fres = xt::norm_l2(m_fres)();
        double r_fext = xt::norm_l2(m_fext)();
        if (r_fext != 0.0) {
            return r_fres / r_fext;
        }
        return r_fres;
    }

public:
    /**
    Set nodal velocities and accelerations equal to zero.
    Call this function after an energy minimisation (taken care of in System::minimise).
    */
    void quench()
    {
        m_v.fill(0.0);
        m_a.fill(0.0);
        this->updated_v();
    }

public:
    /**
    Current time.
    */
    double t() const
    {
        return m_inc * m_dt;
    }

    /**
    Overwrite time.
    */
    void setT(double arg)
    {
        m_inc = static_cast<size_t>(arg / m_dt);
        FRICTIONQPOTFEM_REQUIRE(xt::allclose(this->t(), arg));
    }

    /**
    The increment, see setInc().
    \return size_t.
    */
    size_t inc() const
    {
        return m_inc;
    }

    /**
    Set increment.
    \param arg size_t.
    */
    void setInc(size_t arg)
    {
        m_inc = arg;
    }

public:
    /**
    Integration point volume (of each integration point)
    \return Integration point volume (System::m_quad::dV).
    */
    const auto& dV() const
    {
        return m_quad.dV();
    }

public:
    /**
    GooseFEM vector definition. Takes care of bookkeeping.
    \return GooseFEM::VectorPartitioned (System::m_vector)
    */
    const GooseFEM::VectorPartitioned& vector() const
    {
        return m_vector;
    }

public:
    /**
    GooseFEM quadrature definition. Takes case of interpolation, integration, and differentiation.
    \return GooseFEM::Element::Quad4::Quadrature (System::m_quad)
    */
    const GooseFEM::Element::Quad4::Quadrature& quad() const
    {
        return m_quad;
    }

public:
    /**
    Elastic material mode, used for all elastic elements.
    \return GMatElastoPlasticQPot::Cartesian2d::Elastic<2>&
    */
    GMatElastoPlasticQPot::Cartesian2d::Elastic<2>& elastic()
    {
        return m_elas;
    }

public:
    /**
    Elastic material mode, used for all elastic elements.
    \return GMatElastoPlasticQPot::Cartesian2d::Cusp<2>&
    */
    GMatElastoPlasticQPot::Cartesian2d::Cusp<2>& plastic()
    {
        return m_plas;
    }

public:
    /**
    Bulk modulus per integration point.
    Convenience function: assembles output from elastic() and plastic().
    \return Integration point scalar (copy). Shape: `[nelem, nip]`.
    */
    array_type::tensor<double, 2> K() const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>({m_nelem, m_nip});

        const auto& ret_elas = m_elas.K();
        const auto& ret_plas = m_plas.K();
        size_t n = xt::strides(ret_elas, 0);
        FRICTIONQPOTFEM_ASSERT(n == m_nip);

        for (size_t e = 0; e < m_nelem_elas; ++e) {
            std::copy(
                &ret_elas(e, xt::missing),
                &ret_elas(e, xt::missing) + n,
                &ret(m_elem_elas(e), xt::missing));
        }

        for (size_t e = 0; e < m_nelem_plas; ++e) {
            std::copy(
                &ret_plas(e, xt::missing),
                &ret_plas(e, xt::missing) + n,
                &ret(m_elem_plas(e), xt::missing));
        }

        return ret;
    }

public:
    /**
    Shear modulus per integration point.
    Convenience function: assembles output from elastic() and plastic().
    \return Integration point scalar (copy). Shape: `[nelem, nip]`.
    */
    array_type::tensor<double, 2> G() const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>({m_nelem, m_nip});

        const auto& ret_elas = m_elas.G();
        const auto& ret_plas = m_plas.G();
        size_t n = xt::strides(ret_elas, 0);
        FRICTIONQPOTFEM_ASSERT(n == m_nip);

        for (size_t e = 0; e < m_nelem_elas; ++e) {
            std::copy(
                &ret_elas(e, xt::missing),
                &ret_elas(e, xt::missing) + n,
                &ret(m_elem_elas(e), xt::missing));
        }

        for (size_t e = 0; e < m_nelem_plas; ++e) {
            std::copy(
                &ret_plas(e, xt::missing),
                &ret_plas(e, xt::missing) + n,
                &ret(m_elem_plas(e), xt::missing));
        }

        return ret;
    }

public:
    /**
    Stress tensor of each integration point.
    \return Integration point tensor (pointer). Shape: `[nelem, nip, 2, 2]`.
    */
    const array_type::tensor<double, 4>& Sig()
    {
        this->computeEpsSig();
        return m_Sig;
    }

public:
    /**
    Strain tensor of each integration point.
    \return Integration point tensor (pointer). Shape: `[nelem, nip, 2, 2]`.
    */
    const array_type::tensor<double, 4>& Eps()
    {
        this->computeEpsSig();
        return m_Eps;
    }

public:
    /**
    Strain-rate tensor (the symmetric gradient of the nodal velocities) of each integration point.
    \return Integration point tensor (copy). Shape: `[nelem, nip, 2, 2]`.
    */
    array_type::tensor<double, 4> Epsdot() const
    {
        return m_quad.SymGradN_vector(m_vector.AsElement(m_v));
    }

public:
    /**
    Symmetric gradient of the nodal accelerations of each integration point.
    \return Integration point tensor (copy). Shape: `[nelem, nip, 2, 2]`.
    */
    array_type::tensor<double, 4> Epsddot() const
    {
        return m_quad.SymGradN_vector(m_vector.AsElement(m_a));
    }

public:
    /**
    Stiffness tensor of each integration point.
    Convenience function: assembles output from elastic() and plastic().
    \return Integration point tensor (copy). Shape: `[nelem, nip, 2, 2, 2, 2]`.
    */
    GooseFEM::MatrixPartitioned stiffness() const
    {
        auto ret = m_quad.allocate_qtensor<4>(0.0);
        const auto& ret_plas = m_plas.C();
        const auto& ret_elas = m_elas.C();
        size_t n = xt::strides(ret_elas, 0);
        FRICTIONQPOTFEM_ASSERT(n == m_nip * 16);

        for (size_t e = 0; e < m_nelem_elas; ++e) {
            std::copy(
                &ret_elas(e, xt::missing),
                &ret_elas(e, xt::missing) + n,
                &ret(m_elem_elas(e), xt::missing));
        }

        for (size_t e = 0; e < m_nelem_plas; ++e) {
            std::copy(
                &ret_plas(e, xt::missing),
                &ret_plas(e, xt::missing) + n,
                &ret(m_elem_plas(e), xt::missing));
        }

        GooseFEM::MatrixPartitioned K(m_vector.conn(), m_vector.dofs(), m_vector.iip());
        K.assemble(m_quad.Int_gradN_dot_tensor4_dot_gradNT_dV(ret));
        return K;
    }

public:
    /**
    Strain-rate tensor of integration points of plastic elements only, see System::plastic.
    \return Integration point tensor (pointer). Shape: [plastic_elem().size(), nip, 2, 2].
    */
    const array_type::tensor<double, 4>& plastic_Epsdot()
    {
        // m_ue_plas, m_fe_plas are temporaries that can be reused
        if (!m_set_visco) {
            m_vector_plas.asElement(m_v, m_ue_plas);
            m_quad_plas.symGradN_vector(m_ue_plas, m_Epsdot_plas);
        }

        return m_Epsdot_plas;
    }

public:
    /**
    Check that the current yield-index is at least `n` away from the end.
    \param n Margin.
    \return `true` if the current yield-index is at least `n` away from the end.
    */
    bool boundcheck_right(size_t n) const
    {
        FRICTIONQPOTFEM_REQUIRE(m_plas.epsy().shape(2) > n);
        return xt::all(m_plas.i() < m_plas.epsy().shape(2) - n);
    }

protected:
    /**
    Compute #m_Sig and #m_Eps using #m_u.
    */
    void computeEpsSig()
    {
        if (!m_full_outdated) {
            return;
        }

        auto& Eps_elas = m_elas.Eps();
        auto& Sig_elas = m_elas.Sig();
        const auto& Eps_plas = m_plas.Eps();
        const auto& Sig_plas = m_plas.Sig();

        m_vector_elas.asElement(m_u, m_ue_elas);
        m_quad_elas.symGradN_vector(m_ue_elas, Eps_elas);
        m_elas.refresh();

        size_t n = xt::strides(Eps_elas, 0);
        FRICTIONQPOTFEM_ASSERT(n == m_nip * 4);

        for (size_t e = 0; e < m_nelem_elas; ++e) {
            std::copy(
                &Eps_elas(e, xt::missing),
                &Eps_elas(e, xt::missing) + n,
                &m_Eps(m_elem_elas(e), xt::missing));
            std::copy(
                &Sig_elas(e, xt::missing),
                &Sig_elas(e, xt::missing) + n,
                &m_Sig(m_elem_elas(e), xt::missing));
        }
        for (size_t e = 0; e < m_nelem_plas; ++e) {
            std::copy(
                &Eps_plas(e, xt::missing),
                &Eps_plas(e, xt::missing) + n,
                &m_Eps(m_elem_plas(e), xt::missing));
            std::copy(
                &Sig_plas(e, xt::missing),
                &Sig_plas(e, xt::missing) + n,
                &m_Sig(m_elem_plas(e), xt::missing));
        }

        m_full_outdated = false;
    }

    /**
    Update #m_fmaterial based on the current displacement field #m_u.
    */
    void computeForceMaterial()
    {
        m_full_outdated = true;

        m_vector_plas.asElement(m_u, m_ue_plas);
        m_quad_plas.symGradN_vector(m_ue_plas, m_plas.Eps());
        m_plas.refresh();
        m_quad_plas.int_gradN_dot_tensor2_dV(m_plas.Sig(), m_fe_plas);
        m_vector_plas.assembleNode(m_fe_plas, m_fplas);

        m_K_elas.dot(m_u, m_felas);

        xt::noalias(m_fmaterial) = m_felas + m_fplas;
    }

protected:
    /**
    Compute:
    -   m_fint = m_fmaterial + m_fdamp
    -   m_fext[iip] = m_fint[iip]
    -   m_fres = m_fext - m_fint

    Internal rule: all relevant forces are expected to be updated before this function is called.
    */
    virtual void computeInternalExternalResidualForce()
    {
        xt::noalias(m_fint) = m_fmaterial + m_fdamp;
        m_vector.copy_p(m_fint, m_fext);
        xt::noalias(m_fres) = m_fext - m_fint;
    }

public:
    /**
    Recompute all forces.
    Under normal circumstances, this function is called automatically when needed.
    */
    void refresh()
    {
        this->updated_u();
        this->updated_v();
        this->computeInternalExternalResidualForce();
    }

public:
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
    double eventDriven_setDeltaU(const T& delta_u, bool autoscale = true)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(delta_u, m_u.shape()));
        m_pert_u = delta_u;

        m_vector_plas.asElement(delta_u, m_ue_plas);
        auto Eps = m_quad_plas.SymGradN_vector(m_ue_plas);
        m_pert_Epsd_plastic = GMatTensor::Cartesian2d::Deviatoric(Eps);

        if (!autoscale) {
            return 1.0;
        }

        auto deps = xt::amax(GMatElastoPlasticQPot::Cartesian2d::Epsd(m_pert_Epsd_plastic))();
        auto d = xt::amin(xt::diff(m_plas.epsy(), 1))();
        double c = 0.1 * d / deps;

        m_pert_u *= c;
        m_pert_Epsd_plastic *= c;

        return c;
    }

    /**
    Get displacement perturbation used for event driven code, see eventDriven_setDeltaU().
    \return Nodal displacements.
    */
    const auto& eventDriven_deltaU() const
    {
        return m_pert_u;
    }

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
        If `false`, increment displacements to `deps / 2` of yielding again.
        If `true`, increment displacements by a affine simple shear increment `deps`.

    \param direction
        If `+1`: apply shear to the right. If `-1` applying shear to the left.

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
        bool iterative = false)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(m_pert_u, m_u.shape()));
        FRICTIONQPOTFEM_ASSERT(direction == 1 || direction == -1);

        auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(m_plas.Eps());
        auto Epsd_plastic = GMatTensor::Cartesian2d::Deviatoric(m_plas.Eps());
        auto idx = m_plas.i();
        const auto& epsy_l = m_plas.epsy_left();
        const auto& epsy_r = m_plas.epsy_right();

        FRICTIONQPOTFEM_WIP(iterative || direction > 0 || !xt::any(xt::equal(idx, 0)));

        array_type::tensor<double, 2> target;

        if (direction > 0 && kick) { // direction > 0 && kick
            target = epsy_r + 0.5 * deps;
        }
        else if (direction > 0) { // direction > 0 && !kick
            target = epsy_r - 0.5 * deps;
        }
        else if (kick) { // direction < 0 && kick
            target = epsy_l - 0.5 * deps;
        }
        else { // direction < 0 && !kick
            target = epsy_l + 0.5 * deps;
        }

        if (!kick) {
            auto d = target - eps;
            if (direction > 0 && xt::any(d < 0.0)) {
                return 0.0;
            }
            if (direction < 0 && xt::any(d > 0.0)) {
                return 0.0;
            }
        }

        auto scale = xt::empty_like(target);

        for (size_t e = 0; e < m_nelem_plas; ++e) {
            for (size_t q = 0; q < m_nip; ++q) {

                double e_t = Epsd_plastic(e, q, 0, 0);
                double g_t = Epsd_plastic(e, q, 0, 1);
                double e_d = m_pert_Epsd_plastic(e, q, 0, 0);
                double g_d = m_pert_Epsd_plastic(e, q, 0, 1);
                double epsd_target = target(e, q);

                double a = e_d * e_d + g_d * g_d;
                double b = 2.0 * (e_t * e_d + g_t * g_d);
                double c = e_t * e_t + g_t * g_t - epsd_target * epsd_target;
                double D = std::sqrt(b * b - 4.0 * a * c);

                FRICTIONQPOTFEM_REQUIRE(b >= 0.0 || iterative);
                scale(e, q) = (-b + D) / (2.0 * a);
            }
        }

        double ret;
        size_t e;
        size_t q;

        if (!iterative) {

            auto index = xt::unravel_index(xt::argmin(xt::abs(scale))(), scale.shape());
            e = index[0];
            q = index[1];

            if (kick && yield_element) {
                q = xt::argmax(xt::view(xt::abs(scale), e, xt::all()))();
            }

            ret = scale(e, q);

            if ((direction > 0 && ret < 0) || (direction < 0 && ret > 0)) {
                if (!kick) {
                    return 0.0;
                }
                else {
                    FRICTIONQPOTFEM_REQUIRE(
                        (direction > 0 && ret > 0) || (direction < 0 && ret < 0));
                }
            }
        }

        else {

            double dir = static_cast<double>(direction);
            auto steps = xt::sort(xt::ravel(xt::eval(xt::abs(scale))));
            auto u_n = m_u;
            xt::xtensor<long, 2> jdx = xt::cast<long>(m_plas.i());
            size_t i;
            long nip = static_cast<long>(m_nip);

            // find first step that:
            // if (!kick || (kick && !yield_element)): is plastic for the first time
            // if (kick && yield_element): yields the element for the first time

            for (i = 0; i < steps.size(); ++i) {
                this->setU(u_n + dir * steps(i) * m_pert_u);
                auto jdx_new = xt::cast<long>(m_plas.i());
                auto S = xt::abs(jdx_new - jdx);
                if (!yield_element || !kick) {
                    if (xt::any(S > 0)) {
                        break;
                    }
                }
                else {
                    if (xt::any(xt::equal(xt::sum(S > 0, 1), nip))) {
                        break;
                    }
                }
            }

            // iterate to acceptable step

            double right = steps(i);
            double left = 0.0;
            ret = right;

            if (i > 0) {
                left = steps(i - 1);
            }

            // iterate to actual step

            for (size_t iiter = 0; iiter < 1100; ++iiter) {

                ret = 0.5 * (right + left);
                this->setU(u_n + dir * ret * m_pert_u);
                auto jdx_new = xt::cast<long>(m_plas.i());
                auto S = xt::abs(jdx_new - jdx);

                if (!kick) {
                    if (xt::any(S > 0)) {
                        right = ret;
                    }
                    else {
                        left = ret;
                    }
                }
                else if (yield_element) {
                    if (xt::any(xt::equal(xt::sum(S > 0, 1), nip))) {
                        right = ret;
                    }
                    else {
                        left = ret;
                    }
                }
                else {
                    if (xt::any(S > 0)) {
                        right = ret;
                    }
                    else {
                        left = ret;
                    }
                }

                if ((right - left) / left < 1e-5) {
                    break;
                }
                FRICTIONQPOTFEM_REQUIRE(iiter < 1000);
            }

            // final assertion: make sure that "left" and "right" are still bounds

            {
                this->setU(u_n + dir * left * m_pert_u);
                auto jdx_new = xt::cast<long>(m_plas.i());
                auto S = xt::abs(jdx_new - jdx);

                FRICTIONQPOTFEM_REQUIRE(kick || xt::all(xt::equal(S, 0)));
                if (kick && yield_element) {
                    FRICTIONQPOTFEM_REQUIRE(!xt::any(xt::equal(xt::sum(S > 0, 1), nip)));
                }
                else if (kick) {
                    FRICTIONQPOTFEM_REQUIRE(xt::all(xt::equal(S, 0)));
                }
            }
            {
                this->setU(u_n + dir * right * m_pert_u);
                auto jdx_new = xt::cast<long>(m_plas.i());
                auto S = xt::abs(jdx_new - jdx);
                FRICTIONQPOTFEM_REQUIRE(!xt::all(xt::equal(S, 0)));

                FRICTIONQPOTFEM_REQUIRE(kick || !xt::all(xt::equal(S, 0)));
                if (kick && yield_element) {
                    FRICTIONQPOTFEM_REQUIRE(xt::any(xt::equal(xt::sum(S > 0, 1), nip)));
                }
                else if (kick) {
                    FRICTIONQPOTFEM_REQUIRE(xt::any(S > 0));
                }
            }

            // "output"

            if (!kick) {
                ret = dir * left;
            }
            else {
                ret = dir * right;
            }
            this->setU(u_n);
            FRICTIONQPOTFEM_REQUIRE((direction > 0 && ret >= 0) || (direction < 0 && ret <= 0));
        }

        this->setU(m_u + ret * m_pert_u);

        const auto& idx_new = m_plas.i();
        FRICTIONQPOTFEM_REQUIRE(kick || xt::all(xt::equal(idx, idx_new)));
        FRICTIONQPOTFEM_REQUIRE(!kick || xt::any(xt::not_equal(idx, idx_new)));

        if (!iterative) {
            auto eps_new = GMatElastoPlasticQPot::Cartesian2d::Epsd(m_plas.Eps())(e, q);
            auto eps_target = target(e, q);
            FRICTIONQPOTFEM_REQUIRE(xt::allclose(eps_new, eps_target));
        }

        return ret;
    }

    /**
    Make a time-step: apply velocity-Verlet integration.
    Forces are computed where needed using:
    updated_u(), updated_v(), and computeInternalExternalResidualForce().
    */
    void timeStep()
    {
        // history

        m_inc++;
        xt::noalias(m_v_n) = m_v;
        xt::noalias(m_a_n) = m_a;

        // new displacement

        xt::noalias(m_u) = m_u + m_dt * m_v + 0.5 * std::pow(m_dt, 2.0) * m_a;
        this->updated_u();

        // estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + m_dt * m_a_n;
        this->updated_v();

        // compute residual force & solve

        this->computeInternalExternalResidualForce();
        m_M.solve(m_fres, m_a);

        // re-estimate new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
        this->updated_v();

        // compute residual force & solve

        this->computeInternalExternalResidualForce();
        m_M.solve(m_fres, m_a);

        // new velocity, update corresponding force

        xt::noalias(m_v) = m_v_n + 0.5 * m_dt * (m_a_n + m_a);
        this->updated_v();

        // compute residual force & solve

        this->computeInternalExternalResidualForce();
        m_M.solve(m_fres, m_a);
    }

    /**
    Make a number of time steps, see timeStep().

    \param n Number of steps to make.

    \param nmargin
        Number of potentials to leave as margin.
        -   `nmargin > 0`: function stops if the yield-index of any box is `nmargin` from the end.
            In that case the function returns a negative number.
        -   `nmargin == 0`: no bounds-check is performed
            (the last potential is assumed infinitely elastic to the right).

    \return
        -   Number of iterations: `== n`
        -   Negative number: if stopped because of a yield-index margin.
    */
    long timeSteps(size_t n, size_t nmargin = 5)
    {
        FRICTIONQPOTFEM_REQUIRE(n + 1 < std::numeric_limits<long>::max());

        size_t nyield = m_plas.epsy().shape(2);
        size_t nmax = nyield - nmargin;
        long iiter;

        FRICTIONQPOTFEM_ASSERT(nmargin < nyield);
        FRICTIONQPOTFEM_REQUIRE(xt::all(m_plas.i() <= nmax));

        for (iiter = 1; iiter < static_cast<long>(n + 1); ++iiter) {

            this->timeStep();

            if (nmargin > 0) {
                if (xt::any(m_plas.i() > nmax)) {
                    return -iiter;
                }
            }
        }

        return iiter;
    }

    /**
    Perform a series of time-steps until the next plastic event, or equilibrium.

    \param nmargin Number of potentials to have as margin **initially**.
    \param tol Relative force tolerance for equilibrium. See System::residual for definition.
    \param niter_tol Enforce the residual check for `niter_tol` consecutive increments.
    \param max_iter Maximum number of iterations.  Throws `std::runtime_error` otherwise.

    \return
        -   Number of steps (`< max_iter` in case of convergence, `== max_iter` for no convergence).
        -   `0` if there was no plastic activity and the residual was reached.
    */
    size_t timeStepsUntilEvent(
        size_t nmargin = 5,
        double tol = 1e-5,
        size_t niter_tol = 20,
        size_t max_iter = 1e7)
    {
        FRICTIONQPOTFEM_ASSERT(tol < 1.0);
        FRICTIONQPOTFEM_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);
        size_t nyield = m_plas.epsy().shape(2);
        size_t nmax = nyield - nmargin;
        auto i_n = m_plas.i();
        size_t iiter;

        FRICTIONQPOTFEM_ASSERT(nmargin < nyield);
        FRICTIONQPOTFEM_REQUIRE(xt::all(m_plas.i() <= nmax));

        for (iiter = 1; iiter < max_iter + 1; ++iiter) {

            this->timeStep();

            if (xt::any(xt::not_equal(m_plas.i(), i_n))) {
                return iiter;
            }

            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return 0;
            }
        }

        return iiter;
    }

    /**
    Make a number of steps with the following protocol.
    1.  Add a displacement \f$ \underline{v} \Delta t \f$ to each of the nodes.
    2.  Make a timeStep().

    \param n
        Number of steps to make.

    \param v
        Nodal velocity `[nnode, ndim]`.s

    \param nmargin
        Number of potentials to leave as margin.
        -   `nmargin > 0`: stop if the yield-index of any box is `nmargin` from the end.
            In that case the function returns a negative number.
        -   `nmargin == 0`: no bounds-check is performed
            (the last potential is assumed infinitely elastic to the right).

    \return
        Number of time-steps made (negative if failure).
    */
    template <class T>
    long flowSteps(size_t n, const T& v, size_t nmargin = 5)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, m_u.shape()));
        FRICTIONQPOTFEM_REQUIRE(n + 1 < std::numeric_limits<long>::max());

        size_t nyield = m_plas.epsy().shape(2);
        size_t nmax = nyield - nmargin;
        long iiter;

        FRICTIONQPOTFEM_ASSERT(nmargin < nyield);
        FRICTIONQPOTFEM_REQUIRE(xt::all(m_plas.i() <= nmax));

        for (iiter = 1; iiter < static_cast<long>(n + 1); ++iiter) {

            m_u += v * m_dt;
            this->timeStep();

            if (nmargin > 0) {
                if (xt::any(m_plas.i() > nmax)) {
                    return -iiter;
                }
            }
        }

        return iiter;
    }

    /**
    Minimise energy: run System::timeStep until a mechanical equilibrium has been reached.
    Can also be used to run `n` time-steps (or `< n` is equilibrium is reached before).
    In that case, call with `minimise(max_iter=n, max_iter_is_error=False)`

    \param nmargin
        Number of potentials to leave as margin.
        -   `nmargin > 0`: function stops if the yield-index of any box is `nmargin` from the end.
            In that case the function returns a negative number.
        -   `nmargin == 0`: no bounds-check is performed
            (the last potential is assumed infinitely elastic to the right).

    \param tol
        Relative force tolerance for equilibrium. See System::residual for definition.

    \param niter_tol
        Enforce the residual check for `niter_tol` consecutive increments.

    \param max_iter
        Maximum number of time-steps. Throws `std::runtime_error` otherwise.

    \param time_activity
        If `true` plastic activity is timed. After this function you can find:
        -   quasistaticActivityFirst() : Increment with the first plastic event.
        -   quasistaticActivityLast() : Increment with the last plastic event.
        Attention: if you are changing the chunk of yield positions during the minimisation you
        should copy quasistaticActivityFirst() after the first (relevant) call of minimise():
        each time you call minimise(), quasistaticActivityFirst() is reset.

    \param max_iter_is_error
        If `true` an error is thrown when the maximum number of time-steps is reached.
        If `false` the function simply returns `max_iter`.

    \return
        -   `0`: if stopped when the residual is reached (and number of steps `< max_iter`).
        -   `max_iter`: if no residual was reached, and `max_iter_is_error = false`.
        -   Negative number: if stopped because of a yield-index margin.
    */
    size_t minimise(
        size_t nmargin = 5,
        double tol = 1e-5,
        size_t niter_tol = 20,
        size_t max_iter = 1e7,
        bool time_activity = false,
        bool max_iter_is_error = true)
    {
        FRICTIONQPOTFEM_ASSERT(tol < 1.0);
        FRICTIONQPOTFEM_ASSERT(max_iter + 1 < std::numeric_limits<long>::max());

        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        size_t nyield = m_plas.epsy().shape(2);
        size_t nmax = nyield - nmargin;
        array_type::tensor<long, 2> i_n = xt::cast<long>(m_plas.i());
        long s = 0;
        long s_n = 0;
        bool init = true;
        long iiter;

        FRICTIONQPOTFEM_ASSERT(nmargin < nyield);
        FRICTIONQPOTFEM_REQUIRE(xt::all(m_plas.i() <= nmax));

        for (iiter = 1; iiter < static_cast<long>(max_iter + 1); ++iiter) {

            this->timeStep();
            residuals.roll_insert(this->residual());

            if (nmargin > 0) {
                if (xt::any(m_plas.i() > nmax)) {
                    return -iiter;
                }
            }

            if (time_activity) {
                array_type::tensor<long, 2> i = xt::cast<long>(m_plas.i());
                s = xt::sum(xt::abs(i - i_n))();
                if (s != s_n) {
                    if (init) {
                        init = false;
                        m_qs_inc_first = m_inc;
                    }
                    m_qs_inc_last = m_inc;
                }
                s_n = s;
            }

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return 0;
            }
        }

        if (max_iter_is_error) {
            throw std::runtime_error("No convergence found");
        }

        return iiter;
    }

    /**
    Increment with the first plastic event.
    This value is only relevant if `time_activity = true` was used in the last call of minimise().
    \return Increment.
    */
    size_t quasistaticActivityFirst() const
    {
        return m_qs_inc_first;
    }

    /**
    Increment with the last plastic event.
    This value is only relevant if `time_activity = true` was used in the last call of minimise().
    \return Increment.
    */
    size_t quasistaticActivityLast() const
    {
        return m_qs_inc_last;
    }

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
        size_t max_iter = 1e7)
    {
        FRICTIONQPOTFEM_REQUIRE(xt::has_shape(idx_n, std::array<size_t, 1>{m_N}));
        FRICTIONQPOTFEM_REQUIRE(S_truncate == 0);
        FRICTIONQPOTFEM_REQUIRE(A_truncate > 0);
        FRICTIONQPOTFEM_ASSERT(tol < 1.0);
        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

            this->timeStep();
            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return iiter;
            }

            array_type::tensor<size_t, 1> idx = xt::view(m_plas.i(), xt::all(), 0);

            if (static_cast<size_t>(xt::sum(xt::not_equal(idx_n, idx))()) >= A_truncate) {
                return 0;
            }
        }

        std::cout << "residuals = " << xt::adapt(residuals.data()) << std::endl;
        bool converged = false;
        FRICTIONQPOTFEM_REQUIRE(converged == true);
        return 0; // irrelevant, the code never goes here
    }

    /**
    Like Generic2d::minimise(), but stops when a certain number of blocks has yielded at least once.
    In that case the function returns zero (in all other cases it returns a positive number).

    \note `A_truncate` and `S_truncate` are defined on the first integration point.

    \param A_truncate
        Truncate if `A_truncate` blocks have yielded at least once.

    \param S_truncate
        Truncate if the number of times blocks yielded is equal to `S_truncate`.
        **Warning** This option is reserved for future use, but for the moment does nothing.

    \param tol
        Relative force tolerance for equilibrium. See System::residual for definition.

    \param niter_tol
        Enforce the residual check for `niter_tol` consecutive increments.

    \param max_iter
        Maximum number of time-steps. Throws `std::runtime_error` otherwise.
    */
    size_t minimise_truncate(
        size_t A_truncate = 0,
        size_t S_truncate = 0,
        double tol = 1e-5,
        size_t niter_tol = 20,
        size_t max_iter = 1e7)
    {
        array_type::tensor<size_t, 1> idx_n = xt::view(m_plas.i(), xt::all(), 0);
        return this->minimise_truncate(idx_n, A_truncate, S_truncate, tol, niter_tol, max_iter);
    }

    /**
    Get the displacement field that corresponds to an affine simple shear of a certain strain.
    The displacement of the bottom boundary is zero, while it is maximal for the top boundary.

    \param delta_gamma Strain to add (the shear component of deformation gradient is twice that).
    \return Nodal displacements.
    */
    array_type::tensor<double, 2> affineSimpleShear(double delta_gamma) const
    {
        array_type::tensor<double, 2> ret = xt::zeros_like(m_u);

        for (size_t n = 0; n < m_nnode; ++n) {
            ret(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - m_coor(0, 1));
        }

        return ret;
    }

    /**
    Get the displacement field that corresponds to an affine simple shear of a certain strain.
    Similar to affineSimpleShear() with the difference that the displacement is zero
    exactly in the middle, while the displacement at the bottom and the top boundary is maximal
    (with a negative displacement for the bottom boundary).

    \param delta_gamma Strain to add (the shear component of deformation gradient is twice that).
    \return Nodal displacements.
    */
    array_type::tensor<double, 2> affineSimpleShearCentered(double delta_gamma) const
    {
        array_type::tensor<double, 2> ret = xt::zeros_like(m_u);
        size_t ll = m_vector.conn()(m_elem_plas(0), 0);
        size_t ul = m_vector.conn()(m_elem_plas(0), 3);
        double y0 = (m_coor(ul, 1) + m_coor(ll, 1)) / 2.0;

        for (size_t n = 0; n < m_nnode; ++n) {
            ret(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - y0);
        }

        return ret;
    }

protected:
    array_type::tensor<double, 2> m_coor; ///< Nodal coordinates, see coor().
    size_t m_N; ///< Number of plastic elements, alias of #m_nelem_plas.
    size_t m_nelem; ///< Number of elements.
    size_t m_nelem_elas; ///< Number of elastic elements.
    size_t m_nelem_plas; ///< Number of plastic elements.
    size_t m_nne; ///< Number of nodes per element.
    size_t m_nnode; ///< Number of nodes.
    size_t m_ndim; ///< Number of spatial dimensions.
    size_t m_nip; ///< Number of integration points.
    array_type::tensor<size_t, 1> m_elem_elas; ///< Elastic elements.
    array_type::tensor<size_t, 1> m_elem_plas; ///< Plastic elements.
    GooseFEM::Element::Quad4::Quadrature m_quad; ///< Quadrature for all elements.
    GooseFEM::Element::Quad4::Quadrature m_quad_elas; ///< #m_quad for elastic elements only.
    GooseFEM::Element::Quad4::Quadrature m_quad_plas; ///< #m_quad for plastic elements only.
    GooseFEM::VectorPartitioned m_vector; ///< Convert vectors between 'nodevec', 'elemvec', ....
    GooseFEM::VectorPartitioned m_vector_elas; ///< #m_vector for elastic elements only.
    GooseFEM::VectorPartitioned m_vector_plas; ///< #m_vector for plastic elements only.
    GooseFEM::MatrixDiagonalPartitioned m_M; ///< Mass matrix (diagonal).
    GooseFEM::MatrixDiagonal m_D; ///< Damping matrix (diagonal).
    GMatElastoPlasticQPot::Cartesian2d::Elastic<2> m_elas; ///< Material for elastic el.
    GMatElastoPlasticQPot::Cartesian2d::Cusp<2> m_plas; ///< Material for plastic el.

    /**
    Nodal displacements.
    \warning To make sure that the right forces are computed at the right time,
    always call updated_u() after manually updating #m_u (setU() automatically take care of this).
    */
    array_type::tensor<double, 2> m_u;

    /**
    Nodal velocities.
    \warning To make sure that the right forces are computed at the right time,
    always call updated_v() after manually updating #m_v (setV() automatically take care of this).
    */
    array_type::tensor<double, 2> m_v;

    array_type::tensor<double, 2> m_a; ///< Nodal accelerations.
    array_type::tensor<double, 2> m_v_n; ///< Nodal velocities last time-step.
    array_type::tensor<double, 2> m_a_n; ///< Nodal accelerations last time-step.
    array_type::tensor<double, 3> m_ue; ///< Element vector (used for displacements).
    array_type::tensor<double, 3> m_fe; ///< Element vector (used for forces).
    array_type::tensor<double, 3> m_ue_elas; ///< #m_ue for elastic element only
    array_type::tensor<double, 3> m_fe_elas; ///< #m_fe for elastic element only
    array_type::tensor<double, 3> m_ue_plas; ///< #m_ue for plastic element only
    array_type::tensor<double, 3> m_fe_plas; ///< #m_fe for plastic element only
    array_type::tensor<double, 2> m_fmaterial; ///< Nodal force, deriving from elasticity.
    array_type::tensor<double, 2> m_felas; ///< Nodal force from elasticity of elastic elements.
    array_type::tensor<double, 2> m_fplas; ///< Nodal force from elasticity of plastic elements.
    array_type::tensor<double, 2> m_fdamp; ///< Nodal force from damping.
    array_type::tensor<double, 2> m_fvisco; ///< Nodal force from damping at the interface
    array_type::tensor<double, 2> m_ftmp; ///< Temporary for internal use.
    array_type::tensor<double, 2> m_fint; ///< Nodal force: total internal force.
    array_type::tensor<double, 2> m_fext; ///< Nodal force: total external force (reaction force)
    array_type::tensor<double, 2> m_fres; ///< Nodal force: residual force.
    array_type::tensor<double, 4> m_Eps; ///< Quad-point tensor: strain.
    array_type::tensor<double, 4> m_Sig; ///< Quad-point tensor: stress.
    array_type::tensor<double, 4> m_Epsdot_plas; ///< Quad-point tensor: strain-rate for plastic el.
    GooseFEM::Matrix m_K_elas; ///< Stiffness matrix for elastic elements only.
    size_t m_qs_inc_first = 0; ///< First increment with plastic activity during minimisation.
    size_t m_qs_inc_last = 0; ///< Last increment with plastic activity during minimisation.
    size_t m_inc; ///< Current increment (current time = #m_dt * #m_inc).
    double m_dt; ///< Time-step.
    double m_eta; ///< Damping at the interface
    double m_rho; ///< Mass density (non-zero only if homogeneous).
    double m_alpha; ///< Background damping density (non-zero only if homogeneous).
    bool m_set_D; ///< Use #m_D only if it is non-zero.
    bool m_set_visco; ///< Use #m_eta only if it is non-zero.
    bool m_full_outdated; ///< Keep track of the need to recompute fields on full geometry.
    array_type::tensor<double, 2> m_pert_u; ///< See eventDriven_setDeltaU()
    array_type::tensor<double, 4> m_pert_Epsd_plastic; ///< Strain deviator for #m_pert_u.
};

} // namespace Generic2d
} // namespace FrictionQPotFEM

#endif
