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
        const L& elem_plastic)
    {
        this->initSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
    }

protected:
    /**
    Constructor alias, useful for derived classes.

    \tparam C Type of nodal coordinates, e.g. `array_type::tensor<double, 2>`
    \tparam E Type of connectivity and DOFs, e.g. `array_type::tensor<size_t, 2>`
    \tparam L Type of node/element lists, e.g. `array_type::tensor<size_t, 1>`
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
        const L& elem_plastic)
    {
        m_coor = coor;
        m_conn = conn;
        m_dofs = dofs;
        m_iip = iip;
        m_elem_elas = elem_elastic;
        m_elem_plas = elem_plastic;

        m_conn_elas = xt::view(m_conn, xt::keep(m_elem_elas), xt::all());
        m_conn_plas = xt::view(m_conn, xt::keep(m_elem_plas), xt::all());

        m_nnode = m_coor.shape(0);
        m_ndim = m_coor.shape(1);
        m_nelem = m_conn.shape(0);
        m_nne = m_conn.shape(1);

        m_nelem_elas = m_elem_elas.size();
        m_nelem_plas = m_elem_plas.size();
        m_set_elas = !m_nelem_elas;
        m_set_plas = !m_nelem_plas;
        m_N = m_nelem_plas;

#ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
        // check that "elem_plastic" and "elem_plastic" together span all elements
        array_type::tensor<size_t, 1> elem = xt::concatenate(xt::xtuple(m_elem_elas, m_elem_plas));
        FRICTIONQPOTFEM_ASSERT(xt::sort(elem) == xt::arange<size_t>(m_nelem));
        // check that all "iip" or in "dofs"
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::isin(m_iip, m_dofs)));
#endif

        m_vector = GooseFEM::VectorPartitioned(m_conn, m_dofs, m_iip);
        m_vector_elas = GooseFEM::VectorPartitioned(m_conn_elas, m_dofs, m_iip);
        m_vector_plas = GooseFEM::VectorPartitioned(m_conn_plas, m_dofs, m_iip);

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
        m_Eps_elas = m_quad_elas.allocate_qtensor<2>(0.0);
        m_Sig_elas = m_quad_elas.allocate_qtensor<2>(0.0);
        m_Eps_plas = m_quad_plas.allocate_qtensor<2>(0.0);
        m_Sig_plas = m_quad_plas.allocate_qtensor<2>(0.0);
        m_Epsdot_plas = m_quad_plas.allocate_qtensor<2>(0.0);

        m_M = GooseFEM::MatrixDiagonalPartitioned(m_conn, m_dofs, m_iip);
        m_D = GooseFEM::MatrixDiagonal(m_conn, m_dofs);

        m_material_elas = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem_elas, m_nip});
        m_material_plas = GMatElastoPlasticQPot::Cartesian2d::Array<2>({m_nelem_plas, m_nip});

        m_K_elas = GooseFEM::Matrix(m_conn_elas, m_dofs);
    }

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

protected:
    /**
    \todo Remove for new constructor.
    Set #m_allset = ``true`` if all prerequisites are satisfied.
    */
    void evalAllSet()
    {
        m_allset = m_set_M && (m_set_D || m_set_visco) && m_set_elas && m_set_plas && m_dt > 0.0;
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
    Set the mass density to a homogeneous quantity.
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
    To use a homogeneous system, use setRho().
    \tparam T e.g. `array_type::tensor<double, 1>`.
    \param val_elem Density per element.
    */
    template <class T>
    void setMassMatrix(const T& val_elem)
    {
        FRICTIONQPOTFEM_ASSERT(!m_set_M);
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
        m_set_M = true;
        this->evalAllSet();
    }

public:
    /**
    Set the value of the damping at the interface.
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
    Set background damping density (proportional to the velocity),
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
    Set damping matrix, based on certain density (taken uniform per element).
    Note that you can specify either setEta() or setDampingMatrix() or both.

    \param val_elem Damping per element.
    */
    template <class T>
    void setDampingMatrix(const T& val_elem)
    {
        FRICTIONQPOTFEM_ASSERT(!m_set_D);
        FRICTIONQPOTFEM_ASSERT(val_elem.size() == m_nelem);

        if (xt::allclose(val_elem, val_elem(0))) {
            m_alpha = val_elem(0);
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
        m_set_D = true;
        this->evalAllSet();
    }

public:
    /**
    Set material parameters for the elastic elements
    (taken uniform per element, ordering the same as in the constructor).

    \tparam S e.g. `array_type::tensor<double, 1>`.
    \tparam T e.g. `array_type::tensor<double, 1>`.
    \param K_elem Bulk modulus per element.
    \param G_elem Bulk modulus per element.
    */
    template <class S, class T>
    void setElastic(const S& K_elem, const T& G_elem)
    {
        FRICTIONQPOTFEM_ASSERT(!m_set_elas || m_nelem_elas == 0);
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(K_elem, {m_nelem_elas}));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(G_elem, {m_nelem_elas}));

        if (m_nelem_elas > 0) {
            array_type::tensor<size_t, 2> I = xt::ones<size_t>({m_nelem_elas, m_nip});
            array_type::tensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem_elas, m_nip});

            xt::view(idx, xt::range(0, m_nelem_elas), xt::all()) =
                xt::arange<size_t>(m_nelem_elas).reshape({-1, 1});

            m_material_elas.setElastic(I, idx, K_elem, G_elem);
            m_material_elas.setStrain(m_Eps_elas);

            FRICTIONQPOTFEM_REQUIRE(xt::all(xt::not_equal(
                m_material_elas.type(), GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));

            m_K_elas.assemble(
                m_quad_elas.Int_gradN_dot_tensor4_dot_gradNT_dV(m_material_elas.Tangent()));
        }

        m_set_elas = true;
        this->evalAllSet();
    }

public:
    /**
    Set material parameters for the plastic elements
    (taken uniform per element, ordering the same as in the constructor).

    \tparam S e.g. `array_type::tensor<double, 1>`.
    \tparam T e.g. `array_type::tensor<double, 1>`.
    \tparam Y e.g. `array_type::tensor<double, 2>`.
    \param K_elem Bulk modulus per element.
    \param G_elem Bulk modulus per element.
    \param epsy_elem Yield history per element.
    */
    template <class S, class T, class Y>
    void setPlastic(const S& K_elem, const T& G_elem, const Y& epsy_elem)
    {
        FRICTIONQPOTFEM_ASSERT(!m_set_plas || m_nelem_plas == 0);
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(K_elem, {m_nelem_plas}));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(G_elem, {m_nelem_plas}));
        FRICTIONQPOTFEM_ASSERT(epsy_elem.dimension() == 2);
        FRICTIONQPOTFEM_ASSERT(epsy_elem.shape(0) == m_nelem_plas);

        if (m_nelem_plas > 0) {
            array_type::tensor<size_t, 2> I = xt::ones<size_t>({m_nelem_plas, m_nip});
            array_type::tensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem_plas, m_nip});

            xt::view(idx, xt::range(0, m_nelem_plas), xt::all()) =
                xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});

            m_material_plas.setCusp(I, idx, K_elem, G_elem, epsy_elem);
            m_material_plas.setStrain(m_Eps_plas);

            FRICTIONQPOTFEM_REQUIRE(xt::all(xt::not_equal(
                m_material_plas.type(), GMatElastoPlasticQPot::Cartesian2d::Type::Unset)));
        }

        m_set_plas = true;
        this->evalAllSet();
    }

public:
    /**
    Reset yield strains (to avoid re-construction).
    \tparam T e.g. `array_type::tensor<double, 2>`.
    \param epsy_elem Yield history per element.
    */
    template <class T>
    void reset_epsy(const T& epsy_elem)
    {
        FRICTIONQPOTFEM_ASSERT(m_set_plas);
        FRICTIONQPOTFEM_ASSERT(epsy_elem.dimension() == 2);
        FRICTIONQPOTFEM_ASSERT(epsy_elem.shape(0) == m_nelem_plas);

        if (m_nelem_plas > 0) {
            array_type::tensor<size_t, 2> I = xt::ones<size_t>({m_nelem_plas, m_nip});
            array_type::tensor<size_t, 2> idx = xt::zeros<size_t>({m_nelem_plas, m_nip});

            xt::view(idx, xt::range(0, m_nelem_plas), xt::all()) =
                xt::arange<size_t>(m_nelem_plas).reshape({-1, 1});

            m_material_plas.reset_epsy(I, idx, epsy_elem);
        }
    }

public:
    /**
    Get the current yield strains per plastic element.
    Note that in this system the yield strains history is always the same for all the integration
    points in the system.
    \return [plastic().size, n]
    */
    array_type::tensor<double, 2> epsy() const
    {
        using S = typename array_type::tensor<double, 2>::size_type;
        array_type::tensor<double, 2> ret;

        for (size_t e = 0; e < m_nelem_plas; ++e) {
            auto cusp = m_material_plas.crefCusp({e, 0});
            auto val = cusp.epsy();
            if (e == 0) {
                std::array<S, 2> shape = {static_cast<S>(m_nelem_plas), static_cast<S>(val.size())};
                ret.resize(shape);
            }
            std::copy(val.cbegin(), val.cend(), &ret(e, xt::missing));
        }

        return ret;
    }

public:
    /**
    Check if elasticity is homogeneous.

    \return ``true`` is elasticity is homogeneous (``false`` otherwise).
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
    Set time step. Using for example in System::timeStep and System::minimise.
    */
    void setDt(double dt)
    {
        m_dt = dt;
        this->evalAllSet();
    }

public:
    /**
    Set nodal displacements.
    Internally, this updates the relevant forces using updated_u().

    \param u ``[nnode, ndim]``.
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
    Set nodal velocities.
    Internally, this updates the relevant forces using updated_v().

    \param v ``[nnode, ndim]``.
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
    Set nodal accelerations.

    \param a ``[nnode, ndim]``.
    */
    template <class T>
    void setA(const T& a)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(a, {m_nnode, m_ndim}));
        xt::noalias(m_a) = a;
    }

protected:
    /**
    Evaluate relevant forces when m_u is updated.
    */
    virtual void updated_u()
    {
        this->computeForceMaterial();
    }

protected:
    /**
    Evaluate relevant forces when m_v is updated.
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
    Set external force.
    Note: the external force on the DOFs whose displacement are prescribed are response forces
    computed during timeStep(). Internally on the system of unknown DOFs is solved, so any
    change to the response forces is ignored.

    \param fext ``[nnode, ndim]``.
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
    const auto& elastic() const
    {
        return m_elem_elas;
    }

public:
    /**
    List of plastic elements. Shape: [System::m_nelem_plas].

    \return List of element numbers.
    */
    const auto& plastic() const
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
        return m_conn;
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
        return m_dofs;
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
    \return Nodal force. Shape ``[nnode, ndim]``    .
    */
    const auto& fext()
    {
        this->computeInternalExternalResidualForce();
        return m_fext;
    }

public:
    /**
    Internal force.
    \return Nodal force. Shape ``[nnode, ndim]``    .
    */
    const auto& fint()
    {
        this->computeInternalExternalResidualForce();
        return m_fint;
    }

public:
    /**
    Force deriving from elasticity.

    \return Nodal force. Shape ``[nnode, ndim]``    .
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

    \return Nodal force. Shape ``[nnode, ndim]``    .
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
        return m_t;
    }

    /**
    Set time.
    */
    void setT(double t)
    {
        m_t = t;
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
    GooseFEM vector definition.
    Takes care of bookkeeping.

    \return GooseFEM::VectorPartitioned (System::m_vector)
    */
    const GooseFEM::VectorPartitioned& vector() const
    {
        return m_vector;
    }

public:
    /**
    GooseFEM quadrature definition.
    Takes case of interpolation, and taking gradient and integrating.

    \return GooseFEM::Element::Quad4::Quadrature (System::m_quad)
    */
    const GooseFEM::Element::Quad4::Quadrature& quad() const
    {
        return m_quad;
    }

public:
    /**
    GMatElastoPlasticQPot Array definition for the elastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (#m_material_elas).
    */
    const GMatElastoPlasticQPot::Cartesian2d::Array<2>& material_elastic() const
    {
        return m_material_elas;
    }

public:
    /**
    GMatElastoPlasticQPot Array definition for the plastic elements.

    \return GMatElastoPlasticQPot::Cartesian2d::Array<2>" (#m_material_plas).
    */
    const GMatElastoPlasticQPot::Cartesian2d::Array<2>& material_plastic() const
    {
        return m_material_plas;
    }

public:
    /**
    Bulk modulus per integration point.

    \return Integration point scalar. Shape: ``[nelem, nip]``.
    */
    array_type::tensor<double, 2> K() const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>({m_nelem, m_nip});

        auto ret_elas = m_material_elas.K();
        auto ret_plas = m_material_plas.K();
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

    \return Integration point scalar. Shape: ``[nelem, nip]``.
    */
    array_type::tensor<double, 2> G() const
    {
        array_type::tensor<double, 2> ret = xt::empty<double>({m_nelem, m_nip});

        auto ret_elas = m_material_elas.G();
        auto ret_plas = m_material_plas.G();
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

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    const array_type::tensor<double, 4>& Sig()
    {
        this->computeEpsSig();
        return m_Sig;
    }

public:
    /**
    Strain tensor of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    const array_type::tensor<double, 4>& Eps()
    {
        this->computeEpsSig();
        return m_Eps;
    }

public:
    /**
    Strain-rate tensor (the symmetric gradient of the nodal velocities) of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    array_type::tensor<double, 4> Epsdot() const
    {
        return m_quad.SymGradN_vector(m_vector.AsElement(m_v));
    }

public:
    /**
    Symmetric gradient of the nodal accelerations of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2]``.
    */
    array_type::tensor<double, 4> Epsddot() const
    {
        return m_quad.SymGradN_vector(m_vector.AsElement(m_a));
    }

public:
    /**
    Stiffness tensor of each integration point.

    \return Integration point tensor. Shape: ``[nelem, nip, 2, 2, 2, 2]``.
    */
    GooseFEM::MatrixPartitioned stiffness() const
    {
        auto ret = m_quad.allocate_qtensor<4>(0.0);
        auto ret_plas = m_material_plas.Tangent();
        auto ret_elas = m_material_elas.Tangent();
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

        GooseFEM::MatrixPartitioned K(m_conn, m_dofs, m_iip);
        K.assemble(m_quad.Int_gradN_dot_tensor4_dot_gradNT_dV(ret));
        return K;
    }

public:
    /**
    Stress tensor of integration points of plastic elements only, see System::plastic.

    \return Integration point tensor. Shape: [plastic().size(), nip, 2, 2].
    */
    const array_type::tensor<double, 4>& plastic_Sig() const
    {
        return m_Sig_plas;
    }

public:
    /**
    Strain tensor of integration points of plastic elements only, see System::plastic.

    \return Integration point tensor. Shape: [plastic().size(), nip, 2, 2].
    */
    const array_type::tensor<double, 4>& plastic_Eps() const
    {
        return m_Eps_plas;
    }

public:
    /**
    Strain-rate tensor of integration points of plastic elements only, see System::plastic.

    \return Integration point tensor. Shape: [plastic().size(), nip, 2, 2].
    */
    array_type::tensor<double, 4> plastic_Epsdot()
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
    Current yield strain left (in the negative equivalent strain direction).

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    array_type::tensor<double, 2> plastic_CurrentYieldLeft() const
    {
        return m_material_plas.CurrentYieldLeft();
    }

public:
    /**
    Current yield strain right (in the positive equivalent strain direction).

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    array_type::tensor<double, 2> plastic_CurrentYieldRight() const
    {
        return m_material_plas.CurrentYieldRight();
    }

public:
    /**
    Yield strain at an offset to the current yield strain left
    (in the negative equivalent strain direction).
    If ``offset = 0`` the result is the same result as the basic System::plastic_CurrentYieldLeft.

    \param offset Offset (number of yield strains).
    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    array_type::tensor<double, 2> plastic_CurrentYieldLeft(size_t offset) const
    {
        return m_material_plas.CurrentYieldLeft(offset);
    }

public:
    /**
    Yield strain at an offset to the current yield strain right
    (in the positive equivalent strain direction).
    If ``offset = 0`` the result is the same result as the basic System::plastic_CurrentYieldRight.

    \param offset Offset (number of yield strains).
    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    array_type::tensor<double, 2> plastic_CurrentYieldRight(size_t offset) const
    {
        return m_material_plas.CurrentYieldRight(offset);
    }

public:
    /**
    Current index in the landscape.

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    array_type::tensor<size_t, 2> plastic_CurrentIndex() const
    {
        return m_material_plas.CurrentIndex();
    }

public:
    /**
    Plastic strain.

    \return Integration point scalar. Shape: [plastic().size(), nip].
    */
    array_type::tensor<double, 2> plastic_Epsp() const
    {
        return m_material_plas.Epsp();
    }

public:
    /**
    Check that the current yield-index is at least `n` away from the end.
    \param n Margin.
    \return `true` if the current yield-index is at least `n` away from the end.
    */
    bool boundcheck_right(size_t n) const
    {
        return m_material_plas.checkYieldBoundRight(n);
    }

protected:
    /**
    Compute #m_Sig and #m_Eps using #m_u.
    */
    void computeEpsSig()
    {
        FRICTIONQPOTFEM_ASSERT(m_allset);
        if (!m_full_outdated) {
            return;
        }

        m_vector_elas.asElement(m_u, m_ue_elas);
        m_quad_elas.symGradN_vector(m_ue_elas, m_Eps_elas);
        m_material_elas.setStrain(m_Eps_elas);
        m_material_elas.stress(m_Sig_elas);

        size_t n = xt::strides(m_Eps_elas, 0);
        FRICTIONQPOTFEM_ASSERT(n == m_nip * 4);

        for (size_t e = 0; e < m_nelem_elas; ++e) {
            std::copy(
                &m_Eps_elas(e, xt::missing),
                &m_Eps_elas(e, xt::missing) + n,
                &m_Eps(m_elem_elas(e), xt::missing));
            std::copy(
                &m_Sig_elas(e, xt::missing),
                &m_Sig_elas(e, xt::missing) + n,
                &m_Sig(m_elem_elas(e), xt::missing));
        }
        for (size_t e = 0; e < m_nelem_plas; ++e) {
            std::copy(
                &m_Eps_plas(e, xt::missing),
                &m_Eps_plas(e, xt::missing) + n,
                &m_Eps(m_elem_plas(e), xt::missing));
            std::copy(
                &m_Sig_plas(e, xt::missing),
                &m_Sig_plas(e, xt::missing) + n,
                &m_Sig(m_elem_plas(e), xt::missing));
        }

        m_full_outdated = false;
    }

    /**
    Update #m_fmaterial based on the current displacement field #m_u.
    This implies computing #m_Eps_plas and #m_Sig_plas.
    */
    void computeForceMaterial()
    {
        FRICTIONQPOTFEM_ASSERT(m_allset);
        m_full_outdated = true;

        m_vector_plas.asElement(m_u, m_ue_plas);
        m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);
        m_material_plas.setStrain(m_Eps_plas);
        m_material_plas.stress(m_Sig_plas);
        m_quad_plas.int_gradN_dot_tensor2_dV(m_Sig_plas, m_fe_plas);
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
        m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);
        m_pert_Epsd_plastic = GMatElastoPlasticQPot::Cartesian2d::Deviatoric(m_Eps_plas);

        m_vector_plas.asElement(m_u, m_ue_plas);
        m_quad_plas.symGradN_vector(m_ue_plas, m_Eps_plas);

        if (!autoscale) {
            return 1.0;
        }

        auto deps = xt::amax(GMatElastoPlasticQPot::Cartesian2d::Epsd(m_pert_Epsd_plastic))();
        auto d = xt::amin(xt::diff(this->epsy(), 1))();
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
        bool iterative = false)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(m_pert_u, m_u.shape()));
        FRICTIONQPOTFEM_ASSERT(direction == 1 || direction == -1);

        auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
        auto idx = this->plastic_CurrentIndex();
        auto epsy_l = this->plastic_CurrentYieldLeft();
        auto epsy_r = this->plastic_CurrentYieldRight();

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
            auto d = target - GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps());
            if (direction > 0 && xt::any(d < 0.0)) {
                return 0.0;
            }
            if (direction < 0 && xt::any(d > 0.0)) {
                return 0.0;
            }
        }

        auto Epsd_plastic = GMatElastoPlasticQPot::Cartesian2d::Deviatoric(this->plastic_Eps());
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
            auto u_n = this->u();
            auto jdx = xt::cast<long>(this->plastic_CurrentIndex());
            size_t i;
            long nip = static_cast<long>(m_nip);

            // find first step that:
            // if (!kick || (kick && !yield_element)): is plastic for the first time
            // if (kick && yield_element): yields the element for the first time

            for (i = 0; i < steps.size(); ++i) {
                this->setU(u_n + dir * steps(i) * m_pert_u);
                auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
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
                auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
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
                auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
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
                auto jdx_new = xt::cast<long>(this->plastic_CurrentIndex());
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

        auto idx_new = this->plastic_CurrentIndex();
        FRICTIONQPOTFEM_REQUIRE(kick || xt::all(xt::equal(idx, idx_new)));
        FRICTIONQPOTFEM_REQUIRE(!kick || xt::any(xt::not_equal(idx, idx_new)));

        if (!iterative) {
            auto eps_new = GMatElastoPlasticQPot::Cartesian2d::Epsd(this->plastic_Eps())(e, q);
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
        FRICTIONQPOTFEM_ASSERT(m_allset);

        // history

        m_t += m_dt;
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
    */
    void timeSteps(size_t n)
    {
        for (size_t i = 0; i < n; ++i) {
            this->timeStep();
        }
    }

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
    size_t timeSteps_residualcheck(size_t n, double tol = 1e-5, size_t niter_tol = 20)
    {
        FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        for (size_t i = 0; i < n; ++i) {

            this->timeStep();

            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return 0;
            }
        }

        return n;
    }

    /**
    \copydoc timeSteps(size_t)

    This function stops if the yield-index in any of the plastic elements is close the end.
    In that case the function returns zero, in all other cases the function returns a
    positive number.

    \param nmargin Number of potentials to leave as margin.
    */
    size_t timeSteps_boundcheck(size_t n, size_t nmargin = 5)
    {
        if (!this->boundcheck_right(nmargin)) {
            return 0;
        }

        for (size_t i = 0; i < n; ++i) {
            this->timeStep();

            if (!this->boundcheck_right(nmargin)) {
                return 0;
            }
        }

        return n;
    }

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
    size_t timeStepsUntilEvent(double tol = 1e-5, size_t niter_tol = 20, size_t max_iter = 10000000)
    {
        FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
        size_t iiter;
        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        auto idx_n = this->plastic_CurrentIndex();

        for (iiter = 1; iiter < max_iter + 1; ++iiter) {

            this->timeStep();

            auto idx = this->plastic_CurrentIndex();

            if (xt::any(xt::not_equal(idx, idx_n))) {
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
    (1) Add a displacement \f$ \underline{v} \Delta t \f$ to each of the nodes.
    (2) Make a timeStep().

    \param n Number of steps to make.
    \param v Nodal velocity to add ``[nnode, ndim]``.
    */
    template <class T>
    void flowSteps(size_t n, const T& v)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, m_u.shape()));

        for (size_t i = 0; i < n; ++i) {
            m_u += v * m_dt;
            this->timeStep();
        }
    }

    /**
    \copydoc flowSteps(size_t, const T&)

    This function stops if the yield-index in any of the plastic elements is close the end.
    In that case the function returns zero, in all other cases the function returns a
    positive number.

    \param nmargin
        Number of potentials to leave as margin.
    */
    template <class T>
    size_t flowSteps_boundcheck(size_t n, const T& v, size_t nmargin = 5)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(v, m_u.shape()));

        if (!this->boundcheck_right(nmargin)) {
            return 0;
        }

        for (size_t i = 0; i < n; ++i) {
            m_u += v * m_dt;
            this->timeStep();

            if (!this->boundcheck_right(nmargin)) {
                return 0;
            }
        }

        return n;
    }

    /**
    Minimise energy: run System::timeStep until a mechanical equilibrium has been reached.

    \param tol Relative force tolerance for equilibrium. See System::residual for definition.
    \param niter_tol Enforce the residual check for ``niter_tol`` consecutive increments.
    \param max_iter Maximum number of iterations. Throws ``std::runtime_error`` otherwise.

    \return The number of iterations.
    */
    size_t minimise(double tol = 1e-5, size_t niter_tol = 20, size_t max_iter = 10000000)
    {
        FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

            this->timeStep();
            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return iiter;
            }
        }

        std::cout << "residuals = " << xt::adapt(residuals.data()) << std::endl;
        bool converged = false;
        FRICTIONQPOTFEM_REQUIRE(converged == true);
        return 0; // irrelevant, the code never goes here
    }

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
        size_t max_iter = 10000000)
    {
        FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

            this->timeStep();
            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return iiter;
            }

            if (!this->boundcheck_right(nmargin)) {
                return 0;
            }
        }

        std::cout << "residuals = " << xt::adapt(residuals.data()) << std::endl;
        bool converged = false;
        FRICTIONQPOTFEM_REQUIRE(converged == true);
        return 0; // irrelevant, the code never goes here
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
        size_t max_iter = 10000000)
    {
        FRICTIONQPOTFEM_REQUIRE(xt::has_shape(idx_n, std::array<size_t, 1>{m_N}));
        FRICTIONQPOTFEM_REQUIRE(S_truncate == 0);
        FRICTIONQPOTFEM_REQUIRE(A_truncate > 0);
        FRICTIONQPOTFEM_REQUIRE(tol < 1.0);
        double tol2 = tol * tol;
        GooseFEM::Iterate::StopList residuals(niter_tol);

        for (size_t iiter = 1; iiter < max_iter + 1; ++iiter) {

            this->timeStep();
            residuals.roll_insert(this->residual());

            if ((residuals.descending() && residuals.all_less(tol)) || residuals.all_less(tol2)) {
                this->quench();
                return iiter;
            }

            array_type::tensor<size_t, 1> idx =
                xt::view(this->plastic_CurrentIndex(), xt::all(), 0);

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
        size_t max_iter = 10000000)
    {
        array_type::tensor<size_t, 1> idx_n = xt::view(this->plastic_CurrentIndex(), xt::all(), 0);
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
        size_t ll = m_conn(m_elem_plas(0), 0);
        size_t ul = m_conn(m_elem_plas(0), 3);
        double y0 = (m_coor(ul, 1) + m_coor(ll, 1)) / 2.0;

        for (size_t n = 0; n < m_nnode; ++n) {
            ret(n, 0) += 2.0 * delta_gamma * (m_coor(n, 1) - y0);
        }

        return ret;
    }

protected:
    array_type::tensor<size_t, 2> m_conn; ///< Connectivity, see conn().
    array_type::tensor<size_t, 2> m_conn_elas; ///< Slice of #m_conn for elastic elements.
    array_type::tensor<size_t, 2> m_conn_plas; ///< Slice of #m_conn for plastic elements.
    array_type::tensor<double, 2> m_coor; ///< Nodal coordinates, see coor().
    array_type::tensor<size_t, 2> m_dofs; ///< DOFs, shape: [#m_nnode, #m_ndim].
    array_type::tensor<size_t, 1> m_iip; ///< Fixed DOFs.
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
    GMatElastoPlasticQPot::Cartesian2d::Array<2> m_material_elas; ///< Material for elastic el.
    GMatElastoPlasticQPot::Cartesian2d::Array<2> m_material_plas; ///< Material for plastic el.

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
    array_type::tensor<double, 3>
        m_ue_elas; ///< El. vector for elastic elements (used for displacements).
    array_type::tensor<double, 3> m_fe_elas; ///< El. vector for plastic elements (used for forces).
    array_type::tensor<double, 3>
        m_ue_plas; ///< El. vector for elastic elements (used for displacements).
    array_type::tensor<double, 3> m_fe_plas; ///< El. vector for plastic elements (used for forces).
    array_type::tensor<double, 2> m_fmaterial; ///< Nodal force, deriving from elasticity.
    array_type::tensor<double, 2>
        m_felas; ///< Nodal force, deriving from elasticity of elastic elements.
    array_type::tensor<double, 2>
        m_fplas; ///< Nodal force, deriving from elasticity of plastic elements.
    array_type::tensor<double, 2> m_fdamp; ///< Nodal force, deriving from damping.
    array_type::tensor<double, 2> m_fvisco; ///< Nodal force, deriving from damping at the interface
    array_type::tensor<double, 2> m_ftmp; ///< Temporary for internal use.
    array_type::tensor<double, 2> m_fint; ///< Nodal force: total internal force.
    array_type::tensor<double, 2> m_fext; ///< Nodal force: total external force (reaction force)
    array_type::tensor<double, 2> m_fres; ///< Nodal force: residual force.
    array_type::tensor<double, 4> m_Eps; ///< Integration point tensor: strain.
    array_type::tensor<double, 4> m_Sig; ///< Integration point tensor: stress.
    array_type::tensor<double, 4>
        m_Eps_elas; ///< Integration point tensor: strain for elastic elements.
    array_type::tensor<double, 4>
        m_Eps_plas; ///< Integration point tensor: strain for plastic elements.
    array_type::tensor<double, 4>
        m_Sig_elas; ///< Integration point tensor: stress for elastic elements.
    array_type::tensor<double, 4>
        m_Sig_plas; ///< Integration point tensor: stress for plastic elements.
    array_type::tensor<double, 4>
        m_Epsdot_plas; ///< Integration point tensor: strain-rate for plastic el.
    GooseFEM::Matrix m_K_elas; ///< Stiffness matrix for elastic elements only.
    double m_t = 0.0; ///< Current time.
    double m_dt = 0.0; ///< Time-step.
    double m_eta = 0.0; ///< Damping at the interface
    double m_rho = 0.0; ///< Mass density (non-zero only if homogeneous).
    double m_alpha = 0.0; ///< Background damping density (non-zero only if homogeneous).
    bool m_allset = false; ///< Internal allocation check.
    bool m_set_M = false; ///< Internal allocation check: mass matrix was written.
    bool m_set_D = false; ///< Internal allocation check: damping matrix was written.
    bool m_set_visco = false; ///< Internal allocation check: interfacial damping specified.
    bool m_set_elas = false; ///< Internal allocation check: elastic elements were written.
    bool m_set_plas = false; ///< Internal allocation check: plastic elements were written.
    bool m_full_outdated = true; ///< Keep track of the need to recompute fields on full geometry.
    array_type::tensor<double, 2> m_pert_u; ///< See eventDriven_setDeltaU()
    array_type::tensor<double, 4> m_pert_Epsd_plastic; ///< Strain deviator for #m_pert_u.
};

} // namespace Generic2d
} // namespace FrictionQPotFEM

#endif
