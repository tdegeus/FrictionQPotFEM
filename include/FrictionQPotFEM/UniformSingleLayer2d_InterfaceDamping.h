/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_INTERFACEDAMPING_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_INTERFACEDAMPING_H

#include "UniformSingleLayer2d.h"
#include "config.h"
#include "version.h"

/**
\return x^2
*/
#define SQR(x) ((x) * (x))

namespace FrictionQPotFEM {

/**
System in 2-d with:

-   A weak, middle, layer.
-   Uniform elasticity.
-   Damping at the interface and/or damping in the bulk.
*/
namespace UniformSingleLayer2d_InterfaceDamping {

/**
\copydoc Generic2d::version_dependencies()
*/
inline std::vector<std::string> version_dependencies();

/**
Class that uses GMatElastoPlasticQPot to evaluate stress everywhere
*/
class System : public UniformSingleLayer2d::System {

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

    std::string type() const override;

    void evalAllSet() override;

    void updated_v() override;

    /**
    Set the value of the damping at the interface,
    \param eta Damping parameter
    */
    void setEta(double eta);

    /**
    Force deriving from damping at the interface only.
    Note that fdamp() is the total damping force, which differs from `fvisco` if global damping
    was set, using setDampingMatrix().

    \return Nodal force. Shape ``[nnode, ndim]``.
    */
    auto fvisco() const;

protected:

    xt::xtensor<double, 4> m_Epsdot_plas; ///< Integration point tensor: strain-rate for plastic elem.
    double m_eta = 0.0; ///< Damping at the interface
    xt::xtensor<double, 2> m_visco; ///< Nodal force, deriving from damping at the interface
    bool m_set_visco = false; ///< Internal allocation check: interfacial damping specified.

};

} // namespace UniformSingleLayer2d_InterfaceDamping
} // namespace FrictionQPotFEM

#include "UniformSingleLayer2d_InterfaceDamping.hpp"

#endif
