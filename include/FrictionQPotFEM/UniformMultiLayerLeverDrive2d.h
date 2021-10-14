/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_H
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_H

#include "UniformMultiLayerIndividualDrive2d.h"
#include "config.h"
#include "version.h"

namespace FrictionQPotFEM {

/**
System in 2-d with:

-   Several weak layers.
-   Layers connected to a lever, that is driven through a spring.
-   Uniform elasticity.
*/
namespace UniformMultiLayerLeverDrive2d {

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
Similar to UniformMultiLayerIndividualDrive2d::System(), but with the difference that the
target average displacement per layer is not individually prescribed, but determined by
a lever, whose position (whose rotation to be precise) is determined by a driving spring
attached to it.
The assumption is made that the lever has no inertia:
its position is computed by assuming that the sum of moments acting on it is zero.
*/
class System : public UniformMultiLayerIndividualDrive2d::System {

public:
    System() = default;

    virtual ~System(){};

    /**
    Define basic geometry.

    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem Elements per layer.
    \param node Nodes per layer.
    \param layer_is_plastic Per layer set if elastic (= 0) or plastic (= 1).
    */
    System(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const std::vector<xt::xtensor<size_t, 1>>& elem,
        const std::vector<xt::xtensor<size_t, 1>>& node,
        const xt::xtensor<bool, 1>& layer_is_plastic);

    std::string type() const override;

    /**
    Set the lever properties.

    \tparam T e.g. `xt::xtensor<double, 1>`

    \param H The height of the spring pulling the lever.
    \param hi The height \f$ h_i \f$ of the loading frame of each layer [nlayer].
    */
    template <class T>
    void setLeverProperties(double H, const T& hi);

    /**
    Set the target 'position' of the spring pulling the lever.

    \param u Lever position
    */
    void setLeverTarget(double u);

    /**
    Get the current target lever 'position'.

    \return double
    */
    double leverTarget() const;

    /**
    Get the current lever 'position'.

    \return double
    */
    double leverPosition() const;

protected:
    /**
    Define basic geometry.
    This function class UniformMultiLayerIndividualDrive2d::init().

    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem Elements per layer.
    \param node Nodes per layer.
    \param layer_is_plastic Per layer set if elastic (= 0) or plastic (= 1).
    */
    void init_lever(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const std::vector<xt::xtensor<size_t, 1>>& elem,
        const std::vector<xt::xtensor<size_t, 1>>& node,
        const xt::xtensor<bool, 1>& layer_is_plastic);

    /**
    Evaluate relevant forces when m_u is updated.
    The assumption is made that the lever has no inertia:
    its position is computed by assuming that the sum of moments acting on it is zero.
    */
    void updated_u() override;

private:
    double m_lever_H; ///< See setLeverProperties().
    xt::xtensor<double, 1> m_lever_hi; ///< See setLeverProperties().
    xt::xtensor<double, 1> m_lever_hi_H; ///< m_lever_hi / H
    xt::xtensor<double, 1> m_lever_hi_H_2; ///< (m_lever_hi / H)^2
    double m_lever_target; ///< See setLeverTarget().
    double m_lever_u; ///< Current position of the lever.
};

} // namespace UniformMultiLayerLeverDrive2d
} // namespace FrictionQPotFEM

#include "UniformMultiLayerLeverDrive2d.hpp"

#endif
