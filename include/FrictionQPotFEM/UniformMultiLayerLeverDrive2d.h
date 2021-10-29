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
\copydoc Generic2d::version_dependencies()
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

private:
    using UniformMultiLayerIndividualDrive2d::System::initEventDriven;

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

    \param xdrive Lever position
    */
    void setLeverTarget(double xdrive);

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

    /**
    Initialise the event driven protocol by applying a perturbation to loading spring
    and computing and storing the linear, purely elastic, response.
    The system can thereafter be moved forward to the next event.
    Note that this function itself does not change the system in any way,
    it just stores the relevant perturbations.

    \param xdrive Target 'position' of the spring pulling the lever.

    \param active
        For each layer and each degree-of-freedom specify if
        the spring is active (`true`) or not (`false`) [#nlayer, 2].
    \
    */
    template <class T>
    void initEventDriven(double xdrive, const T& active);

    /**
    Restore perturbation used from event driven protocol.
    \param xdrive See eventDriven_leverPosition().
    \param active See eventDriven_targetActive().
    \param delta_u See eventDriven_deltaU().
    \param delta_ubar See eventDriven_deltaUbar().
    \return Value with which the input perturbation is scaled, see also eventDriven_deltaU().
    */
    template <class T, class U, class W>
    double initEventDriven(double xdrive, const T& active, const U& delta_u, const W& delta_ubar);

    double eventDrivenStep(double deps, bool kick, int direction = +1, bool yield_element = false)
        override;

    /**
    Get target 'position' of the spring pulling the lever perturbation used for event driven code.
    \return Value
    */
    double eventDriven_deltaLeverPosition() const;

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
    bool m_lever_set = false; ///< Lock class until properties have been set.
    double m_lever_H; ///< See setLeverProperties().
    xt::xtensor<double, 1> m_lever_hi; ///< See setLeverProperties().
    xt::xtensor<double, 1> m_lever_hi_H; ///< m_lever_hi / H
    xt::xtensor<double, 1> m_lever_hi_H_2; ///< (m_lever_hi / H)^2
    double m_lever_target; ///< See setLeverTarget().
    double m_lever_u; ///< Current position of the lever.
    double m_pert_lever_target; ///< Perturbation in target position for event driven load.
};

} // namespace UniformMultiLayerLeverDrive2d
} // namespace FrictionQPotFEM

#include "UniformMultiLayerLeverDrive2d.hpp"

#endif
