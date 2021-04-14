/**
See FrictionQPotFEM::UniformSingleLayer2d.
Implementation in UniformSingleLayer2d.hpp.

\file UniformSingleLayer2d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_H
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_H

#include "config.h"
#include "version.h"
#include "Generic2d.h"

namespace FrictionQPotFEM {

/**
System in 2-d with:

-   Several weak layers.
-   Each layer driven independently through a spring.
-   Uniform elasticity.
*/
namespace UniformMultiLayerIndividualDrive2d {

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
class System : protected Generic2d::HybridSystem {

    System() = default;

    virtual ~System() {};

    /**
    Define basic geometry.

    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem Elements per layer.
    \param node Nodes per layer.
    \param layer_is_plastic Per layer set if elastic (= 0) or plastic (= 1).
    \param node_is_virtual Per node specify if it is virtual or not.
    */
    System(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const std::vector<xt::xtensor<size_t, 1>>& elem,
        const std::vector<xt::xtensor<size_t, 1>>& node,
        const xt::xtensor<bool, 1>& layer_is_plastic,
        const xt::xtensor<bool, 1>& node_is_virtual);

    /**
    Return the nodes belonging to the i-th layer.

    \param i Index of the layer.
    \return List of node numbers.
    */
    xt::xtensor<size_t, 1> layerNodes(size_t i) const;

    /**
    Return the elements belonging to the i-th layer.

    \param i Index of the layer.
    \return List of element numbers.
    */
    xt::xtensor<size_t, 1> layerElements(size_t i) const;

    /**
    Return if a layer is elastic (``false``) or plastic (``true``).

    \param i Index of the layer.
    \return boolean
    */
    bool layerIsPlastic(size_t i) const;

    /**
    Update the mean displacement of a layer.

    \param ubar Mean position of the i-th layer.
    \param i Index of the layer.
    */
    void layerSetUbar(double ubar, size_t i);

    /**
    Compute force deriving from the drive.
    The force is applied as a force density for each of the layers for which the position
    was specified (at least once) using layerSetUbar().
    */
    void computeForceDrive();

    /**
    Set nodal displacements.
    Internally, this updates:
    -   #m_fmaterial by calling computeForceMaterial().
    -   #m_fdrive by calling computeForceDrive().

    \param u ``[nnode, ndim]``.
    */
    void setU(const xt::xtensor<double, 2>& u) override;

    /**
    Make a time-step: apply velocity-Verlet integration.
    Forces are computed where needed as follows:

    -   After updating the displacement:

        -   #m_fmaterial by calling computeForceMaterial().
        -   #m_fdrive by calling computeForceDrive().

    -   After updating the velocity:

        -   #m_fdamp directly using #m_D
    */
    void timeStep() override;

protected:

    /**
    Constructor alias (as convenience for derived classes).

    \copydoc System::System()
    */
    void init(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const std::vector<xt::xtensor<size_t, 1>>& elem,
        const std::vector<xt::xtensor<size_t, 1>>& node,
        const xt::xtensor<bool, 1>& layer_is_plastic,
        const xt::xtensor<bool, 1>& node_is_virtual);

protected:

    size_t m_n_layer; ///< Number of layers.
    std::vector<xt::xtensor<size_t, 1>> m_layer_node; ///< Nodes per layer.
    std::vector<xt::xtensor<size_t, 1>> m_layer_elem; ///< Elements per layer.
    xt::xtensor<bool, 1> m_node_is_virtual; ///< Per node ``true`` is the layer is plastic.
    xt::xtensor<bool, 1> m_layer_is_plastic; ///< Per layer ``true`` is the layer is plastic.
    xt::xtensor<size_t, 1> m_slice_index; ///< Per layer the index in m_slice_plas or m_slice_elas.
    xt::xtensor<size_t, 1> m_slice_plas; ///< How to slice elastic(): start and end index
    xt::xtensor<size_t, 1> m_slice_elas; ///< How to slice plastic(): start and end index
    xt::xtensor<bool, 1> m_layer_has_drive; ///< Per layer ``true`` if the mean position is be controlled.
    xt::xtensor<double, 1> m_layer_ubar; ///< Per layer, the mean position.

};

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#include "UniformMultiLayerIndividualDrive2d.hpp"

#endif
