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
class System : public Generic2d::HybridSystem {

public:

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
    */
    System(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const std::vector<xt::xtensor<size_t, 1>>& elem,
        const std::vector<xt::xtensor<size_t, 1>>& node,
        const xt::xtensor<bool, 1>& layer_is_plastic);

    /**
     * Return number of layers.
     * \return Number of layers.
     */
    size_t nlayer() const;

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
     *  Return if a layer is elastic (``false``) or plastic (``true``).
     *  \return [#nlayer].
     */
    xt::xtensor<bool, 1> layerIsPlastic() const;

    /**
    Set the stiffness of spring connecting the mean displacement of a layer to the drive frame.

    \param k The stiffness (taken the same for all layers)
    \param symmetric
        If `true` the spring is a normal spring.
        If `false` the spring has no stiffness under compression.
    */
    void setDriveStiffness(double k, bool symmetric = true);

    /**
    The mean displacement of each layer.
    Requires to recompute the average displacements as they are normally only computed on
    the driven DOFs.
    \return Average per layer [nlayer, ndim]
    */
    xt::xtensor<double, 2> layerUbar();

    /**
     *  Get the target mean displacement per layer.
     *  \return [#nlayer, 2]
     */
    xt::xtensor<double, 2> layerTargetUbar() const;

    /**
    Set the mean displacement per layer.

    \tparam S e.g. `xt::xtensor<double, 2>`
    \tparam T e.g. `xt::xtensor<bool, 2>`
    \param ubar The target mean position of each layer [nlayer, ndim].
    \param prescribe For each entry `ubar` set `true` to 'enforce' the position [nlayer, ndim].
    */
    template <class S, class T>
    void layerSetTargetUbar(const S& ubar, const T& prescribe);

    /**
    \copydoc layerSetTargetUbar(const S&, const T&)

    Note that the displacement is updated such that the mean of each prescribed layer is equal
    to the prescribed mean.
    */
    template <class S, class T>
    void layerSetTargetUbarAndDistribute(const S& ubar, const T& prescribe);

    /**
    Simple shear increment:
    -   Distributes the displacement uniformly.
    -   Updates the displacement of the (relevant) loading frames.

    Note that the height of each node is taken relative to `coor[0, 1]`.

    \tparam S e.g. `xt::xtensor<bool, 2>`
    \tparam T e.g. `xt::xtensor<double, 1>`
    \param delta_gamma Simple strain to add.
    \param prescribe For each entry `ubar` set `true` to 'enforce' the position [nlayer, ndim].
    \param height The height of the loading frame per layer [nlayer].
    */
    template <class S, class T>
    void addAffineSimpleShear(double delta_gamma, const S& prescribe, const T& height);

    /**
    Force related to the driving frame.
    \return nodevec [nnode, ndim].
    */
    xt::xtensor<double, 2> fdrive() const;

    /**
    Force of each of the driving springs
    \return [nlayer, ndim].
    */
    xt::xtensor<double, 2> layerFdrive() const;

protected:

    /**
    Compute force deriving from the drive.
    The force is applied as a force density for each of the layers for which the position
    was specified (at least once) using layerSetTargetUbar().
    */
    void computeForceDrive();

    /**
    Evaluate relevant forces when m_u is updated.
    */
    void updated_u() override;

    /**
    Compute:
    -   m_fint = m_fdrive + m_fmaterial + m_fdamp
    -   m_fext[iip] = m_fint[iip]
    -   m_fres = m_fext - m_fint

    Internal rule: all relevant forces are expected to be updated before this function is called.
    */
    void computeInternalExternalResidualForce() override;

    /**
    Constructor alias (as convenience for derived classes).

    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elem Elements per layer.
    \param node Nodes per layer.
    \param layer_is_plastic Per layer set if elastic (= 0) or plastic (= 1).
    */
    void init(
        const xt::xtensor<double, 2>& coor,
        const xt::xtensor<size_t, 2>& conn,
        const xt::xtensor<size_t, 2>& dofs,
        const xt::xtensor<size_t, 1>& iip,
        const std::vector<xt::xtensor<size_t, 1>>& elem,
        const std::vector<xt::xtensor<size_t, 1>>& node,
        const xt::xtensor<bool, 1>& layer_is_plastic);

protected:

    size_t m_n_layer; ///< Number of layers.
    std::vector<xt::xtensor<size_t, 1>> m_layer_node; ///< Nodes per layer.
    std::vector<xt::xtensor<size_t, 1>> m_layer_elem; ///< Elements per layer.
    xt::xtensor<bool, 1> m_layer_is_plastic; ///< Per layer ``true`` if the layer is plastic.
    xt::xtensor<size_t, 1> m_slice_index; ///< Per layer the index in m_slice_plas or m_slice_elas.
    xt::xtensor<size_t, 1> m_slice_plas; ///< How to slice elastic(): start and end index
    xt::xtensor<size_t, 1> m_slice_elas; ///< How to slice plastic(): start and end index

    bool m_drive_spring_symmetric = true; ///< If false the drive spring buckles in compression
    double m_k_drive = 1.0; ///< Stiffness of the drive control frame
    xt::xtensor<bool, 2> m_layer_ubar_set; ///< See `prescribe` in layerSetTargetUbar()
    xt::xtensor<double, 2> m_layer_ubar_target; ///< Per layer, the prescribed mean position.
    xt::xtensor<double, 2> m_layer_ubar_value; ///< Per layer, the prescribed mean position.
    xt::xtensor<double, 2> m_fdrive; ///< Force related to driving frame
    xt::xtensor<double, 2> m_layer_dV1; ///< volume per layer (as vector, same for all dimensions)
    xt::xtensor<double, 2> m_dV; ///< copy of m_quad.dV()
    xt::xtensor<double, 3> m_uq; ///< qvector
    xt::xtensor<double, 1> m_ud; ///< dofval

};

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#include "UniformMultiLayerIndividualDrive2d.hpp"

#endif
