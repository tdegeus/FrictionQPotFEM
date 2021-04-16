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
    Set the stiffness of spring connecting the mean displacement of a layer to the drive frame.

    \param k The stiffness (taken the same for all layers)
    */
    void setDriveStiffness(double k);

    /**
    \return The mean displacement of each layer [nlayer, ndim].
    */
    xt::xtensor<double, 2> layerUbar() const;

    /**
    Set the mean displacement per layer.

    \param ubar The target mean position of each layer [nlayer, ndim].
    \param prescribe For each entry ``ubar`` set ``true`` to 'enforce' the position [nlayer, ndim].
    */
    void layerSetUbar(const xt::xtensor<double, 2>& ubar, const xt::xtensor<bool, 2> prescribe);

    /**
    \return Force related to the driving frame.
    */
    xt::xtensor<double, 2> fdrive() const;

protected:

    /**
    Compute force deriving from the drive.
    The force is applied as a force density for each of the layers for which the position
    was specified (at least once) using layerSetUbar().
    */
    void computeForceDrive();

    /**
    Evaluate relevant forces when m_u is updates.
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

    \copydoc System::System()
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
    xt::xtensor<bool, 1> m_layer_is_plastic; ///< Per layer ``true`` is the layer is plastic.
    xt::xtensor<size_t, 1> m_slice_index; ///< Per layer the index in m_slice_plas or m_slice_elas.
    xt::xtensor<size_t, 1> m_slice_plas; ///< How to slice elastic(): start and end index
    xt::xtensor<size_t, 1> m_slice_elas; ///< How to slice plastic(): start and end index

    double m_k_drive = 1.0; ///< Stiffness of the drive control frame
    xt::xtensor<bool, 2> m_layer_ubar_set; ///< See `prescribe` in layerSetUbar()
    xt::xtensor<double, 2> m_layer_ubar_target; ///< Per layer, the prescribed mean position.
    xt::xtensor<double, 2> m_layer_ubar_value; ///< Per layer, the prescribed mean position.
    xt::xtensor<double, 2> m_fdrive; ///< Force related to driving frame
    xt::xtensor<double, 2> m_layer_dV1; ///< volume per layer (same of all dimensions)
    xt::xtensor<double, 2> m_dV; ///< copy of m_quad.dV()
    xt::xtensor<double, 3> m_uq; ///< qvector

};

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#include "UniformMultiLayerIndividualDrive2d.hpp"

#endif
