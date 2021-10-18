/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_H
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_H

#include "Generic2d.h"
#include "config.h"
#include "version.h"

namespace FrictionQPotFEM {

/**
System in 2-d with:
-   Several weak layers.
-   Each layer driven independently through a spring.
-   Uniform elasticity.
*/
namespace UniformMultiLayerIndividualDrive2d {

/**
\copydoc Generic2d::version_dependencies()
*/
inline std::vector<std::string> version_dependencies();

/**
System that comprises several layers (elastic or plastic).
The average displacement of each layer can be coupled to a prescribed target value
using a linear spring (one spring per spatial dimension):
-   To set its stiffness use layerSetDriveStiffness().
-   Each spring can be switched on individually using layerSetTargetActive().
    By default all springs are inactive (their stiffness is effectively zero).

Terminology:
-   `ubar`: the average displacement per layer [#nlayer, 2],
            see layerUbar() and layerSetUbar().

-   `target_ubar`: the target average displacement per layer [#nlayer, 2],
                   see layerTargetUbar() and layerSetTargetUbar().

-   `target_active`: the average displacement per layer/DOF is only enforced if the spring
                     is active, see layerTargetActive() and layerSetTargetActive().
*/
class System : public Generic2d::HybridSystem {

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

    size_t N() const override;
    std::string type() const override;

    /**
    Return number of layers.
    \return Number of layers.
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
    Return if a layer is elastic (`false`) or plastic (`true`).
    \return [#nlayer].
    */
    xt::xtensor<bool, 1> layerIsPlastic() const;

    /**
    Set the stiffness of the springs connecting
    the average displacement of a layer ("ubar") to its set target value.
    Note that the stiffness of all springs is taken the same.

    \param k The stiffness.
    \param symmetric
        If `true` the spring is a normal spring.
        If `false` the spring has no stiffness under compression.
    */
    void layerSetDriveStiffness(double k, bool symmetric = true);

    /**
    Initialise the event driven protocol by applying a perturbation to the boundary conditions
    and computing and storing the linear, purely elastic, response.
    The system can thereafter be moved forward to the next event.
    Note that this function itself does not change the system in any way.

    \tparam S `xt::xtensor<double, 2>`
    \param delta_ubar
        The perturbation in the target average position of each layer [#nlayer, 2].

    \tparam T `xt::xtensor<bool, 2>`
    \param active
        For each layer and each degree-of-freedom specify if
        the spring is active (`true`) or not (`false`) [#nlayer, 2].
    */
    template <class S, class T>
    void initEventDriven(const S& delta_ubar, const T& active);

    /**
    \copydoc initEventDriven(const S&, const T&)

    \tparam U `xt::xtensor<double, 2>`
    \param delta_u
        Use a precomputed displacement field.
    */
    template <class S, class T, class U>
    void initEventDriven(const S& delta_ubar, const T& active, const U& delta_u);

    double eventDrivenStep(double deps, bool kick, int direction = +1) override;

    /**
    Turn on (or off) springs connecting
    the average displacement of a layer ("ubar") to its set target value.

    \tparam T e.g. `xt::xtensor<bool, 2>`
    \param active
        For each layer and each degree-of-freedom specify if
        the spring is active (`true`) or not (`false`) [#nlayer, 2].
    */
    template <class T>
    void layerSetTargetActive(const T& active);

    /**
    List the average displacement of each layer.
    Requires to recompute the average displacements
    (as they are normally only computed on the driven DOFs).

    \return Average displacement per layer [#nlayer, 2]
    */
    xt::xtensor<double, 2> layerUbar();

    /**
    List the target average displacement per layer.
    \return [#nlayer, 2]
    */
    xt::xtensor<double, 2> layerTargetUbar() const;

    /**
    List if the driving spring is activate.
    \return [#nlayer, 2]
    */
    xt::xtensor<bool, 2> layerTargetActive() const;

    /**
    Set the target average displacement per layer.
    Only layers that have an active driving spring will feel a force
    (if its average displacement is different from the target displacement),
    see layerSetTargetActive().

    \tparam T e.g. `xt::xtensor<double, 2>`
    \param ubar The target average position of each layer [#nlayer, 2].
    */
    template <class T>
    void layerSetTargetUbar(const T& ubar);

    /**
    Move the layer such that the average displacement is exactly equal to its input value.

    \tparam S e.g. `xt::xtensor<double, 2>`
    \tparam T e.g. `xt::xtensor<bool, 2>`

    \param ubar
        The target average position of each layer [#nlayer, 2].

    \param prescribe
        Per layers/degree-of-freedom specify if its average is modified [#nlayer, 2].
        Note that this not modify which of the driving springs is active or not,
        that can only be changed using layerSetTargetActive().
    */
    template <class S, class T>
    void layerSetUbar(const S& ubar, const T& prescribe);

    /**
    Add affine simple shear to all the nodes of the body.
    The displacement of the bottom boundary is zero,
    while it is maximal for the top boundary.
    The input is the strain increment,
    the increment of the shear component of deformation gradient is twice that.

    \param delta_gamma Strain increment.
    */
    void addAffineSimpleShear(double delta_gamma);

    /**
    Add simple shear to each of the target average displacements.
    In particular \f$ \bar{u}_x^i = 2 \Delta \gamma h_i \f$
    with \f$ \bar{u}_x^i \f$ the \f$ x \f$-component of the target average displacement
    of layer \f$ i \f$.

    \tparam T e.g. `xt::xtensor<double, 1>`

    \param delta_gamma Affine strain to add.
    \param height The height \f$ h_i \f$ of the loading frame of each layer [#nlayer].
    */
    template <class T>
    void layerTagetUbar_addAffineSimpleShear(double delta_gamma, const T& height);

    /**
    Nodal force induced by the driving springs.
    The only non-zero contribution comes from:
    -   springs that are active, see layerSetTargetActive(), and
    -   layers whose average displacement is different from its target value.

    \return nodevec [nnode, ndim].
    */
    xt::xtensor<double, 2> fdrive() const;

    /**
    Force of each of the driving springs.
    The only non-zero contribution comes from:
    -   springs that are active, see layerSetTargetActive(), and
    -   layers whose average displacement is different from its target value.

    \return [#nlayer, ndim].
    */
    xt::xtensor<double, 2> layerFdrive() const;

protected:
    /**
    Compute the average displacement of all layers with an active driving spring.
    */
    void computeLayerUbarActive();

    /**
    Compute force deriving from the activate springs between the average displacement of
    the layer and its target value.
    The force is applied as a force density.

    Internal rule: computeLayerUbarActive() is called before this function,
    if the displacement changed since the last time the average was computed.
    */
    void computeForceFromTargetUbar();

    /**
    Evaluate relevant forces when m_u is updated.
    */
    void updated_u() override;

    /**
    Compute:
    -   m_fint = m_fdrive + m_fmaterial + m_fdamp
    -   m_fext[iip] = m_fint[iip]
    -   m_fres = m_fext - m_fint

    Internal rule: all relevant forces are expected to be updated before this function is
    called.
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
    size_t m_N; ///< Linear system size.
    size_t m_n_layer; ///< Number of layers.
    std::vector<xt::xtensor<size_t, 1>> m_layer_node; ///< Nodes per layer.
    std::vector<xt::xtensor<size_t, 1>> m_layer_elem; ///< Elements per layer.
    xt::xtensor<bool, 1> m_layer_is_plastic; ///< Per layer ``true`` if the layer is plastic.
    xt::xtensor<size_t, 1> m_slice_index; ///< Per layer the index in m_slice_plas or m_slice_elas.
    xt::xtensor<size_t, 1> m_slice_plas; ///< How to slice elastic(): start and end index
    xt::xtensor<size_t, 1> m_slice_elas; ///< How to slice plastic(): start and end index

    bool m_drive_spring_symmetric = true; ///< If false the drive spring buckles in compression
    double m_drive_k = 1.0; ///< Stiffness of the drive control frame
    xt::xtensor<bool, 2> m_layerdrive_active; ///< See `prescribe` in layerSetTargetUbar()
    xt::xtensor<double, 2> m_layerdrive_targetubar; ///< Per layer, the prescribed average position.
    xt::xtensor<double, 2> m_layer_ubar; ///< See computeLayerUbarActive().
    xt::xtensor<double, 2> m_fdrive; ///< Force related to driving frame
    xt::xtensor<double, 2> m_layer_dV1; ///< volume per layer (as vector, same for all dimensions)
    xt::xtensor<double, 2> m_dV; ///< copy of m_quad.dV()
    xt::xtensor<double, 3> m_uq; ///< qvector
    xt::xtensor<double, 1> m_ud; ///< dofval

    xt::xtensor<bool, 2> m_pert_layerdrive_active; ///< Event driven: applied lever setting.
    xt::xtensor<double, 2> m_pert_layerdrive_targetubar; ///< Event driven: applied lever setting.
};

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#include "UniformMultiLayerIndividualDrive2d.hpp"

#endif
