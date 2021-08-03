/**
See FrictionQPotFEM::UniformMultiLayerLeverDrive2d.
Implementation in UniformMultiLayerLeverDrive2d.hpp.

\file UniformMultiLayerLeverDrive2d.h
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_H
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_H

#include "config.h"
#include "version.h"
#include "UniformMultiLayerIndividualDrive2d.h"

namespace FrictionQPotFEM {

/**
System in 2-d with:

-   Several weak layers.
-   Layers connected to a lever, that is driven throught a spring.
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
Class that uses GMatElastoPlasticQPot to evaluate stress everywhere
*/
class System : public UniformMultiLayerIndividualDrive2d::System {

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

protected:

    /**
    Evaluate relevant forces when m_u is updated.
    */
    void updated_u() override;

};

} // namespace UniformMultiLayerLeverDrive2d
} // namespace FrictionQPotFEM

#include "UniformMultiLayerLeverDrive2d.hpp"

#endif
