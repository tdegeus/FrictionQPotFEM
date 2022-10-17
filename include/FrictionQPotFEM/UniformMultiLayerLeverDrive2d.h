/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_H
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERLEVERDRIVE2D_H

#include "UniformMultiLayerIndividualDrive2d.h"
#include "config.h"

namespace FrictionQPotFEM {

/**
 * System in 2-d with:
 *
 * -   Several weak layers.
 * -   Layers connected to a lever, that is driven through a spring.
 * -   Uniform elasticity.
 */
namespace UniformMultiLayerLeverDrive2d {

/**
 * @copydoc Generic2d::version_dependencies()
 */
inline std::vector<std::string> version_dependencies()
{
    return Generic2d::version_dependencies();
}

/**
 * @copydoc Generic2d::version_compiler()
 */
inline std::vector<std::string> version_compiler()
{
    return Generic2d::version_compiler();
}

/**
 * Similar to UniformMultiLayerIndividualDrive2d::System(), but with the difference that the
 * target average displacement per layer is not individually prescribed, but determined by
 * a lever, whose position (whose rotation to be precise) is determined by a driving spring
 * attached to it.
 * The assumption is made that the lever has no inertia:
 * its position is computed by assuming that the sum of moments acting on it is zero.
 */
class System : public UniformMultiLayerIndividualDrive2d::System {

private:
    using UniformMultiLayerIndividualDrive2d::System::initEventDriven;

public:
    System() = default;

    virtual ~System(){};

    /**
     * Define basic geometry.
     *
     * @param coor Nodal coordinates.
     * @param conn Connectivity.
     * @param dofs DOFs per node.
     * @param iip DOFs whose displacement is fixed.
     * @param elem Elements per layer.
     * @param node Nodes per layer.
     * @param layer_is_plastic Per layer set if elastic (= 0) or plastic (= 1).
     * @param elastic_K Bulk modulus per quad. point of each elastic element, see setElastic().
     * @param elastic_G Shear modulus per quad. point of each elastic element, see setElastic().
     * @param plastic_K Bulk modulus per quad. point of each plastic element, see Plastic().
     * @param plastic_G Shear modulus per quad. point of each plastic element, see Plastic().
     * @param plastic_epsy Yield strain per quad. point of each plastic element, see Plastic().
     * @param dt Time step, set setDt().
     * @param rho Mass density, see setMassMatrix().
     * @param alpha Background damping density, see setDampingMatrix().
     * @param eta Damping at the interface (homogeneous), see setEta().
     * @param drive_is_active Per layer (de)activate drive for each DOF, see layerSetTargetActive().
     * @param k_drive Stiffness of driving spring, see layerSetDriveStiffness().
     * @param H The height of the spring pulling the lever, see setLeverProperties().
     * @param hi The height of the loading frame of each layer, setLeverProperties().
     */
    System(
        const array_type::tensor<double, 2>& coor,
        const array_type::tensor<size_t, 2>& conn,
        const array_type::tensor<size_t, 2>& dofs,
        const array_type::tensor<size_t, 1>& iip,
        const std::vector<array_type::tensor<size_t, 1>>& elem,
        const std::vector<array_type::tensor<size_t, 1>>& node,
        const array_type::tensor<bool, 1>& layer_is_plastic,
        const array_type::tensor<double, 2>& elastic_K,
        const array_type::tensor<double, 2>& elastic_G,
        const array_type::tensor<double, 2>& plastic_K,
        const array_type::tensor<double, 2>& plastic_G,
        const array_type::tensor<double, 3>& plastic_epsy,
        double dt,
        double rho,
        double alpha,
        double eta,
        const array_type::tensor<bool, 2>& drive_is_active,
        double k_drive,
        double H,
        const array_type::tensor<double, 1>& hi)
    {
        this->init(
            coor,
            conn,
            dofs,
            iip,
            elem,
            node,
            layer_is_plastic,
            elastic_K,
            elastic_G,
            plastic_K,
            plastic_G,
            plastic_epsy,
            dt,
            rho,
            alpha,
            eta,
            drive_is_active,
            k_drive);

        using size_type = typename decltype(m_lever_hi)::size_type;
        std::array<size_type, 1> shape = {static_cast<size_type>(m_n_layer)};

        m_lever_hi.resize(shape);
        m_lever_hi.fill(0.0);

        m_lever_hi_H.resize(shape);
        m_lever_hi_H.fill(0.0);

        m_lever_hi_H_2.resize(shape);
        m_lever_hi_H_2.fill(0.0);

        m_lever_H = 0.0;
        m_lever_target = 0.0;
        m_lever_u = 0.0;

        this->setLeverProperties(H, hi);
    }

public:
    std::string type() const override
    {
        return "FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System";
    }

    /**
     * Overwrite the lever properties.
     *
     * @tparam T e.g. `array_type::tensor<double, 1>`
     * @param H The height of the spring pulling the lever.
     * @param hi The height \f$ h_i \f$ of the loading frame of each layer [nlayer].
     */
    template <class T>
    void setLeverProperties(double H, const T& hi)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(hi, m_lever_hi.shape()));
        m_lever_hi = hi;
        m_lever_H = H;

        m_lever_hi_H = m_lever_hi / H;
        m_lever_hi_H_2 = xt::pow(m_lever_hi / H, 2.0);

        this->updated_u(); // updates forces
    }

    /**
     * Overwrite the target 'position' of the spring pulling the lever.
     *
     * @param xdrive Lever position
     */
    void setLeverTarget(double xdrive)
    {
        m_lever_target = xdrive;
        this->updated_u(); // updates target average displacement per layer, and forces
    }

    /**
     * Get the current target lever 'position'.
     *
     * @return double
     */
    double leverTarget() const
    {
        return m_lever_target;
    }

    /**
     * Get the current lever 'position'.
     *
     * @return double
     */
    double leverPosition() const
    {
        return m_lever_u;
    }

    /**
     * Initialise the event driven protocol by applying a perturbation to loading spring
     * and computing and storing the linear, purely elastic, response.
     * The system can thereafter be moved forward to the next event.
     * Note that this function itself does not change the system in any way,
     * it just stores the relevant perturbations.
     *
     * @param xlever Target 'position' of the spring pulling the lever.
     *
     * @param active
     *     For each layer and each degree-of-freedom specify if
     *     the spring is active (`true`) or not (`false`) [#nlayer, 2].
     * \
     */
    template <class T>
    void initEventDriven(double xlever, const T& active)
    {
        FRICTIONQPOTFEM_ASSERT(this->isHomogeneousElastic());
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));

        // backup system

        auto u0 = m_u;
        auto v0 = m_v;
        auto a0 = m_a;
        auto active0 = m_layerdrive_active;
        auto ubar0 = m_layerdrive_targetubar;
        auto xdrive0 = m_lever_target;
        auto inc0 = m_inc;

        array_type::tensor<double, 3> epsy0 = m_plas.epsy();
        array_type::tensor<double, 3> rigid = xt::empty<double>({m_nelem_plas, m_nip, size_t(2)});

        double infty = std::numeric_limits<double>::max();

        for (size_t e = 0; e < m_nelem_plas; ++e) {
            for (size_t q = 0; q < m_nip; ++q) {
                rigid(e, q, 0) = -infty;
                rigid(e, q, 1) = infty;
            }
        }

        m_plas.set_epsy(rigid);

        // perturbation

        m_u.fill(0.0);
        m_v.fill(0.0);
        m_a.fill(0.0);
        this->updated_u();
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(m_plas.i(), 0)));
        this->layerSetTargetActive(active);
        this->setLeverTarget(xlever);
        this->minimise(0);
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(m_plas.i(), 0)));
        auto up = m_u;
        m_u.fill(0.0);

        // restore yield strains

        m_plas.set_epsy(epsy0);

        // store result

        auto c = this->eventDriven_setDeltaU(up);
        m_pert_layerdrive_active = active;
        m_pert_layerdrive_targetubar = c * m_layerdrive_targetubar;
        m_pert_lever_target = c * xlever;

        // restore system

        this->setU(u0);
        this->setV(v0);
        this->setA(a0);
        this->layerSetTargetActive(active0);
        this->layerSetTargetUbar(ubar0);
        this->setLeverTarget(xdrive0);
        this->setInc(inc0);
    }

    /**
     * Restore perturbation used from event driven protocol.
     * @param xlever See eventDriven_leverPosition().
     * @param active See eventDriven_targetActive().
     * @param delta_u See eventDriven_deltaU().
     * @param delta_ubar See eventDriven_deltaUbar().
     * @return Value with which the input perturbation is scaled, see also eventDriven_deltaU().
     */
    template <class T, class U, class W>
    double initEventDriven(double xlever, const T& active, const U& delta_u, const W& delta_ubar)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(delta_ubar, m_layerdrive_targetubar.shape()));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(delta_u, m_u.shape()));
        double c = this->eventDriven_setDeltaU(delta_u);
        m_pert_layerdrive_active = active;
        m_pert_layerdrive_targetubar = c * delta_ubar;
        m_pert_lever_target = c * xlever;
        return c;
    }

    double eventDrivenStep(
        double deps,
        bool kick,
        int direction = +1,
        bool yield_element = false,
        bool fallback = false) override
    {
        double c =
            Generic2d::System::eventDrivenStep(deps, kick, direction, yield_element, fallback);
        this->layerSetTargetUbar(m_layerdrive_targetubar + c * m_pert_layerdrive_targetubar);
        this->setLeverTarget(m_lever_target + c * m_pert_lever_target);
        return c;
    }

    /**
     * Get target 'position' of the spring pulling the lever perturbation used for event driven
     * code.
     * @return Value
     */
    double eventDriven_deltaLeverPosition() const
    {
        return m_pert_lever_target;
    }

protected:
    /**
     * Evaluate relevant forces when m_u is updated.
     * The assumption is made that the lever has no inertia:
     * its position is computed by assuming that the sum of moments acting on it is zero.
     */
    void updated_u() override
    {
        this->computeLayerUbarActive();

        // Position of the lever based on equilibrium

        m_lever_u = m_lever_target;
        double n = 1.0;

        for (size_t i = 0; i < m_n_layer; ++i) {
            if (m_layerdrive_active(i, 0)) {
                m_lever_u += m_layer_ubar(i, 0) * m_lever_hi_H(i);
                n += m_lever_hi_H_2(i);
            }
        }

        m_lever_u /= n;

        // Update position of driving springs

        for (size_t i = 0; i < m_n_layer; ++i) {
            if (m_layerdrive_active(i, 0)) {
                m_layerdrive_targetubar(i, 0) = m_lever_hi_H(i) * m_lever_u;
            }
        }

        // Update forces
        this->computeForceFromTargetUbar();
        this->computeForceMaterial();
    }

private:
    double m_lever_H; ///< See setLeverProperties().
    array_type::tensor<double, 1> m_lever_hi; ///< See setLeverProperties().
    array_type::tensor<double, 1> m_lever_hi_H; ///< m_lever_hi / H
    array_type::tensor<double, 1> m_lever_hi_H_2; ///< (m_lever_hi / H)^2
    double m_lever_target; ///< See setLeverTarget().
    double m_lever_u; ///< Current position of the lever.
    double m_pert_lever_target; ///< Perturbation in target position for event driven load.
};

} // namespace UniformMultiLayerLeverDrive2d
} // namespace FrictionQPotFEM

#endif
