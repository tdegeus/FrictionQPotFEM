/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_H
#define FRICTIONQPOTFEM_UNIFORMMULTILAYERINDIVIDUALDRIVE2D_H

#include "Generic2d.h"
#include "config.h"

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
inline std::vector<std::string> version_dependencies()
{
    return Generic2d::version_dependencies();
}

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
class System : public Generic2d::System {

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
    \param elastic_K Bulk modulus per quad. point of each elastic element, see setElastic().
    \param elastic_G Shear modulus per quad. point of each elastic element, see setElastic().
    \param plastic_K Bulk modulus per quad. point of each plastic element, see Plastic().
    \param plastic_G Shear modulus per quad. point of each plastic element, see Plastic().
    \param plastic_epsy Yield strain per quad. point of each plastic element, see Plastic().
    \param dt Time step, set setDt().
    \param rho Mass density, see setMassMatrix().
    \param alpha Background damping density, see setDampingMatrix().
    \param eta Damping at the interface (homogeneous), see setEta().
    \param drive_is_active Per layer (de)activate drive for each DOF, see layerSetTargetActive().
    \param k_drive Stiffness of driving spring, see layerSetDriveStiffness().
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
        double k_drive)
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
    }

protected:
    /**
    \cond
    */
    void init(
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
        double k_drive)
    {
        FRICTIONQPOTFEM_ASSERT(layer_is_plastic.size() == elem.size());
        FRICTIONQPOTFEM_ASSERT(layer_is_plastic.size() == node.size());

        using size_type = typename decltype(m_layerdrive_active)::size_type;

        m_n_layer = node.size();
        m_layer_node = node;
        m_layer_elem = elem;
        m_layer_is_plastic = layer_is_plastic;

        std::array<size_type, 1> shape1 = {static_cast<size_type>(m_n_layer)};
        std::array<size_type, 2> shape2 = {static_cast<size_type>(m_n_layer), 2};
        m_layerdrive_active.resize(shape2);
        m_layerdrive_targetubar.resize(shape2);
        m_layer_ubar.resize(shape2);
        m_layer_dV1.resize(shape2);
        m_slice_index.resize(shape1);

        m_layerdrive_targetubar.fill(0.0);
        m_layerdrive_active.fill(false);

        size_t n_elem_plas = 0;
        size_t n_elem_elas = 0;
        size_t n_layer_plas = 0;
        size_t n_layer_elas = 0;

        for (size_t i = 0; i < elem.size(); ++i) {
            if (m_layer_is_plastic(i)) {
                n_elem_plas += elem[i].size();
                n_layer_plas++;
            }
            else {
                n_elem_elas += elem[i].size();
                n_layer_elas++;
            }
        }

        array_type::tensor<size_t, 1> plas = xt::empty<size_t>({n_elem_plas});
        array_type::tensor<size_t, 1> elas = xt::empty<size_t>({n_elem_elas});

        std::array<size_type, 1> size_plas = {static_cast<size_type>(n_layer_plas) + 1};
        std::array<size_type, 1> size_elas = {static_cast<size_type>(n_layer_elas) + 1};
        m_slice_plas.resize(size_plas);
        m_slice_elas.resize(size_elas);
        m_slice_plas(0) = 0;
        m_slice_elas(0) = 0;
        m_N = 0;

        n_elem_plas = 0;
        n_elem_elas = 0;
        n_layer_plas = 0;
        n_layer_elas = 0;

        for (size_t i = 0; i < m_n_layer; ++i) {
            if (m_layer_is_plastic(i)) {
                size_t l = m_slice_plas(n_layer_plas);
                size_t u = n_elem_plas + elem[i].size();

                m_slice_index(i) = n_layer_plas;
                m_slice_plas(n_layer_plas + 1) = u;

                xt::view(plas, xt::range(l, u)) = elem[i];

                n_elem_plas += elem[i].size();
                n_layer_plas++;

                FRICTIONQPOTFEM_REQUIRE(m_N == elem[i].size() || m_N == 0);
                m_N = elem[i].size();
            }
            else {
                size_t l = m_slice_elas(n_layer_elas);
                size_t u = n_elem_elas + elem[i].size();

                m_slice_index(i) = n_layer_elas;
                m_slice_elas(n_layer_elas + 1) = u;

                xt::view(elas, xt::range(l, u)) = elem[i];

                n_elem_elas += elem[i].size();
                n_layer_elas++;
            }
        }

        this->initSystem(
            coor,
            conn,
            dofs,
            iip,
            elas,
            elastic_K,
            elastic_G,
            plas,
            plastic_K,
            plastic_G,
            plastic_epsy,
            dt,
            rho,
            alpha,
            eta);

        m_fdrive = m_vector.allocate_nodevec(0.0);
        m_ud = m_vector.allocate_dofval(0.0);
        m_uq = m_quad.allocate_qtensor<1>(0.0);
        m_dV = m_quad.dV();

        size_t nip = m_quad.nip();
        m_layer_dV1.fill(0.0);

        for (size_t i = 0; i < m_n_layer; ++i) {
            for (auto& e : m_layer_elem[i]) {
                for (size_t q = 0; q < nip; ++q) {
                    for (size_t d = 0; d < 2; ++d) {
                        m_layer_dV1(i, d) += m_dV(e, q);
                    }
                }
            }
        }

// sanity check nodes per layer
#ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
        for (size_t i = 0; i < m_n_layer; ++i) {
            auto e = this->layerElements(i);
            auto n = xt::unique(xt::view(m_vector.conn(), xt::keep(e)));
            FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(xt::sort(n), xt::sort(node[i]))));
        }
#endif

// sanity check elements per layer + slice of elas/plas element sets
#ifdef FRICTIONQPOTFEM_ENABLE_ASSERT
        for (size_t i = 0; i < m_n_layer; ++i) {
            array_type::tensor<size_t, 1> e;
            size_t j = m_slice_index(i);
            if (m_layer_is_plastic(i)) {
                e = xt::view(m_elem_plas, xt::range(m_slice_plas(j), m_slice_plas(j + 1)));
            }
            else {
                e = xt::view(m_elem_elas, xt::range(m_slice_elas(j), m_slice_elas(j + 1)));
            }
            FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(xt::sort(e), xt::sort(elem[i]))));
        }
#endif

        this->layerSetDriveStiffness(k_drive);
        this->layerSetTargetActive(drive_is_active);
    }
    /**
    \endcond
    */

public:
    size_t N() const override
    {
        return m_N;
    }

    std::string type() const override
    {
        return "FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System";
    }

    /**
    Return number of layers.
    \return Number of layers.
    */
    size_t nlayer() const
    {
        return m_n_layer;
    }

    /**
    Return the elements belonging to the i-th layer.
    \param i Index of the layer.
    \return List of element numbers.
    */
    array_type::tensor<size_t, 1> layerElements(size_t i) const
    {
        FRICTIONQPOTFEM_ASSERT(i < m_n_layer);
        return m_layer_elem[i];
    }

    /**
    Return the nodes belonging to the i-th layer.
    \param i Index of the layer.
    \return List of node numbers.
    */
    array_type::tensor<size_t, 1> layerNodes(size_t i) const
    {
        FRICTIONQPOTFEM_ASSERT(i < m_n_layer);
        return m_layer_node[i];
    }

    /**
    Return if a layer is elastic (`false`) or plastic (`true`).
    \return [#nlayer].
    */
    const array_type::tensor<bool, 1>& layerIsPlastic() const
    {
        return m_layer_is_plastic;
    }

    /**
    Set the stiffness of the springs connecting
    the average displacement of a layer ("ubar") to its set target value.
    Note that the stiffness of all springs is taken the same.

    \param k The stiffness.
    \param symmetric
        If `true` the spring is a normal spring.
        If `false` the spring has no stiffness under compression.
    */
    void layerSetDriveStiffness(double k, bool symmetric = true)
    {
        m_drive_spring_symmetric = symmetric;
        m_drive_k = k;
        this->computeLayerUbarActive();
        this->computeForceFromTargetUbar();
    }

    /**
    Initialise the event driven protocol by applying a perturbation to the loading springs
    and computing and storing the linear, purely elastic, response.
    The system can thereafter be moved forward to the next event.
    Note that this function itself does not change the system in any way,
    it just stores the relevant perturbations.

    \tparam S `array_type::tensor<double, 2>`
    \param ubar
        The perturbation in the target average position of each layer [#nlayer, 2].

    \tparam T `array_type::tensor<bool, 2>`
    \param active
        For each layer and each degree-of-freedom specify if
        the spring is active (`true`) or not (`false`) [#nlayer, 2].

    \return Value with which the input perturbation is scaled, see also eventDriven_deltaUbar().
    */
    template <class S, class T>
    double initEventDriven(const S& ubar, const T& active)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(ubar, m_layerdrive_targetubar.shape()));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));

        // backup system

        auto u0 = m_u;
        auto v0 = m_v;
        auto a0 = m_a;
        auto active0 = m_layerdrive_active;
        auto ubar0 = m_layerdrive_targetubar;
        auto t0 = m_t;

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
        this->layerSetTargetUbar(ubar);
        this->minimise();
        FRICTIONQPOTFEM_ASSERT(xt::all(xt::equal(m_plas.i(), 0)));
        auto up = m_u;
        m_u.fill(0.0);

        // restore yield strains

        m_plas.set_epsy(epsy0);

        // store result

        auto c = this->eventDriven_setDeltaU(up);
        m_pert_layerdrive_active = active;
        m_pert_layerdrive_targetubar = c * ubar;

        // restore system

        this->setU(u0);
        this->setV(v0);
        this->setA(a0);
        this->layerSetTargetActive(active0);
        this->layerSetTargetUbar(ubar0);
        this->setT(t0);

        return c;
    }

    /**
    Restore perturbation used from event driven protocol.
    \param ubar See eventDriven_deltaUbar().
    \param active See eventDriven_targetActive().
    \param delta_u See eventDriven_deltaU().
    \return Value with which the input perturbation is scaled, see also eventDriven_deltaU().
    */
    template <class S, class T, class U>
    double initEventDriven(const S& ubar, const T& active, const U& delta_u)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(ubar, m_layerdrive_targetubar.shape()));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(delta_u, m_u.shape()));
        double c = this->eventDriven_setDeltaU(delta_u);
        m_pert_layerdrive_active = active;
        m_pert_layerdrive_targetubar = c * ubar;
        return c;
    }

    /**
    Get target average position perturbation used for event driven code.
    \return [#nlayer, 2]
    */
    const array_type::tensor<double, 2>& eventDriven_deltaUbar() const
    {
        return m_pert_layerdrive_targetubar;
    }

    /**
    Get if the target average position is prescribed in the event driven code.
    \return [#nlayer, 2]
    */
    const array_type::tensor<bool, 2>& eventDriven_targetActive() const
    {
        return m_pert_layerdrive_active;
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
        return c;
    }

    /**
    Turn on (or off) springs connecting
    the average displacement of a layer ("ubar") to its set target value.

    \tparam T e.g. `array_type::tensor<bool, 2>`
    \param active
        For each layer and each degree-of-freedom specify if
        the spring is active (`true`) or not (`false`) [#nlayer, 2].
    */
    template <class T>
    void layerSetTargetActive(const T& active)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(active, m_layerdrive_active.shape()));
        m_layerdrive_active = active;
        this->computeLayerUbarActive();
        this->computeForceFromTargetUbar();
    }

    /**
    List the average displacement of each layer.
    Requires to recompute the average displacements
    (as they are normally only computed on the driven DOFs).

    \return Average displacement per layer [#nlayer, 2]
    */
    array_type::tensor<double, 2> layerUbar()
    {
        // Recompute needed because computeLayerUbarActive() only computes the average
        // on layers with an active spring.
        // This function, in contrast, returns the average on all layers.

        m_layer_ubar.fill(0.0);
        size_t nip = m_quad.nip();

        m_vector.asElement(m_u, m_ue);
        m_quad.interpQuad_vector(m_ue, m_uq);

        for (size_t i = 0; i < m_n_layer; ++i) {
            for (auto& e : m_layer_elem[i]) {
                for (size_t d = 0; d < 2; ++d) {
                    for (size_t q = 0; q < nip; ++q) {
                        m_layer_ubar(i, d) += m_uq(e, q, d) * m_dV(e, q);
                    }
                }
            }
        }

        m_layer_ubar /= m_layer_dV1;

        return m_layer_ubar;
    }

    /**
    List the target average displacement per layer.
    \return [#nlayer, 2]
    */
    const array_type::tensor<double, 2>& layerTargetUbar() const
    {
        return m_layerdrive_targetubar;
    }

    /**
    List if the driving spring is activate.
    \return [#nlayer, 2]
    */
    const array_type::tensor<bool, 2>& layerTargetActive() const
    {
        return m_layerdrive_active;
    }

    /**
    Overwrite the target average displacement per layer.
    Only layers that have an active driving spring will feel a force
    (if its average displacement is different from the target displacement),
    see layerSetTargetActive().

    \tparam T e.g. `array_type::tensor<double, 2>`
    \param ubar The target average position of each layer [#nlayer, 2].
    */
    template <class T>
    void layerSetTargetUbar(const T& ubar)
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(ubar, m_layerdrive_targetubar.shape()));
        m_layerdrive_targetubar = ubar;
        this->computeForceFromTargetUbar(); // average displacement and other forces do not change
    }

    /**
    Move the layer such that the average displacement is exactly equal to its input value.

    \tparam S e.g. `array_type::tensor<double, 2>`
    \tparam T e.g. `array_type::tensor<bool, 2>`

    \param ubar
        The target average position of each layer [#nlayer, 2].

    \param prescribe
        Per layers/degree-of-freedom specify if its average is modified [#nlayer, 2].
        Note that this not modify which of the driving springs is active or not,
        that can only be changed using layerSetTargetActive().
    */
    template <class S, class T>
    void layerSetUbar(const S& ubar, const T& prescribe)
    {
        auto current = this->layerUbar();

        for (size_t i = 0; i < m_n_layer; ++i) {
            for (size_t d = 0; d < 2; ++d) {
                if (prescribe(i, d)) {
                    double du = ubar(i, d) - current(i, d);
                    for (auto& n : m_layer_node[i]) {
                        m_u(n, d) += du;
                    }
                }
            }
        }

        this->computeLayerUbarActive();
        this->computeForceFromTargetUbar();
        this->computeForceMaterial();
    }

    /**
    Get target average displacements that correspond to affine simple shear
    (with the bottom fixed).
    In particular \f$ \bar{u}_x^i = 2 \Delta \gamma h_i \f$
    with \f$ \bar{u}_x^i \f$ the \f$ x \f$-component of the target average displacement
    of layer \f$ i \f$.

    \tparam T e.g. `array_type::tensor<double, 1>`
    \param delta_gamma Affine strain to add.
    \param height The height \f$ h_i \f$ of the loading frame of each layer [#nlayer].
    */
    template <class T>
    array_type::tensor<double, 2>
    layerTargetUbar_affineSimpleShear(double delta_gamma, const T& height) const
    {
        FRICTIONQPOTFEM_ASSERT(xt::has_shape(height, {m_n_layer}));

        auto ret = xt::zeros_like(m_layerdrive_targetubar);

        for (size_t i = 0; i < m_n_layer; ++i) {
            ret(i, 0) += 2.0 * delta_gamma * height(i);
        }

        return ret;
    }

    /**
    Nodal force induced by the driving springs.
    The only non-zero contribution comes from:
    -   springs that are active, see layerSetTargetActive(), and
    -   layers whose average displacement is different from its target value.

    \return nodevec [nnode, ndim].
    */
    const array_type::tensor<double, 2>& fdrive() const
    {
        return m_fdrive;
    }

    /**
    Force of each of the driving springs.
    The only non-zero contribution comes from:
    -   springs that are active, see layerSetTargetActive(), and
    -   layers whose average displacement is different from its target value.

    \return [#nlayer, ndim].
    */
    array_type::tensor<double, 2> layerFdrive() const
    {
        array_type::tensor<double, 2> ret = xt::zeros<double>({m_n_layer, size_t(2)});

        for (size_t i = 0; i < m_n_layer; ++i) {
            for (size_t d = 0; d < 2; ++d) {
                if (m_layerdrive_active(i, d)) {
                    ret(i, d) = m_drive_k * (m_layerdrive_targetubar(i, d) - m_layer_ubar(i, d));
                }
            }
        }

        return ret;
    }

protected:
    /**
    Compute the average displacement of all layers with an active driving spring.
    */
    void computeLayerUbarActive()
    {
        m_layer_ubar.fill(0.0);
        size_t nip = m_quad.nip();

        m_vector.asElement(m_u, m_ue);
        m_quad.interpQuad_vector(m_ue, m_uq);

        for (size_t i = 0; i < m_n_layer; ++i) {
            for (size_t d = 0; d < 2; ++d) {
                if (m_layerdrive_active(i, d)) {
                    for (auto& e : m_layer_elem[i]) {
                        for (size_t q = 0; q < nip; ++q) {
                            m_layer_ubar(i, d) += m_uq(e, q, d) * m_dV(e, q);
                        }
                    }
                }
            }
        }

        m_layer_ubar /= m_layer_dV1;
    }

    /**
    Compute force deriving from the activate springs between the average displacement of
    the layer and its target value.
    The force is applied as a force density.

    Internal rule: computeLayerUbarActive() is called before this function,
    if the displacement changed since the last time the average was computed.
    */
    void computeForceFromTargetUbar()
    {
        m_uq.fill(0.0); // pre-allocated value that an be freely used
        size_t nip = m_quad.nip();

        for (size_t i = 0; i < m_n_layer; ++i) {
            for (size_t d = 0; d < 2; ++d) {
                if (m_layerdrive_active(i, d)) {
                    double f = m_drive_k * (m_layer_ubar(i, d) - m_layerdrive_targetubar(i, d));
                    if (m_drive_spring_symmetric || f < 0) { // buckle under compression
                        for (auto& e : m_layer_elem[i]) {
                            for (size_t q = 0; q < nip; ++q) {
                                m_uq(e, q, d) = f;
                            }
                        }
                    }
                }
            }
        }

        m_quad.int_N_vector_dV(m_uq, m_ue);
        m_vector.assembleDofs(m_ue, m_ud);
        m_vector.asNode(m_ud, m_fdrive);
    }

    /**
    Evaluate relevant forces when m_u is updated.
    */
    void updated_u() override
    {
        this->computeLayerUbarActive();
        this->computeForceFromTargetUbar();
        this->computeForceMaterial();
    }

    /**
    Compute:
    -   m_fint = m_fdrive + m_fmaterial + m_fdamp
    -   m_fext[iip] = m_fint[iip]
    -   m_fres = m_fext - m_fint

    Internal rule: all relevant forces are expected to be updated before this function is
    called.
    */
    void computeInternalExternalResidualForce() override
    {
        xt::noalias(m_fint) = m_fdrive + m_fmaterial + m_fdamp;
        m_vector.copy_p(m_fint, m_fext);
        xt::noalias(m_fres) = m_fext - m_fint;
    }

protected:
    size_t m_N; ///< Linear system size.
    size_t m_n_layer; ///< Number of layers.
    std::vector<array_type::tensor<size_t, 1>> m_layer_node; ///< Nodes per layer.
    std::vector<array_type::tensor<size_t, 1>> m_layer_elem; ///< Elements per layer.
    array_type::tensor<bool, 1> m_layer_is_plastic; ///< Per layer ``true`` if the layer is plastic.
    array_type::tensor<size_t, 1>
        m_slice_index; ///< Per layer the index in m_slice_plas or m_slice_elas.
    array_type::tensor<size_t, 1> m_slice_plas; ///< How to slice elastic_elem(): start & end index
    array_type::tensor<size_t, 1> m_slice_elas; ///< How to slice plastic_elem(): start & end index

    bool m_drive_spring_symmetric; ///< If false the drive spring buckles in compression
    double m_drive_k; ///< Stiffness of the drive control frame
    array_type::tensor<bool, 2> m_layerdrive_active; ///< See `prescribe` in layerSetTargetUbar()
    array_type::tensor<double, 2>
        m_layerdrive_targetubar; ///< Per layer, the prescribed average position.
    array_type::tensor<double, 2> m_layer_ubar; ///< See computeLayerUbarActive().
    array_type::tensor<double, 2> m_fdrive; ///< Force related to driving frame
    array_type::tensor<double, 2>
        m_layer_dV1; ///< volume per layer (as vector, same for all dimensions)
    array_type::tensor<double, 2> m_dV; ///< copy of m_quad.dV()
    array_type::tensor<double, 3> m_uq; ///< qvector
    array_type::tensor<double, 1> m_ud; ///< dofval

    array_type::tensor<bool, 2> m_pert_layerdrive_active; ///< Event driven: applied lever setting.
    array_type::tensor<double, 2>
        m_pert_layerdrive_targetubar; ///< Event driven: applied lever setting.
};

} // namespace UniformMultiLayerIndividualDrive2d
} // namespace FrictionQPotFEM

#endif
