/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYERTHERMAL2D_H
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYERTHERMAL2D_H

#include <prrng.h>

#include "Generic2d.h"
#include "UniformSingleLayer2d.h"
#include "config.h"

namespace FrictionQPotFEM {

/**
System in 2-d with:

-   A weak, middle, layer.
-   Uniform elasticity.
-   Thermal fluctuations.
*/
namespace UniformSingleLayerThermal2d {

/**
\copydoc Generic2d::version_dependencies()
*/
inline std::vector<std::string> version_dependencies()
{
    return Generic2d::version_dependencies();
}

/**
\copydoc Generic2d::version_compiler()
*/
inline std::vector<std::string> version_compiler()
{
    return Generic2d::version_compiler();
}

/**
Compared to UniformSingleLayer2d::System() this class adds thermal fluctuations.
The fluctuations are implemented as a stress field that is random for each Gauss point.
The stress tensor per Gauss point is constructed such that the equivalent stress is the
absolute value of a random variable drawn from a normal distribution `N(0, temperature)`.
That value is then randomly distributed between simple and pure shear contributions with
random directions.
*/
class System : public UniformSingleLayer2d::System {

public:
    System() = default;

public:
    virtual ~System(){};

public:
    /**
    Define the geometry, including boundary conditions and element sets.

    \tparam C Type of nodal coordinates, e.g. `array_type::tensor<double, 2>`
    \tparam E Type of connectivity and DOFs, e.g. `array_type::tensor<size_t, 2>`
    \tparam L Type of node/element lists, e.g. `array_type::tensor<size_t, 1>`
    \param coor Nodal coordinates.
    \param conn Connectivity.
    \param dofs DOFs per node.
    \param iip DOFs whose displacement is fixed.
    \param elastic_elem Elastic elements.
    \param elastic_K Bulk modulus per quad. point of each elastic element, see setElastic().
    \param elastic_G Shear modulus per quad. point of each elastic element, see setElastic().
    \param plastic_elem Plastic elements.
    \param plastic_K Bulk modulus per quad. point of each plastic element, see Plastic().
    \param plastic_G Shear modulus per quad. point of each plastic element, see Plastic().
    \param plastic_epsy Yield strain per quad. point of each plastic element, see Plastic().
    \param dt Time step, set setDt().
    \param rho Mass density, see setMassMatrix().
    \param alpha Background damping density, see setDampingMatrix().
    \param eta Damping at the interface (homogeneous), see setEta().
    \param temperature_dinc Duration to keep the same random thermal stress tensor.
    \param temperature_seed Seed random generator thermal stress (`s + arange(3 * nelem)` are used).
    \param temperature 'Temperature': standard deviation of the equivalent stress.
    */
    template <class C, class E, class L, class M, class Y>
    System(
        const C& coor,
        const E& conn,
        const E& dofs,
        const L& iip,
        const L& elastic_elem,
        const M& elastic_K,
        const M& elastic_G,
        const L& plastic_elem,
        const M& plastic_K,
        const M& plastic_G,
        const Y& plastic_epsy,
        double dt,
        double rho,
        double alpha,
        double eta,
        double temperature_dinc,
        size_t temperature_seed,
        double temperature)
    {
        this->initSystem(
            coor,
            conn,
            dofs,
            iip,
            elastic_elem,
            elastic_K,
            elastic_G,
            plastic_elem,
            plastic_K,
            plastic_G,
            plastic_epsy,
            dt,
            rho,
            alpha,
            eta);

        auto& pconn = m_vector_plas.conn();
        FRICTIONQPOTFEM_REQUIRE(pconn(0, 0) == pconn(m_N - 1, 1));
        FRICTIONQPOTFEM_REQUIRE(pconn(0, 3) == pconn(m_N - 1, 2));
        FRICTIONQPOTFEM_REQUIRE(xt::all(
            xt::equal(xt::unique(pconn), xt::arange<size_t>(pconn(0, 0), pconn(m_N - 1, 2) + 1))));

        m_T = temperature;
        m_dinc = temperature_dinc;
        m_fthermal = xt::zeros_like(m_fint);
        m_gen = prrng::pcg32(temperature_seed, 0);
        m_ncache = 100;
        m_cache_start = -m_ncache;
        this->setInc(m_inc);
    }

protected:
    size_t m_computed; ///< Increment at which the thermal stress tensor was generated.
    size_t m_dinc; ///< Duration to keep the same random thermal stress tensor.
    double m_T; ///< Standard deviation for signed equivalent thermal stress.
    array_type::tensor<double, 2> m_fthermal; ///< Nodal force from temperature.
    array_type::tensor<double, 3> m_cache; ///< Cache [ncache, m_elem_plas.size(), 3, 2]
    int64_t m_cache_start; ///< Start index of the cache.
    int64_t m_ncache; ///< Number of cached items.
    prrng::pcg32 m_gen; ///< Random generator for thermal forces

public:
    std::string type() const override
    {
        return "FrictionQPotFEM.UniformSingleLayerThermal2d.System";
    }

protected:
    /**
     * @brief Update cache of thermal forces on the weak layer.
     */
    void updateCache(int64_t index)
    {
        int64_t advance = index - m_cache_start - m_ncache;

        if (advance != 0) {
            m_gen.advance(advance * static_cast<int64_t>(m_N * 6));
        }

        m_cache_start += m_ncache;
        size_t n = static_cast<size_t>(m_ncache);
        m_cache = m_gen.normal({n, m_N, size_t(3), size_t(2)}, 0, m_T);
    }

public:
    void setInc(size_t arg) override
    {
        m_inc = arg;
        this->updateCache(static_cast<int64_t>((m_inc - m_inc % m_dinc) / m_dinc));
        this->computeThermalForce(true);
    }

protected:
    void computeThermalForce(bool force = false)
    {
        if (m_inc == m_computed && !force) {
            return;
        }

        if (m_inc % m_dinc != 0 && !force) {
            return;
        }

        int64_t index = static_cast<int64_t>(m_inc / m_dinc);

        if (index - m_cache_start >= m_ncache) {
            this->updateCache(index);
        }

        auto& conn = m_vector_plas.conn();

        std::fill(&m_fthermal(conn(0, 0), 0), &m_fthermal(conn(m_N - 1, 2), 1), 0);

        // todo: remove
        FRICTIONQPOTFEM_REQUIRE(xt::allclose(m_fthermal, 0));

        for (size_t e = 0; e < m_N; ++e) {
            const size_t* elem = &conn(e, 0);
            double* cache = &m_cache(index, e, 0, 0);

            m_fthermal(elem[0], 0) += cache[0];
            m_fthermal(elem[3], 0) -= cache[0];
            m_fthermal(elem[0], 1) += cache[1];
            m_fthermal(elem[3], 1) -= cache[1];

            m_fthermal(elem[0], 0) += cache[2];
            m_fthermal(elem[1], 0) -= cache[2];
            m_fthermal(elem[0], 1) += cache[3];
            m_fthermal(elem[1], 1) -= cache[3];

            m_fthermal(elem[2], 0) += cache[4];
            m_fthermal(elem[3], 0) -= cache[4];
            m_fthermal(elem[2], 1) += cache[5];
            m_fthermal(elem[3], 1) -= cache[5];
        }

        // apply periodic boundary conditions
        m_fthermal(conn(m_N - 1, 1), 0) = m_fthermal(conn(0, 0), 0);
        m_fthermal(conn(m_N - 1, 1), 1) = m_fthermal(conn(0, 0), 1);
        m_fthermal(conn(m_N - 1, 2), 0) = m_fthermal(conn(0, 3), 0);
        m_fthermal(conn(m_N - 1, 2), 1) = m_fthermal(conn(0, 3), 1);

        m_computed = m_inc;
    }

protected:
    /**
    Compute:
    -   m_fint = m_fmaterial + m_fdamp + m_fthermal
    -   m_fext[iip] = m_fint[iip]
    -   m_fres = m_fext - m_fint

    Internal rule: all relevant forces are expected to be updated before this function is called.
    */
    void computeInternalExternalResidualForce() override
    {
        this->computeThermalForce();
        xt::noalias(m_fint) = m_fmaterial + m_fdamp + m_fthermal;
        m_vector.copy_p(m_fint, m_fext);
        xt::noalias(m_fres) = m_fext - m_fint;
    }
};

} // namespace UniformSingleLayerThermal2d
} // namespace FrictionQPotFEM

#endif
