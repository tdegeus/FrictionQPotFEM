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
 * @brief Compared to UniformSingleLayer2d::System() this class adds thermal fluctuations.
 *
 * The fluctuations are implemented as a dipolar force on each element edge.
 * Both its components are drawn from a normal distribution with zero mean and 'temperature'
 * standard deviation.
 * The temperature specification is thereby in units of the equivalent stress, see
 * GMatElastoPlasticQPot::Cartesian2d::Sigd().
 */
class System : public UniformSingleLayer2d::System {

public:
    System() = default;

public:
    virtual ~System(){};

public:
    /**
     * @brief Define the geometry, including boundary conditions and element sets.
     *
     * @tparam C Type of nodal coordinates, e.g. `array_type::tensor<double, 2>`
     * @tparam E Type of connectivity and DOFs, e.g. `array_type::tensor<size_t, 2>`
     * @tparam L Type of node/element lists, e.g. `array_type::tensor<size_t, 1>`
     * @param coor Nodal coordinates.
     * @param conn Connectivity.
     * @param dofs DOFs per node.
     * @param iip DOFs whose displacement is fixed.
     * @param elastic_elem Elastic elements.
     * @param elastic_K Bulk modulus per quad. point of each elastic element, see setElastic().
     * @param elastic_G Shear modulus per quad. point of each elastic element, see setElastic().
     * @param plastic_elem Plastic elements.
     * @param plastic_K Bulk modulus per quad. point of each plastic element, see Plastic().
     * @param plastic_G Shear modulus per quad. point of each plastic element, see Plastic().
     * @param plastic_epsy Yield strain per quad. point of each plastic element, see Plastic().
     * @param dt Time step, set setDt().
     * @param rho Mass density, see setMassMatrix().
     * @param alpha Background damping density, see setDampingMatrix().
     * @param eta Damping at the interface (homogeneous), see setEta().
     * @param temperature_dinc Duration to keep the same random thermal stress tensor.
     * @param temperature_seed Seed random generator thermal stress (`s + arange(3 * nelem)` are used).
     * @param temperature 'Temperature': in units of the the equivalent stress.
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
        double h = coor(pconn(0, 1), 0) - coor(pconn(0, 0), 0);
        for (size_t d = 0; d < dofs.shape(1); ++d) {
            FRICTIONQPOTFEM_REQUIRE(dofs(pconn(0, 0), d) == dofs(pconn(m_N - 1, 1), d));
            FRICTIONQPOTFEM_REQUIRE(dofs(pconn(0, 3), d) == dofs(pconn(m_N - 1, 2), d));
        }

        for (size_t e = 0; e < m_N; ++e) {
            FRICTIONQPOTFEM_REQUIRE(xt::allclose(coor(pconn(e, 1), 0) - coor(pconn(e, 0), 0), h));
            FRICTIONQPOTFEM_REQUIRE(xt::allclose(coor(pconn(e, 2), 1) - coor(pconn(e, 1), 1), h));
            FRICTIONQPOTFEM_REQUIRE(xt::allclose(coor(pconn(e, 2), 0) - coor(pconn(e, 3), 0), h));
            FRICTIONQPOTFEM_REQUIRE(xt::allclose(coor(pconn(e, 3), 1) - coor(pconn(e, 1), 1), h));
        }

        m_T = temperature;
        m_std = m_T * h;
        m_dinc = temperature_dinc;
        m_fthermal = xt::zeros_like(m_fint);
        m_gen = prrng::pcg32(temperature_seed, 0);
        m_ncache = 100;
        m_cache = m_gen.normal({size_t(m_ncache), m_N, size_t(3), size_t(2)}, 0, m_std);
        m_cache_start = 0;
        this->setInc(m_inc);
    }

protected:
    size_t m_computed; ///< Increment at which the thermal stress tensor was generated.
    size_t m_dinc; ///< Duration to keep the same random thermal stress tensor.
    double m_T; ///< Definition of temperature (units of equivalent stress).
    double m_std; ///< Standard deviation for signed equivalent thermal stress.
    array_type::tensor<double, 2> m_fthermal; ///< Nodal force from temperature.
    array_type::tensor<double, 4> m_cache; ///< Cache [ncache, m_elem_plas.size(), 3, 2]
    int64_t m_cache_start; ///< Start index of the cache.
    int64_t m_ncache; ///< Number of cached items.
    prrng::pcg32 m_gen; ///< Random generator for thermal forces

public:
    std::string type() const override
    {
        return "FrictionQPotFEM.UniformSingleLayerThermal2d.System";
    }

    /**
     * @brief The force vector that represents the effect of temperature (on the weak layer only).
     * @return nodevec
     */
    const array_type::tensor<double, 2>& fthermal() const
    {
        return m_fthermal;
    }

    /**
     * @brief Return the target temperature.
     * @return double
     */
    double temperature() const
    {
        return m_T;
    }

protected:
    /**
     * @brief Update cache of thermal forces on the weak layer.
     */
    void updateCache(int64_t index)
    {
        if (index >= m_cache_start && index < m_cache_start + m_ncache) {
            return;
        }

        int64_t advance = index - m_cache_start - m_ncache;
        if (advance != 0) {
            m_gen.advance(advance * static_cast<int64_t>(m_N * 6));
        }

        m_cache = m_gen.normal({size_t(m_ncache), m_N, size_t(3), size_t(2)}, 0, m_std);
        m_cache_start = index;
    }

public:
    void setInc(size_t arg) override
    {
        m_inc = arg;
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
        this->updateCache(index);

        auto& conn = m_vector_plas.conn();

        std::fill(&m_fthermal(conn(0, 0), 0), &m_fthermal(conn(m_N - 1, 2), 1) + 1, 0);

        for (size_t e = 0; e < m_N; ++e) {
            const size_t* elem = &conn(e, 0);
            double* cache = &m_cache(index - m_cache_start, e, 0, 0);

            for (size_t d = 0; d < 2; ++d) {
                m_fthermal(elem[0], d) += cache[0 + d];
                m_fthermal(elem[3], d) -= cache[0 + d];
                m_fthermal(elem[0], d) += cache[2 + d];
                m_fthermal(elem[1], d) -= cache[2 + d];
                m_fthermal(elem[2], d) += cache[4 + d];
                m_fthermal(elem[3], d) -= cache[4 + d];
            }
        }

        // apply periodic boundary conditions
        for (size_t d = 0; d < 2; ++d) {
            m_fthermal(conn(0, 0), d) += m_fthermal(conn(m_N - 1, 1), d);
            m_fthermal(conn(0, 3), d) += m_fthermal(conn(m_N - 1, 2), d);
            m_fthermal(conn(m_N - 1, 1), d) = m_fthermal(conn(0, 0), d);
            m_fthermal(conn(m_N - 1, 2), d) = m_fthermal(conn(0, 3), d);
        }

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
