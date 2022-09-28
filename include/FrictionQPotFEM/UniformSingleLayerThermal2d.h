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

        m_T = temperature;
        m_dinc = temperature_dinc;

        m_fthermal = xt::empty_like(m_fint);
        m_TSig = xt::zeros_like(m_Sig);

        size_t n = m_nelem * m_nip;
        array_type::tensor<size_t, 2> istate = xt::arange<size_t>(n).reshape({m_nelem, m_nip});
        array_type::tensor<size_t, 2> iseq = xt::zeros<size_t>({m_nelem, m_nip});

        size_t seed = temperature_seed;
        m_gen_sig = prrng::pcg32_tensor<2>(xt::eval(seed + istate), iseq);
        m_gen_phi = prrng::pcg32_tensor<2>(xt::eval(m_nelem * m_nip * seed + istate), iseq);
        m_gen_sgn = prrng::pcg32_tensor<2>(xt::eval(2 * m_nelem * m_nip * seed + istate), iseq);
        m_ncache = 100;
        m_cache_start = -m_ncache;
        m_cache = xt::empty<double>({(size_t)m_ncache, m_nelem, m_nip, (size_t)2, (size_t)2});

        this->setInc(m_inc);
    }

protected:
    size_t m_computed; ///< Increment at which the thermal stress tensor was generated.
    size_t m_dinc; ///< Duration to keep the same random thermal stress tensor.
    double m_T; ///< Standard deviation for signed equivalent thermal stress.
    array_type::tensor<double, 4> m_TSig; ///< Quad-point tensor: thermal stress.
    array_type::tensor<double, 2> m_fthermal; ///< Nodal force from temperature.
    array_type::tensor<double, 5> m_cache; ///< Cache for #m_TSig.
    int64_t m_cache_start; ///< Start index of the cache.
    int64_t m_ncache; ///< Number of cached items.
    prrng::pcg32_tensor<2> m_gen_sig; ///< Random generator for equivalent thermal stress.
    prrng::pcg32_tensor<2> m_gen_phi; ///< Random generator for ratio between simple and pure shear.
    prrng::pcg32_tensor<2> m_gen_sgn; ///< Random generator for sign (of pure shear).

public:
    std::string type() const override
    {
        return "FrictionQPotFEM.UniformSingleLayerThermal2d.System";
    }

    /**
     * @brief Thermal stress tensor.
     */
    const array_type::tensor<double, 4>& SigThermal() const
    {
        return m_TSig;
    }

protected:
    /**
     * @brief Update the cache of stress tensors.
     */
    void updateCache(int64_t index)
    {
        int64_t advance = index - m_cache_start - m_ncache;

        if (advance != 0) {
            m_gen_sig.advance(xt::eval(advance * xt::ones<int64_t>({m_nelem * m_nip})));
            m_gen_phi.advance(xt::eval(advance * xt::ones<int64_t>({m_nelem * m_nip})));
            m_gen_sgn.advance(xt::eval(advance * xt::ones<int64_t>({m_nelem * m_nip})));
        }

        m_cache_start += m_ncache;

        size_t n = static_cast<size_t>(m_ncache);
        auto sig = m_gen_sig.normal({n}, 0, m_T);
        auto phi = m_gen_phi.random({n});
        auto sgn = m_gen_sgn.randint({n}, (int)2);

        for (size_t i = 0; i < n; ++i) {
            for (size_t e = 0; e < m_nelem; ++e) {
                for (size_t q = 0; q < m_nip; ++q) {
                    double fac = 1;
                    if (sgn(i, e, q) == 0) {
                        fac = -1;
                    }
                    m_cache(i, e, q, 0, 0) = -fac * std::sqrt(phi(e, q, i)) * 0.5 * sig(e, q, i);
                    m_cache(i, e, q, 1, 1) = fac * std::sqrt(phi(e, q, i)) * 0.5 * sig(e, q, i);
                    m_cache(i, e, q, 0, 1) = std::sqrt(1 - phi(e, q, i)) * 0.5 * sig(e, q, i);
                    m_cache(i, e, q, 1, 0) = std::sqrt(1 - phi(e, q, i)) * 0.5 * sig(e, q, i);
                }
            }
        }
    }

public:
    void setInc(size_t arg) override
    {
        m_inc = arg;
        m_computed = m_inc;
        this->updateCache(static_cast<int64_t>((m_inc - m_inc % m_dinc) / m_dinc));

        std::copy(m_cache.data(), m_cache.data() + m_TSig.size(), m_TSig.data());
        m_quad.int_gradN_dot_tensor2_dV(m_TSig, m_fe);
        m_vector.assembleNode(m_fe, m_fthermal);
    }

protected:
    void computeThermalForce()
    {
        if (m_inc == m_computed) {
            return;
        }

        if (m_inc % m_dinc != 0) {
            return;
        }

        int64_t index = static_cast<int64_t>(m_inc / m_dinc);

        if (index - m_cache_start >= m_ncache) {
            this->updateCache(index);
        }

        int64_t offset = index - m_cache_start;
        std::copy(m_cache.data() + offset, m_cache.data() + offset + m_TSig.size(), m_TSig.data());
        m_quad.int_gradN_dot_tensor2_dV(m_TSig, m_fe);
        m_vector.assembleNode(m_fe, m_fthermal);
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
