/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#ifndef FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_INTERFACEDAMPING_HPP
#define FRICTIONQPOTFEM_UNIFORMSINGLELAYER2D_INTERFACEDAMPING_HPP

#include "UniformSingleLayer2d_InterfaceDamping.h"

namespace FrictionQPotFEM {
namespace UniformSingleLayer2d_InterfaceDamping {

inline std::vector<std::string> version_dependencies()
{
    return Generic2d::version_dependencies();
}

template <class C, class E, class L>
inline System::System(
    const C& coor,
    const E& conn,
    const E& dofs,
    const L& iip,
    const L& elem_elastic,
    const L& elem_plastic)
{
    this->initHybridSystem(coor, conn, dofs, iip, elem_elastic, elem_plastic);
    m_eta = 0.0;
    m_fvisco = m_vector.allocate_nodevec(0.0);
    m_Epsdot_plas = m_quad_plas.allocate_qtensor<2>(0.0);
}

inline std::string System::type() const
{
    return "FrictionQPotFEM.UniformSingleLayer2d_InterfaceDamping.System";
}

inline void System::evalAllSet()
{
    m_allset = m_set_M && (m_set_D || m_set_visco) && m_set_elas && m_set_plas && m_dt > 0.0;
}

inline void System::updated_v()
{
    // m_ue_plas is a temporary that can be reused
    m_vector_plas.asElement(m_v, m_ue_plas);
    m_quad_plas.symGradN_vector(m_ue_plas, m_Epsdot_plas);

    // m_fe_plas is a temporary that can be reused
    m_quad_plas.int_gradN_dot_tensor2_dV(xt::eval(m_Epsdot_plas * m_eta), m_fe_plas);
    m_vector_plas.assembleNode(m_fe_plas, m_fvisco);

    if (m_set_D) {
        Generic2d::HybridSystem::updated_v();
        m_fdamp += m_fvisco;
    }
}

inline void System::setEta(double eta)
{
    m_set_visco = true;
    m_eta = eta;
}

inline auto System::fvisco() const
{
    return m_fvisco;
}

} // namespace UniformSingleLayer2d_InterfaceDamping
} // namespace FrictionQPotFEM

#endif
