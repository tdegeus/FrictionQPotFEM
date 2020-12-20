/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM

*/

#include <pybind11/pybind11.h>
#include <pyxtensor/pyxtensor.hpp>

// #define QPOT_ENABLE_ASSERT
// #define GMATELASTOPLASTICQPOT_ENABLE_ASSERT
// #define FRICTIONQPOTFEM_ENABLE_ASSERT

#include <FrictionQPotFEM/UniformSingleLayer2d.h>

namespace py = pybind11;

PYBIND11_MODULE(FrictionQPotFEM, m)
{

    m.doc() = "Friction model based on GooseFEM and FrictionQPotFEM";

    // ---------------------------------
    // FrictionQPotFEM.Cartesian2d
    // ---------------------------------

    py::module sm = m.def_submodule("UniformSingleLayer2d", "UniformSingleLayer2d");

    namespace SM = FrictionQPotFEM::UniformSingleLayer2d;

    // sm.def("versionInfo", &SM::versionInfo, "Return version information.");

    py::class_<SM::System>(sm, "System")

        .def(
            py::init<
                const xt::xtensor<double, 2>&,
                const xt::xtensor<size_t, 2>&,
                const xt::xtensor<size_t, 2>&,
                const xt::xtensor<size_t, 1>&,
                const xt::xtensor<size_t, 1>&,
                const xt::xtensor<size_t, 1>&>(),
            "System",
            py::arg("coor"),
            py::arg("conn"),
            py::arg("dofs"),
            py::arg("iip"),
            py::arg("elem_elastic"),
            py::arg("elem_plastic"))

        .def(
            "setMassMatrix",
            &SM::System::setMassMatrix,
            "setMassMatrix",
            py::arg("rho_elem"))

        .def(
            "setDampingMatrix",
            &SM::System::setDampingMatrix,
            "setDampingMatrix",
            py::arg("alpha_elem"))

        .def(
            "setElastic",
            &SM::System::setElastic,
            "setElastic",
            py::arg("K_elem"),
            py::arg("G_elem"))

        .def(
            "setPlastic",
            &SM::System::setPlastic,
            "setPlastic",
            py::arg("K_elem"),
            py::arg("G_elem"),
            py::arg("epsy_elem"))

        .def("setDt", &SM::System::setDt, "setDt", py::arg("dt"))
        .def("setU", &SM::System::setU, "setU", py::arg("u"))
        .def("quench", &SM::System::quench, "quench")
        .def("elastic", &SM::System::elastic, "elastic")
        .def("plastic", &SM::System::plastic, "plastic")
        .def("conn", &SM::System::conn, "conn")
        .def("coor", &SM::System::coor, "coor")
        .def("dofs", &SM::System::dofs, "dofs")
        .def("u", &SM::System::u, "u")
        .def("fmaterial", &SM::System::fmaterial, "fmaterial")
        .def("residual", &SM::System::residual, "residual")
        .def("t", &SM::System::t, "t")
        .def("dV", &SM::System::dV, "dV")
        .def("vector", &SM::System::vector, "vector")
        .def("quad", &SM::System::quad, "quad")
        .def("material", &SM::System::material, "material")
        .def("Eps", &SM::System::Eps, "Eps")
        .def("Sig", &SM::System::Sig, "Sig")
        .def("timeStep", &SM::System::timeStep, "timeStep")

        .def("plastic_Eps", &SM::System::plastic_Eps, "plastic_Eps")
        .def("plastic_Sig", &SM::System::plastic_Sig, "plastic_Sig")

        .def(
            "plastic_CurrentYieldLeft",
            py::overload_cast<>(&SM::System::plastic_CurrentYieldLeft, py::const_),
            "plastic_CurrentYieldLeft")

        .def(
            "plastic_CurrentYieldRight",
            py::overload_cast<>(&SM::System::plastic_CurrentYieldRight, py::const_),
            "plastic_CurrentYieldRight")

        .def(
            "plastic_CurrentYieldLeft",
            py::overload_cast<size_t>(&SM::System::plastic_CurrentYieldLeft, py::const_),
            "plastic_CurrentYieldLeft",
            py::arg("offset"))

        .def(
            "plastic_CurrentYieldRight",
            py::overload_cast<size_t>(&SM::System::plastic_CurrentYieldRight, py::const_),
            "plastic_CurrentYieldRight",
            py::arg("offset"))

        .def(
            "plastic_CurrentIndex",
            &SM::System::plastic_CurrentIndex,
            "plastic_CurrentIndex")

        .def("__repr__", [](const SM::System&) {
            return "<FrictionQPotFEM.UniformSingleLayer2d.System>";
        });

    py::class_<SM::HybridSystem, SM::System>(sm, "HybridSystem")

        .def(
            py::init<
                const xt::xtensor<double, 2>&,
                const xt::xtensor<size_t, 2>&,
                const xt::xtensor<size_t, 2>&,
                const xt::xtensor<size_t, 1>&,
                const xt::xtensor<size_t, 1>&,
                const xt::xtensor<size_t, 1>&>(),
            "HybridSystem",
            py::arg("coor"),
            py::arg("conn"),
            py::arg("dofs"),
            py::arg("iip"),
            py::arg("elem_elastic"),
            py::arg("elem_plastic"))

        .def(
            "setMassMatrix",
            &SM::HybridSystem::setMassMatrix,
            "setMassMatrix",
            py::arg("rho_elem"))

        .def(
            "setDampingMatrix",
            &SM::HybridSystem::setDampingMatrix,
            "setDampingMatrix",
            py::arg("alpha_elem"))

        .def(
            "setElastic",
            &SM::HybridSystem::setElastic,
            "setElastic",
            py::arg("K_elem"),
            py::arg("G_elem"))

        .def(
            "setPlastic",
            &SM::HybridSystem::setPlastic,
            "setPlastic",
            py::arg("K_elem"),
            py::arg("G_elem"),
            py::arg("epsy_elem"))

        .def("setDt", &SM::HybridSystem::setDt, "setDt", py::arg("dt"))
        .def("setU", &SM::HybridSystem::setU, "setU", py::arg("u"))
        .def("quench", &SM::HybridSystem::quench, "quench")
        .def("elastic", &SM::HybridSystem::elastic, "elastic")
        .def("plastic", &SM::HybridSystem::plastic, "plastic")
        .def("conn", &SM::HybridSystem::conn, "conn")
        .def("coor", &SM::HybridSystem::coor, "coor")
        .def("dofs", &SM::HybridSystem::dofs, "dofs")
        .def("u", &SM::HybridSystem::u, "u")
        .def("fmaterial", &SM::HybridSystem::fmaterial, "fmaterial")
        .def("residual", &SM::HybridSystem::residual, "residual")
        .def("t", &SM::HybridSystem::t, "t")
        .def("dV", &SM::HybridSystem::dV, "dV")
        .def("vector", &SM::HybridSystem::vector, "vector")
        .def("quad", &SM::HybridSystem::quad, "quad")
        .def("material", &SM::HybridSystem::material, "material")
        .def("Eps", &SM::HybridSystem::Eps, "Eps")
        .def("Sig", &SM::HybridSystem::Sig, "Sig")
        .def("timeStep", &SM::HybridSystem::timeStep, "timeStep")

        .def("material_elastic", &SM::HybridSystem::material_elastic, "material_elastic")
        .def("material_plastic", &SM::HybridSystem::material_plastic, "material_plastic")
        .def("plastic_Eps", &SM::HybridSystem::plastic_Eps, "plastic_Eps")
        .def("plastic_Sig", &SM::HybridSystem::plastic_Sig, "plastic_Sig")

        .def(
            "plastic_CurrentYieldLeft",
            py::overload_cast<>(&SM::HybridSystem::plastic_CurrentYieldLeft, py::const_),
            "plastic_CurrentYieldLeft")

        .def(
            "plastic_CurrentYieldRight",
            py::overload_cast<>(&SM::HybridSystem::plastic_CurrentYieldRight, py::const_),
            "plastic_CurrentYieldRight")

        .def(
            "plastic_CurrentYieldLeft",
            py::overload_cast<size_t>(&SM::HybridSystem::plastic_CurrentYieldLeft, py::const_),
            "plastic_CurrentYieldLeft",
            py::arg("offset"))

        .def(
            "plastic_CurrentYieldRight",
            py::overload_cast<size_t>(&SM::HybridSystem::plastic_CurrentYieldRight, py::const_),
            "plastic_CurrentYieldRight",
            py::arg("offset"))

        .def(
            "plastic_CurrentIndex",
            &SM::HybridSystem::plastic_CurrentIndex,
            "plastic_CurrentIndex")

        .def(
            "minimise",
            &SM::HybridSystem::minimise,
            "minimise",
            py::arg("tol") = 1e-5,
            py::arg("niter_tol") = 20,
            py::arg("max_iter") = 1000000)

        .def("__repr__", [](const SM::HybridSystem&) {
            return "<FrictionQPotFEM.UniformSingleLayer2d.HybridSystem>";
        });

    py::class_<SM::LocalTriggerFineLayerFull>(sm, "LocalTriggerFineLayerFull")

        .def(py::init<const SM::System&>(), "LocalTriggerFineLayerFull")

        .def(
            "setState",
            &SM::LocalTriggerFineLayerFull::setState,
            py::arg("Eps"),
            py::arg("Sig"),
            py::arg("epsy"),
            py::arg("ntest") = 100)

        .def(
            "setStateMinimalSearch",
            &SM::LocalTriggerFineLayerFull::setStateMinimalSearch,
            py::arg("Eps"),
            py::arg("Sig"),
            py::arg("epsy"))

        .def(
            "setStateSimpleShear",
            &SM::LocalTriggerFineLayerFull::setStateSimpleShear,
            py::arg("Eps"),
            py::arg("Sig"),
            py::arg("epsy"))

        .def("barriers", &SM::LocalTriggerFineLayerFull::barriers)
        .def("delta_u", &SM::LocalTriggerFineLayerFull::delta_u)

        .def("u_s", &SM::LocalTriggerFineLayerFull::u_s)
        .def("u_p", &SM::LocalTriggerFineLayerFull::u_p)
        .def("Eps_s", &SM::LocalTriggerFineLayerFull::Eps_s)
        .def("Eps_p", &SM::LocalTriggerFineLayerFull::Eps_p)
        .def("Sig_s", &SM::LocalTriggerFineLayerFull::Sig_s)
        .def("Sig_p", &SM::LocalTriggerFineLayerFull::Sig_p)

        .def("__repr__", [](const SM::LocalTriggerFineLayerFull&) {
            return "<FrictionQPotFEM.UniformSingleLayer2d.LocalTriggerFineLayerFull>";
        });

    py::class_<SM::LocalTriggerFineLayer>(sm, "LocalTriggerFineLayer")

        .def(
            py::init<const SM::System&, size_t>(),
            "LocalTriggerFineLayer",
            py::arg("sys"),
            py::arg("roi") = 5)

        .def(
            "setState",
            &SM::LocalTriggerFineLayer::setState,
            py::arg("Eps"),
            py::arg("Sig"),
            py::arg("epsy"),
            py::arg("ntest") = 100)

        .def(
            "setStateMinimalSearch",
            &SM::LocalTriggerFineLayer::setStateMinimalSearch,
            py::arg("Eps"),
            py::arg("Sig"),
            py::arg("epsy"))

        .def(
            "setStateSimpleShear",
            &SM::LocalTriggerFineLayer::setStateSimpleShear,
            py::arg("Eps"),
            py::arg("Sig"),
            py::arg("epsy"))

        .def("barriers", &SM::LocalTriggerFineLayer::barriers)
        .def("delta_u", &SM::LocalTriggerFineLayer::delta_u)

        .def("u_s", &SM::LocalTriggerFineLayer::u_s)
        .def("u_p", &SM::LocalTriggerFineLayer::u_p)
        .def("Eps_s", &SM::LocalTriggerFineLayer::Eps_s)
        .def("Eps_p", &SM::LocalTriggerFineLayer::Eps_p)
        .def("Sig_s", &SM::LocalTriggerFineLayer::Sig_s)
        .def("Sig_p", &SM::LocalTriggerFineLayer::Sig_p)

        .def("__repr__", [](const SM::LocalTriggerFineLayer&) {
            return "<FrictionQPotFEM.UniformSingleLayer2d.LocalTriggerFineLayer>";
        });

}
