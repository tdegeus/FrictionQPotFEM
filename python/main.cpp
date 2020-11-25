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

    sm.def("versionInfo", &SM::versionInfo, "Return version information.");

    py::class_<SM::HybridSystem>(sm, "HybridSystem")

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
        .def("material_elastic", &SM::HybridSystem::material_elastic, "material_elastic")
        .def("material_plastic", &SM::HybridSystem::material_plastic, "material_plastic")
        .def("Eps", &SM::HybridSystem::Eps, "Eps")
        .def("Sig", &SM::HybridSystem::Sig, "Sig")
        .def("timeStep", &SM::HybridSystem::timeStep, "timeStep")

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
}
