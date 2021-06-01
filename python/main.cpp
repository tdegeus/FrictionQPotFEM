/*

(c - MIT) T.W.J. de Geus (Tom) | www.geus.me | github.com/tdegeus/FrictionQPotFEM

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <pyxtensor/pyxtensor.hpp>

// #define QPOT_ENABLE_ASSERT
// #define GMATELASTOPLASTICQPOT_ENABLE_ASSERT
// #define FRICTIONQPOTFEM_ENABLE_ASSERT
// #define FRICTIONQPOTFEM_ENABLE_WARNING_PYTHON

#include <FrictionQPotFEM/version.h>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>

namespace py = pybind11;

PYBIND11_MODULE(FrictionQPotFEM, m)
{
     xt::import_numpy();

    m.doc() = "Friction model based on GooseFEM and FrictionQPotFEM";

    namespace M = FrictionQPotFEM;

    m.def("version",
          &M::version,
          "Return version string.");

    // -------------------------
    // FrictionQPotFEM.Generic2d
    // -------------------------

    {

    py::module sm = m.def_submodule("Generic2d", "Generic2d");

    namespace SM = FrictionQPotFEM::Generic2d;

    sm.def("version_dependencies",
           &SM::version_dependencies,
           "Return version information of library and its dependencies.");

    py::class_<SM::System>(sm, "System")

        .def(py::init<
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

        .def("setMassMatrix",
             &SM::System::setMassMatrix,
             "setMassMatrix",
             py::arg("rho_elem"))

        .def("setDampingMatrix",
             &SM::System::setDampingMatrix,
             "setDampingMatrix",
             py::arg("alpha_elem"))

        .def("setElastic",
             &SM::System::setElastic,
             "setElastic",
             py::arg("K_elem"),
             py::arg("G_elem"))

        .def("setPlastic",
             &SM::System::setPlastic,
             "setPlastic",
             py::arg("K_elem"),
             py::arg("G_elem"),
             py::arg("epsy_elem"))

        .def("isHomogeneousElastic", &SM::System::isHomogeneousElastic, "isHomogeneousElastic")
        .def("setDt", &SM::System::setDt, "setDt", py::arg("dt"))
        .def("setU", &SM::System::setU<xt::pytensor<double, 2>>, "setU", py::arg("u"))
        .def("setV", &SM::System::setV<xt::pytensor<double, 2>>, "setV", py::arg("v"))
        .def("setA", &SM::System::setA<xt::pytensor<double, 2>>, "setA", py::arg("a"))
        .def("setFext", &SM::System::setFext<xt::pytensor<double, 2>>, "setFext", py::arg("fext"))
        .def("quench", &SM::System::quench, "quench")
        .def("elastic", &SM::System::elastic, "elastic")
        .def("plastic", &SM::System::plastic, "plastic")
        .def("conn", &SM::System::conn, "conn")
        .def("coor", &SM::System::coor, "coor")
        .def("dofs", &SM::System::dofs, "dofs")
        .def("u", &SM::System::u, "u")
        .def("v", &SM::System::v, "v")
        .def("a", &SM::System::a, "a")
        .def("mass", &SM::System::mass, "mass", py::return_value_policy::reference_internal)
        .def("damping", &SM::System::damping, "damping", py::return_value_policy::reference_internal)
        .def("fext", &SM::System::fext, "fext")
        .def("fint", &SM::System::fint, "fint")
        .def("fmaterial", &SM::System::fmaterial, "fmaterial")
        .def("fdamp", &SM::System::fdamp, "fdamp")
        .def("residual", &SM::System::residual, "residual")
        .def("t", &SM::System::t, "t")
        .def("dV", &SM::System::dV, "dV")
        .def("stiffness", &SM::System::stiffness, "stiffness", py::return_value_policy::reference_internal)
        .def("vector", &SM::System::vector, "vector", py::return_value_policy::reference_internal)
        .def("quad", &SM::System::quad, "quad", py::return_value_policy::reference_internal)
        .def("material", &SM::System::material, "material", py::return_value_policy::reference_internal)
        .def("Sig", &SM::System::Sig, "Sig")
        .def("Eps", &SM::System::Eps, "Eps")
        .def("plastic_Sig", &SM::System::plastic_Sig, "plastic_Sig")
        .def("plastic_Eps", &SM::System::plastic_Eps, "plastic_Eps")

        .def("plastic_CurrentYieldLeft",
             py::overload_cast<>(&SM::System::plastic_CurrentYieldLeft, py::const_),
             "plastic_CurrentYieldLeft")

        .def("plastic_CurrentYieldRight",
             py::overload_cast<>(&SM::System::plastic_CurrentYieldRight, py::const_),
             "plastic_CurrentYieldRight")

        .def("plastic_CurrentYieldLeft",
             py::overload_cast<size_t>(&SM::System::plastic_CurrentYieldLeft, py::const_),
             "plastic_CurrentYieldLeft",
             py::arg("offset"))

        .def("plastic_CurrentYieldRight",
             py::overload_cast<size_t>(&SM::System::plastic_CurrentYieldRight, py::const_),
             "plastic_CurrentYieldRight",
             py::arg("offset"))

        .def("plastic_CurrentIndex",
             &SM::System::plastic_CurrentIndex,
             "plastic_CurrentIndex")

        .def("plastic_Epsp",
             &SM::System::plastic_Epsp,
             "plastic_Epsp")

        .def("timeStep", &SM::System::timeStep, "timeStep")

        .def("minimise",
             &SM::System::minimise,
             "minimise",
             py::arg("tol") = 1e-5,
             py::arg("niter_tol") = 20,
             py::arg("max_iter") = 1000000)

        .def("__repr__", [](const SM::System&) {
            return "<FrictionQPotFEM.Generic2d.System>";
        });

    py::class_<SM::HybridSystem, SM::System>(sm, "HybridSystem")

        .def(py::init<
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

        .def("setElastic",
             &SM::HybridSystem::setElastic,
             "setElastic",
             py::arg("K_elem"),
             py::arg("G_elem"))

        .def("setPlastic",
             &SM::HybridSystem::setPlastic,
             "setPlastic",
             py::arg("K_elem"),
             py::arg("G_elem"),
             py::arg("epsy_elem"))

        .def("evalSystem", &SM::HybridSystem::evalSystem, "evalSystem")

        .def("material_elastic",
            &SM::HybridSystem::material_elastic,
            "material_elastic",
            py::return_value_policy::reference_internal)

        .def("material_plastic",
            &SM::HybridSystem::material_plastic,
            "material_plastic",
            py::return_value_policy::reference_internal)

        .def("Sig", &SM::HybridSystem::Sig, "Sig")
        .def("Eps", &SM::HybridSystem::Eps, "Eps")
        .def("plastic_Sig", &SM::HybridSystem::plastic_Sig, "plastic_Sig")
        .def("plastic_Eps", &SM::HybridSystem::plastic_Eps, "plastic_Eps")

        .def("__repr__", [](const SM::System&) {
            return "<FrictionQPotFEM.Generic2d.HybridSystem>";
        });

    }

    // ------------------------------------
    // FrictionQPotFEM.UniformSingleLayer2d
    // ------------------------------------

    {

    py::module sm = m.def_submodule("UniformSingleLayer2d", "UniformSingleLayer2d");

    namespace SM = FrictionQPotFEM::UniformSingleLayer2d;

    sm.def("version_dependencies",
           &SM::version_dependencies,
           "Return version information of library and its dependencies.");

    py::class_<SM::System, FrictionQPotFEM::Generic2d::HybridSystem>(sm, "System")

        .def(py::init<
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

        .def("plastic_signOfPerturbation",
             &SM::System::plastic_signOfPerturbation,
             "plastic_signOfPerturbation",
             py::arg("delta_u"))

        .def("addAffineSimpleShear",
             &SM::System::addAffineSimpleShear,
             "addAffineSimpleShear",
             py::arg("delta_gamma"))

        .def("addAffineSimpleShearCentered",
             &SM::System::addAffineSimpleShearCentered,
             "addAffineSimpleShearCentered",
             py::arg("delta_gamma"))

        .def("addSimpleShearEventDriven",
             &SM::System::addSimpleShearEventDriven,
             "addSimpleShearEventDriven",
             py::arg("deps"),
             py::arg("kick"),
             py::arg("direction") = +1.0,
             py::arg("dry_run") = false)

        .def("addSimpleShearToFixedStress",
             &SM::System::addSimpleShearToFixedStress,
             "addSimpleShearToFixedStress",
             py::arg("stress"),
             py::arg("dry_run") = false)

        .def("triggerElementWithLocalSimpleShear",
             &SM::System::triggerElementWithLocalSimpleShear,
             "triggerElementWithLocalSimpleShear",
             py::arg("deps"),
             py::arg("plastic_element"),
             py::arg("trigger_weakest_integration_point") = true,
             py::arg("amplify") = 1.0)

        .def("plastic_ElementYieldBarrierForSimpleShear",
             &SM::System::plastic_ElementYieldBarrierForSimpleShear,
             "plastic_ElementYieldBarrierForSimpleShear",
             py::arg("deps_kick") = 0.0,
             py::arg("iquad") = 0)

        .def("__repr__", [](const SM::System&) {
            return "<FrictionQPotFEM.UniformSingleLayer2d.System>";
        });

    py::class_<SM::LocalTriggerFineLayerFull>(sm, "LocalTriggerFineLayerFull")

        .def(py::init<const SM::System&>(), "LocalTriggerFineLayerFull")

        .def("setState",
             &SM::LocalTriggerFineLayerFull::setState,
             py::arg("Eps"),
             py::arg("Sig"),
             py::arg("epsy"),
             py::arg("ntest") = 100)

        .def("setStateMinimalSearch",
             &SM::LocalTriggerFineLayerFull::setStateMinimalSearch,
             py::arg("Eps"),
             py::arg("Sig"),
             py::arg("epsy"))

        .def("setStateSimpleShear",
             &SM::LocalTriggerFineLayerFull::setStateSimpleShear,
             py::arg("Eps"),
             py::arg("Sig"),
             py::arg("epsy"))

        .def("barriers", &SM::LocalTriggerFineLayerFull::barriers)
        .def("p", &SM::LocalTriggerFineLayerFull::p)
        .def("s", &SM::LocalTriggerFineLayerFull::s)
        .def("delta_u", &SM::LocalTriggerFineLayerFull::delta_u)
        .def("u_s", &SM::LocalTriggerFineLayerFull::u_s)
        .def("u_p", &SM::LocalTriggerFineLayerFull::u_p)
        .def("Eps_s", &SM::LocalTriggerFineLayerFull::Eps_s)
        .def("Eps_p", &SM::LocalTriggerFineLayerFull::Eps_p)
        .def("Sig_s", &SM::LocalTriggerFineLayerFull::Sig_s)
        .def("Sig_p", &SM::LocalTriggerFineLayerFull::Sig_p)
        .def("dgamma", &SM::LocalTriggerFineLayerFull::dgamma)
        .def("dE", &SM::LocalTriggerFineLayerFull::dE)

        .def("__repr__", [](const SM::LocalTriggerFineLayerFull&) {
            return "<FrictionQPotFEM.UniformSingleLayer2d.LocalTriggerFineLayerFull>";
        });

    py::class_<SM::LocalTriggerFineLayer, SM::LocalTriggerFineLayerFull>(sm, "LocalTriggerFineLayer")

        .def(py::init<const SM::System&, size_t>(),
             "LocalTriggerFineLayer",
             py::arg("sys"),
             py::arg("roi") = 5)

        .def("Eps_s", &SM::LocalTriggerFineLayer::Eps_s)
        .def("Eps_p", &SM::LocalTriggerFineLayer::Eps_p)
        .def("Sig_s", &SM::LocalTriggerFineLayer::Sig_s)
        .def("Sig_p", &SM::LocalTriggerFineLayer::Sig_p)

        .def("__repr__", [](const SM::LocalTriggerFineLayer&) {
            return "<FrictionQPotFEM.UniformSingleLayer2d.LocalTriggerFineLayer>";
        });

    }

    // --------------------------------------------------
    // FrictionQPotFEM.UniformMultiLayerIndividualDrive2d
    // --------------------------------------------------

    {

    py::module sm = m.def_submodule("UniformMultiLayerIndividualDrive2d", "UniformMultiLayerIndividualDrive2d");

    namespace SM = FrictionQPotFEM::UniformMultiLayerIndividualDrive2d;

    sm.def("version_dependencies",
           &SM::version_dependencies,
           "Return version information of library and its dependencies.");

    py::class_<SM::System, FrictionQPotFEM::Generic2d::HybridSystem>(sm, "System")

        .def(py::init<
                const xt::xtensor<double, 2>&,
                const xt::xtensor<size_t, 2>&,
                const xt::xtensor<size_t, 2>&,
                const xt::xtensor<size_t, 1>&,
                const std::vector<xt::xtensor<size_t, 1>>&,
                const std::vector<xt::xtensor<size_t, 1>>&,
                const xt::xtensor<bool, 1>&>(),
             "System",
             py::arg("coor"),
             py::arg("conn"),
             py::arg("dofs"),
             py::arg("iip"),
             py::arg("elem"),
             py::arg("node"),
             py::arg("layer_is_plastic"))

        .def("layerNodes",
             &SM::System::layerNodes,
             "layerNodes",
             py::arg("i"))

        .def("layerElements",
             &SM::System::layerElements,
             "layerElements",
             py::arg("i"))

        .def("layerIsPlastic",
             &SM::System::layerIsPlastic,
             "layerIsPlastic",
             py::arg("i"))

        .def("setDriveStiffness",
             &SM::System::setDriveStiffness,
             "setDriveStiffness",
             py::arg("k"),
             py::arg("symmetric") = true)

        .def("layerUbar",
             &SM::System::layerUbar,
             "layerUbar")

        .def("layerSetUbar",
             &SM::System::layerSetUbar<xt::xtensor<double, 2>, xt::xtensor<bool, 2>>,
             "layerSetUbar",
             py::arg("ubar"),
             py::arg("prescribe"))

        .def("layerSetDistributeUbar",
             &SM::System::layerSetDistributeUbar<xt::xtensor<double, 2>, xt::xtensor<bool, 2>>,
             "layerSetDistributeUbar",
             py::arg("ubar"),
             py::arg("prescribe"))

        .def("addAffineSimpleShear",
             &SM::System::addAffineSimpleShear<xt::xtensor<bool, 2>, xt::xtensor<double, 1>>,
             "addAffineSimpleShear",
             py::arg("delta_gamma"),
             py::arg("prescribe"),
             py::arg("height"))

        .def("fdrive",
             &SM::System::fdrive,
             "fdrive")

        .def("fdrivespring",
             &SM::System::fdrivespring,
             "fdrivespring")

        .def("__repr__", [](const SM::System&) {
            return "<FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System>";
        });

    }
}
