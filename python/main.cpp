/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the GNU Public License (MIT).
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>

#include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>
#include <FrictionQPotFEM/UniformMultiLayerLeverDrive2d.h>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <FrictionQPotFEM/version.h>

namespace py = pybind11;

PYBIND11_MODULE(_FrictionQPotFEM, mod)
{
    xt::import_numpy();

    mod.doc() = "Friction model based on GooseFEM and FrictionQPotFEM";

    namespace M = FrictionQPotFEM;

    mod.def("version", &M::version, "Return version string.");

    // -------------------------
    // FrictionQPotFEM.Generic2d
    // -------------------------

    {

        py::module sub = mod.def_submodule("Generic2d", "Generic2d");

        namespace SUB = FrictionQPotFEM::Generic2d;

        sub.def(
            "version_dependencies",
            &SUB::version_dependencies,
            "Return version information of library and its dependencies.");

        {

            py::class_<SUB::System> cls(sub, "System");

            cls.def(
                py::init<
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&>(),
                "System",
                py::arg("coor"),
                py::arg("conn"),
                py::arg("dofs"),
                py::arg("iip"),
                py::arg("elem_elastic"),
                py::arg("elem_plastic"));

            cls.def(
                "setMassMatrix",
                &SUB::System::setMassMatrix<xt::pytensor<double, 1>>,
                "setMassMatrix",
                py::arg("rho_elem"));

            cls.def(
                "setDampingMatrix",
                &SUB::System::setDampingMatrix<xt::pytensor<double, 1>>,
                "setDampingMatrix",
                py::arg("alpha_elem"));

            cls.def(
                "setElastic",
                &SUB::System::setElastic,
                "setElastic",
                py::arg("K_elem"),
                py::arg("G_elem"));

            cls.def(
                "setPlastic",
                &SUB::System::setPlastic,
                "setPlastic",
                py::arg("K_elem"),
                py::arg("G_elem"),
                py::arg("epsy_elem"));

            cls.def(
                "isHomogeneousElastic", &SUB::System::isHomogeneousElastic, "isHomogeneousElastic");

            cls.def("setDt", &SUB::System::setDt, "setDt", py::arg("dt"));

            cls.def("setT", &SUB::System::setT, "setT", py::arg("t"));

            cls.def("setU", &SUB::System::setU<xt::pytensor<double, 2>>, "setU", py::arg("u"));

            cls.def("setV", &SUB::System::setV<xt::pytensor<double, 2>>, "setV", py::arg("v"));

            cls.def("setA", &SUB::System::setA<xt::pytensor<double, 2>>, "setA", py::arg("a"));

            cls.def(
                "setFext",
                &SUB::System::setFext<xt::pytensor<double, 2>>,
                "setFext",
                py::arg("fext"));

            cls.def("quench", &SUB::System::quench, "quench");

            cls.def("elastic", &SUB::System::elastic, "elastic");

            cls.def("plastic", &SUB::System::plastic, "plastic");

            cls.def("conn", &SUB::System::conn, "conn");

            cls.def("coor", &SUB::System::coor, "coor");

            cls.def("dofs", &SUB::System::dofs, "dofs");

            cls.def("u", &SUB::System::u, "u");

            cls.def("v", &SUB::System::v, "v");

            cls.def("a", &SUB::System::a, "a");

            cls.def(
                "mass", &SUB::System::mass, "mass", py::return_value_policy::reference_internal);

            cls.def(
                "damping",
                &SUB::System::damping,
                "damping",
                py::return_value_policy::reference_internal);

            cls.def("fext", &SUB::System::fext, "fext");

            cls.def("fint", &SUB::System::fint, "fint");

            cls.def("fmaterial", &SUB::System::fmaterial, "fmaterial");

            cls.def("fdamp", &SUB::System::fdamp, "fdamp");

            cls.def("residual", &SUB::System::residual, "residual");

            cls.def("t", &SUB::System::t, "t");

            cls.def("dV", &SUB::System::dV, "dV");

            cls.def(
                "stiffness",
                &SUB::System::stiffness,
                "stiffness",
                py::return_value_policy::reference_internal);

            cls.def(
                "vector",
                &SUB::System::vector,
                "vector",
                py::return_value_policy::reference_internal);

            cls.def(
                "quad", &SUB::System::quad, "quad", py::return_value_policy::reference_internal);

            cls.def(
                "material",
                &SUB::System::material,
                "material",
                py::return_value_policy::reference_internal);

            cls.def("Sig", &SUB::System::Sig, "Sig");

            cls.def("Eps", &SUB::System::Eps, "Eps");

            cls.def("plastic_Sig", &SUB::System::plastic_Sig, "plastic_Sig");

            cls.def("plastic_Eps", &SUB::System::plastic_Eps, "plastic_Eps");

            cls.def(
                "boundcheck_right",
                &SUB::System::boundcheck_right,
                "boundcheck_right",
                py::arg("n"));

            cls.def(
                "plastic_CurrentYieldLeft",
                py::overload_cast<>(&SUB::System::plastic_CurrentYieldLeft, py::const_),
                "plastic_CurrentYieldLeft");

            cls.def(
                "plastic_CurrentYieldRight",
                py::overload_cast<>(&SUB::System::plastic_CurrentYieldRight, py::const_),
                "plastic_CurrentYieldRight");

            cls.def(
                "plastic_CurrentYieldLeft",
                py::overload_cast<size_t>(&SUB::System::plastic_CurrentYieldLeft, py::const_),
                "plastic_CurrentYieldLeft",
                py::arg("offset"));

            cls.def(
                "plastic_CurrentYieldRight",
                py::overload_cast<size_t>(&SUB::System::plastic_CurrentYieldRight, py::const_),
                "plastic_CurrentYieldRight",
                py::arg("offset"));

            cls.def(
                "plastic_CurrentIndex", &SUB::System::plastic_CurrentIndex, "plastic_CurrentIndex");

            cls.def("plastic_Epsp", &SUB::System::plastic_Epsp, "plastic_Epsp");

            cls.def("timeStep", &SUB::System::timeStep, "timeStep");

            cls.def(
                "timeStepsUntilEvent",
                &SUB::System::timeStepsUntilEvent,
                "timeStepsUntilEvent",
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 1000000);

            cls.def(
                "minimise",
                &SUB::System::minimise,
                "minimise",
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 1000000);

            cls.def(
                "minimise_boundcheck",
                &SUB::System::minimise_boundcheck,
                "minimise_boundcheck",
                py::arg("nmargin") = 5,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 1000000);

            cls.def("__repr__", [](const SUB::System&) {
                return "<FrictionQPotFEM.Generic2d.System>";
            });
        }

        {

            py::class_<SUB::HybridSystem, SUB::System> cls(sub, "HybridSystem");

            cls.def(
                py::init<
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&>(),
                "HybridSystem",
                py::arg("coor"),
                py::arg("conn"),
                py::arg("dofs"),
                py::arg("iip"),
                py::arg("elem_elastic"),
                py::arg("elem_plastic"));

            cls.def(
                "setElastic",
                &SUB::HybridSystem::setElastic,
                "setElastic",
                py::arg("K_elem"),
                py::arg("G_elem"));

            cls.def(
                "setPlastic",
                &SUB::HybridSystem::setPlastic,
                "setPlastic",
                py::arg("K_elem"),
                py::arg("G_elem"),
                py::arg("epsy_elem"));

            cls.def("evalSystem", &SUB::HybridSystem::evalSystem, "evalSystem");

            cls.def(
                "material_elastic",
                &SUB::HybridSystem::material_elastic,
                "material_elastic",
                py::return_value_policy::reference_internal);

            cls.def(
                "material_plastic",
                &SUB::HybridSystem::material_plastic,
                "material_plastic",
                py::return_value_policy::reference_internal);

            cls.def("Sig", &SUB::HybridSystem::Sig, "Sig");

            cls.def("Eps", &SUB::HybridSystem::Eps, "Eps");

            cls.def("plastic_Sig", &SUB::HybridSystem::plastic_Sig, "plastic_Sig");

            cls.def("plastic_Eps", &SUB::HybridSystem::plastic_Eps, "plastic_Eps");

            cls.def("__repr__", [](const SUB::System&) {
                return "<FrictionQPotFEM.Generic2d.HybridSystem>";
            });
        }
    }

    // ------------------------------------
    // FrictionQPotFEM.UniformSingleLayer2d
    // ------------------------------------

    {

        py::module sub = mod.def_submodule("UniformSingleLayer2d", "UniformSingleLayer2d");

        namespace SUB = FrictionQPotFEM::UniformSingleLayer2d;

        sub.def(
            "version_dependencies",
            &SUB::version_dependencies,
            "Return version information of library and its dependencies.");

        {

            py::class_<SUB::System, FrictionQPotFEM::Generic2d::HybridSystem> cls(sub, "System");

            cls.def(
                py::init<
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&>(),
                "System",
                py::arg("coor"),
                py::arg("conn"),
                py::arg("dofs"),
                py::arg("iip"),
                py::arg("elem_elastic"),
                py::arg("elem_plastic"));

            cls.def(
                "plastic_signOfPerturbation",
                &SUB::System::plastic_signOfPerturbation,
                "plastic_signOfPerturbation",
                py::arg("delta_u"));

            cls.def(
                "addAffineSimpleShear",
                &SUB::System::addAffineSimpleShear,
                "addAffineSimpleShear",
                py::arg("delta_gamma"));

            cls.def(
                "addAffineSimpleShearCentered",
                &SUB::System::addAffineSimpleShearCentered,
                "addAffineSimpleShearCentered",
                py::arg("delta_gamma"));

            cls.def(
                "addSimpleShearEventDriven",
                &SUB::System::addSimpleShearEventDriven,
                "addSimpleShearEventDriven",
                py::arg("deps"),
                py::arg("kick"),
                py::arg("direction") = +1.0,
                py::arg("dry_run") = false);

            cls.def(
                "addSimpleShearToFixedStress",
                &SUB::System::addSimpleShearToFixedStress,
                "addSimpleShearToFixedStress",
                py::arg("stress"),
                py::arg("dry_run") = false);

            cls.def(
                "addElasticSimpleShearToFixedStress",
                &SUB::System::addElasticSimpleShearToFixedStress,
                "addElasticSimpleShearToFixedStress",
                py::arg("stress"),
                py::arg("dry_run") = false);

            cls.def(
                "triggerElementWithLocalSimpleShear",
                &SUB::System::triggerElementWithLocalSimpleShear,
                "triggerElementWithLocalSimpleShear",
                py::arg("deps"),
                py::arg("plastic_element"),
                py::arg("trigger_weakest_integration_point") = true,
                py::arg("amplify") = 1.0);

            cls.def(
                "plastic_ElementYieldBarrierForSimpleShear",
                &SUB::System::plastic_ElementYieldBarrierForSimpleShear,
                "plastic_ElementYieldBarrierForSimpleShear",
                py::arg("deps_kick") = 0.0,
                py::arg("iquad") = 0);

            cls.def("__repr__", [](const SUB::System&) {
                return "<FrictionQPotFEM.UniformSingleLayer2d.System>";
            });
        }

        {

            py::class_<SUB::LocalTriggerFineLayerFull> cls(sub, "LocalTriggerFineLayerFull");

            cls.def(py::init<const SUB::System&>(), "LocalTriggerFineLayerFull");

            cls.def(
                "setState",
                &SUB::LocalTriggerFineLayerFull::setState,
                py::arg("Eps"),
                py::arg("Sig"),
                py::arg("epsy"),
                py::arg("ntest") = 100);

            cls.def(
                "setStateMinimalSearch",
                &SUB::LocalTriggerFineLayerFull::setStateMinimalSearch,
                py::arg("Eps"),
                py::arg("Sig"),
                py::arg("epsy"));

            cls.def(
                "setStateSimpleShear",
                &SUB::LocalTriggerFineLayerFull::setStateSimpleShear,
                py::arg("Eps"),
                py::arg("Sig"),
                py::arg("epsy"));

            cls.def("barriers", &SUB::LocalTriggerFineLayerFull::barriers);
            cls.def("p", &SUB::LocalTriggerFineLayerFull::p);
            cls.def("s", &SUB::LocalTriggerFineLayerFull::s);
            cls.def("delta_u", &SUB::LocalTriggerFineLayerFull::delta_u);
            cls.def("u_s", &SUB::LocalTriggerFineLayerFull::u_s);
            cls.def("u_p", &SUB::LocalTriggerFineLayerFull::u_p);
            cls.def("Eps_s", &SUB::LocalTriggerFineLayerFull::Eps_s);
            cls.def("Eps_p", &SUB::LocalTriggerFineLayerFull::Eps_p);
            cls.def("Sig_s", &SUB::LocalTriggerFineLayerFull::Sig_s);
            cls.def("Sig_p", &SUB::LocalTriggerFineLayerFull::Sig_p);
            cls.def("dgamma", &SUB::LocalTriggerFineLayerFull::dgamma);
            cls.def("dE", &SUB::LocalTriggerFineLayerFull::dE);

            cls.def("__repr__", [](const SUB::LocalTriggerFineLayerFull&) {
                return "<FrictionQPotFEM.UniformSingleLayer2d.LocalTriggerFineLayerFull>";
            });
        }

        {

            py::class_<SUB::LocalTriggerFineLayer, SUB::LocalTriggerFineLayerFull> cls(
                sub, "LocalTriggerFineLayer");

            cls.def(
                py::init<const SUB::System&, size_t>(),
                "LocalTriggerFineLayer",
                py::arg("sys"),
                py::arg("roi") = 5);

            cls.def("Eps_s", &SUB::LocalTriggerFineLayer::Eps_s);
            cls.def("Eps_p", &SUB::LocalTriggerFineLayer::Eps_p);
            cls.def("Sig_s", &SUB::LocalTriggerFineLayer::Sig_s);
            cls.def("Sig_p", &SUB::LocalTriggerFineLayer::Sig_p);

            cls.def("__repr__", [](const SUB::LocalTriggerFineLayer&) {
                return "<FrictionQPotFEM.UniformSingleLayer2d.LocalTriggerFineLayer>";
            });
        }
    }

    // --------------------------------------------------
    // FrictionQPotFEM.UniformMultiLayerIndividualDrive2d
    // --------------------------------------------------

    {

        py::module sub = mod.def_submodule(
            "UniformMultiLayerIndividualDrive2d", "UniformMultiLayerIndividualDrive2d");

        namespace SUB = FrictionQPotFEM::UniformMultiLayerIndividualDrive2d;

        sub.def(
            "version_dependencies",
            &SUB::version_dependencies,
            "Return version information of library and its dependencies.");

        py::class_<SUB::System, FrictionQPotFEM::Generic2d::HybridSystem> cls(sub, "System");

        cls.def(
            py::init<
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
            py::arg("layer_is_plastic"));

        cls.def("nlayer", &SUB::System::nlayer, "nlayer");

        cls.def("layerNodes", &SUB::System::layerNodes, "layerNodes", py::arg("i"));

        cls.def("layerElements", &SUB::System::layerElements, "layerElements", py::arg("i"));

        cls.def("layerIsPlastic", &SUB::System::layerIsPlastic, "layerIsPlastic");

        cls.def(
            "layerSetDriveStiffness",
            &SUB::System::layerSetDriveStiffness,
            "layerSetDriveStiffness",
            py::arg("k"),
            py::arg("symmetric") = true);

        cls.def(
            "layerSetTargetActive",
            &SUB::System::layerSetTargetActive<xt::pytensor<double, 2>>,
            "layerSetTargetActive",
            py::arg("active"));

        cls.def("layerUbar", &SUB::System::layerUbar, "layerUbar");

        cls.def("layerTargetUbar", &SUB::System::layerTargetUbar, "layerTargetUbar");

        cls.def("layerTargetActive", &SUB::System::layerTargetActive, "layerTargetActive");

        cls.def(
            "layerSetTargetUbar",
            &SUB::System::layerSetTargetUbar<xt::xtensor<double, 2>>,
            "layerSetTargetUbar",
            py::arg("ubar"));

        cls.def(
            "layerSetUbar",
            &SUB::System::layerSetUbar<xt::xtensor<double, 2>, xt::xtensor<bool, 2>>,
            "layerSetUbar",
            py::arg("ubar"),
            py::arg("prescribe"));

        cls.def(
            "addAffineSimpleShear",
            &SUB::System::addAffineSimpleShear,
            "addAffineSimpleShear",
            py::arg("delta_gamma"));

        cls.def(
            "layerTagetUbar_addAffineSimpleShear",
            &SUB::System::layerTagetUbar_addAffineSimpleShear<xt::pytensor<double, 1>>,
            "layerTagetUbar_addAffineSimpleShear",
            py::arg("delta_gamma"),
            py::arg("height"));

        cls.def("fdrive", &SUB::System::fdrive, "fdrive");

        cls.def("layerFdrive", &SUB::System::layerFdrive, "layerFdrive");

        cls.def("__repr__", [](const SUB::System&) {
            return "<FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System>";
        });
    }

    // --------------------------------------------------
    // FrictionQPotFEM.UniformMultiLayerLeverDrive2d
    // --------------------------------------------------

    {

        py::module sub =
            mod.def_submodule("UniformMultiLayerLeverDrive2d", "UniformMultiLayerLeverDrive2d");

        namespace SUB = FrictionQPotFEM::UniformMultiLayerLeverDrive2d;

        sub.def(
            "version_dependencies",
            &SUB::version_dependencies,
            "Return version information of library and its dependencies.");

        py::class_<SUB::System, FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System> cls(
            sub, "System");

        cls.def(
            py::init<
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
            py::arg("layer_is_plastic"));

        cls.def(
            "setLeverProperties",
            &SUB::System::setLeverProperties<xt::pytensor<double, 1>>,
            "setLeverProperties",
            py::arg("H"),
            py::arg("hi"));

        cls.def("setLeverTarget", &SUB::System::setLeverTarget, "setLeverTarget", py::arg("u"));

        cls.def("leverTarget", &SUB::System::leverTarget, "leverTarget");

        cls.def("leverPosition", &SUB::System::leverPosition, "leverPosition");

        cls.def("__repr__", [](const SUB::System&) {
            return "<FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System>";
        });
    }
}
