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
#include <xtensor-python/xtensor_python_config.hpp>

#define FRICTIONQPOTFEM_USE_XTENSOR_PYTHON
#define GMATELASTOPLASTICQPOT_USE_XTENSOR_PYTHON
#define GOOSEFEM_USE_XTENSOR_PYTHON
#define QPOT_USE_XTENSOR_PYTHON
// #include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>
// #include <FrictionQPotFEM/UniformMultiLayerLeverDrive2d.h>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>

namespace py = pybind11;

/**
Overrides the `__name__` of a module.
Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
which affects the `__repr__` of the class type objects.
*/
class ScopedModuleNameOverride {
public:
    explicit ScopedModuleNameOverride(py::module m, std::string name) : module_(std::move(m))
    {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride()
    {
        module_.attr("__name__") = original_name_;
    }

private:
    py::module module_;
    py::object original_name_;
};

    template <class C, class P>
    void register_SystemBase(P& cls)
    {
            cls.def_property_readonly("N", &C::N, "Number of plastic elements");
            cls.def_property_readonly("type", &C::type, "Class identifier");

            // todo
            cls.def(
                "setMassMatrix",
                &C::template setMassMatrix<xt::pytensor<double, 1>>,
                "setMassMatrix",
                py::arg("rho_elem"));

            // todo
            cls.def(
                "setDampingMatrix",
                &C::template setDampingMatrix<xt::pytensor<double, 1>>,
                "setDampingMatrix",
                py::arg("alpha_elem"));

            // todo
            cls.def(
                "setElastic",
                &C::template setElastic<xt::pytensor<double, 1>, xt::pytensor<double, 1>>,
                "setElastic",
                py::arg("K_elem"),
                py::arg("G_elem"));

            // todo
            cls.def(
                "setPlastic",
                &C::template setPlastic<
                    xt::pytensor<double, 1>,
                    xt::pytensor<double, 1>,
                    xt::pytensor<double, 2>>,
                "setPlastic",
                py::arg("K_elem"),
                py::arg("G_elem"),
                py::arg("epsy_elem"));

            // todo
            cls.def(
                "reset_epsy",
                &C::template reset_epsy<xt::pytensor<double, 2>>,
                "reset_epsy",
                py::arg("epsy_elem"));

            cls.def_property_readonly(
                "isHomogeneousElastic",
                &C::isHomogeneousElastic,
                "``True`` is elasticity is the same everywhere");

            cls.def_property(
                "eta",
                &C::eta,
                &C::setEta,
                "Damping proportional to the strain-rate at the interface");

            cls.def_property(
                "alpha",
                &C::alpha,
                &C::setAlpha,
                "Homogeneous background damping density. Otherwise use setDampingMatrix.");

            cls.def_property(
                "rho",
                &C::rho,
                &C::setRho,
                "Homogeneous mass density. Otherwise use setMassMatrix.");

            cls.def_property("dt", &C::dt, &C::setDt, "Time step");

            cls.def_property("t", &C::t, &C::setT, "Current time");

            cls.def("setU", &C::template setU<xt::pytensor<double, 2>>, "setU", py::arg("u"));
            cls.def("setV", &C::template setV<xt::pytensor<double, 2>>, "setV", py::arg("v"));
            cls.def("setA", &C::template setA<xt::pytensor<double, 2>>, "setA", py::arg("a"));

            cls.def(
                "setFext",
                &C::template setFext<xt::pytensor<double, 2>>,
                "setFext",
                py::arg("fext"));

            cls.def("quench", &C::quench, "quench");
            cls.def_property_readonly("elastic", &C::elastic, "elastic");
            cls.def_property_readonly("plastic", &C::plastic, "plastic");
            cls.def_property_readonly("conn", &C::conn, "conn");
            cls.def_property_readonly("coor", &C::coor, "coor");
            cls.def_property_readonly("dofs", &C::dofs, "dofs");
            cls.def_property_readonly("u", &C::u, "u");
            cls.def_property_readonly("v", &C::v, "v");
            cls.def_property_readonly("a", &C::a, "a");

            cls.def_property_readonly("mass", &C::mass, "mass");

            cls.def_property_readonly("damping", &C::damping, "damping");

            cls.def_property_readonly("fext", &C::fext, "fext");
            cls.def_property_readonly("fint", &C::fint, "fint");
            cls.def_property_readonly("fmaterial", &C::fmaterial, "fmaterial");
            cls.def_property_readonly("fdamp", &C::fdamp, "fdamp");
            cls.def("residual", &C::residual, "residual");
            cls.def_property_readonly("dV", &C::dV, "dV");

            cls.def("stiffness", &C::stiffness, "stiffness");

            cls.def_property_readonly("vector", &C::vector, "vector");

            cls.def_property_readonly("quad", &C::quad, "quad");

            cls.def_property_readonly(
                "material_elastic", &C::material_elastic, "material_elastic");

            cls.def_property_readonly(
                "material_plastic", &C::material_plastic, "material_plastic");

            cls.def("K", &C::K, "K");
            cls.def("G", &C::G, "G");
            cls.def_property_readonly("Sig", &C::Sig, "Sig");
            cls.def_property_readonly("Eps", &C::Eps, "Eps");
            cls.def("Epsdot", &C::Epsdot, "Epsdot");
            cls.def("Epsddot", &C::Epsddot, "Epsddot");
            cls.def_property_readonly("plastic_Sig", &C::plastic_Sig, "plastic_Sig");
            cls.def_property_readonly("plastic_Eps", &C::plastic_Eps, "plastic_Eps");
            cls.def("plastic_Epsdot", &C::plastic_Epsdot, "plastic_Epsdot");

            cls.def(
                "boundcheck_right",
                &C::boundcheck_right,
                "boundcheck_right",
                py::arg("n"));

            cls.def(
                "plastic_CurrentYieldLeft",
                py::overload_cast<>(&C::plastic_CurrentYieldLeft, py::const_),
                "plastic_CurrentYieldLeft");

            cls.def(
                "plastic_CurrentYieldRight",
                py::overload_cast<>(&C::plastic_CurrentYieldRight, py::const_),
                "plastic_CurrentYieldRight");

            cls.def(
                "plastic_CurrentYieldLeft",
                py::overload_cast<size_t>(&C::plastic_CurrentYieldLeft, py::const_),
                "plastic_CurrentYieldLeft",
                py::arg("offset"));

            cls.def(
                "plastic_CurrentYieldRight",
                py::overload_cast<size_t>(&C::plastic_CurrentYieldRight, py::const_),
                "plastic_CurrentYieldRight",
                py::arg("offset"));

            cls.def(
                "plastic_CurrentIndex", &C::plastic_CurrentIndex, "plastic_CurrentIndex");

            cls.def("plastic_Epsp", &C::plastic_Epsp, "plastic_Epsp");

            cls.def(
                "eventDriven_setDeltaU",
                &C::template eventDriven_setDeltaU<xt::pytensor<double, 2>>,
                "eventDriven_setDeltaU",
                py::arg("delta_u"),
                py::arg("autoscale") = true);

            cls.def_property_readonly(
                "eventDriven_deltaU", &C::eventDriven_deltaU, "eventDriven_deltaU");

            cls.def(
                "eventDrivenStep",
                &C::eventDrivenStep,
                "eventDrivenStep",
                py::arg("deps"),
                py::arg("kick"),
                py::arg("direction") = +1,
                py::arg("yield_element") = false,
                py::arg("iterative") = false);

            cls.def("timeStep", &C::timeStep, "timeStep");
            cls.def("timeSteps", &C::timeSteps, "timeSteps");

            cls.def(
                "timeSteps_residualcheck",
                &C::timeSteps_residualcheck,
                "timeSteps_residualcheck",
                py::arg("n"),
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20);

            cls.def(
                "timeSteps_boundcheck",
                &C::timeSteps_boundcheck,
                "timeSteps_boundcheck",
                py::arg("n"),
                py::arg("nmargin") = 5);

            cls.def(
                "flowSteps",
                &C::template flowSteps<xt::pytensor<double, 2>>,
                "flowSteps",
                py::arg("n"),
                py::arg("v"));

            cls.def(
                "flowSteps_boundcheck",
                &C::template flowSteps_boundcheck<xt::pytensor<double, 2>>,
                "flowSteps_boundcheck",
                py::arg("n"),
                py::arg("v"),
                py::arg("nmargin") = 5);

            cls.def(
                "timeStepsUntilEvent",
                &C::timeStepsUntilEvent,
                "timeStepsUntilEvent",
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 10000000);

            cls.def(
                "minimise",
                &C::minimise,
                "minimise",
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 10000000);

            cls.def(
                "minimise_boundcheck",
                &C::minimise_boundcheck,
                "minimise_boundcheck",
                py::arg("nmargin") = 5,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 10000000);

            cls.def(
                "minimise_truncate",
                static_cast<size_t (C::*)(size_t, size_t, double, size_t, size_t)>(
                    &C::minimise_truncate),
                "minimise_truncate",
                py::arg("A_truncate") = 0,
                py::arg("S_truncate") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 10000000);

            cls.def(
                "minimise_truncate",
                static_cast<size_t (C::*)(
                    const xt::pytensor<size_t, 1>&, size_t, size_t, double, size_t, size_t)>(
                    &C::template minimise_truncate<xt::pytensor<size_t, 1>>),
                "minimise_truncate",
                py::arg("idx_n"),
                py::arg("A_truncate") = 0,
                py::arg("S_truncate") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = 10000000);

            cls.def(
                "affineSimpleShear",
                &C::affineSimpleShear,
                "affineSimpleShear",
                py::arg("delta_gamma"));

            cls.def(
                "affineSimpleShearCentered",
                &C::affineSimpleShearCentered,
                "affineSimpleShearCentered",
                py::arg("delta_gamma"));

            cls.def("__repr__", [](const C&) {
                return "<FrictionQPotFEM.Generic2d.System>";
            });
        }

PYBIND11_MODULE(_FrictionQPotFEM, mod)
{
    // Ensure members to display as `FrictionQPotFEM.X` (not `FrictionQPotFEM._FrictionQPotFEM.X`)
    ScopedModuleNameOverride name_override(mod, "FrictionQPotFEM");

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

        py::class_<SUB::System> cls(sub, "System");

        register_SystemBase<SUB::System>(cls);

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

            py::class_<SUB::System> cls(sub, "System");

            register_SystemBase<SUB::System>(cls);

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

            cls.def("typical_plastic_h", &SUB::System::typical_plastic_h, "typical_plastic_h");
            cls.def("typical_plastic_dV", &SUB::System::typical_plastic_dV, "typical_plastic_dV");

            cls.def(
                "initEventDrivenSimpleShear",
                &SUB::System::initEventDrivenSimpleShear,
                "initEventDrivenSimpleShear");

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
            cls.def_property_readonly("p", &SUB::LocalTriggerFineLayerFull::p);
            cls.def_property_readonly("s", &SUB::LocalTriggerFineLayerFull::s);
            cls.def("delta_u", &SUB::LocalTriggerFineLayerFull::delta_u);
            cls.def("u_s", &SUB::LocalTriggerFineLayerFull::u_s);
            cls.def("u_p", &SUB::LocalTriggerFineLayerFull::u_p);
            cls.def("Eps_s", &SUB::LocalTriggerFineLayerFull::Eps_s);
            cls.def("Eps_p", &SUB::LocalTriggerFineLayerFull::Eps_p);
            cls.def("Sig_s", &SUB::LocalTriggerFineLayerFull::Sig_s);
            cls.def("Sig_p", &SUB::LocalTriggerFineLayerFull::Sig_p);
            cls.def_property_readonly("dgamma", &SUB::LocalTriggerFineLayerFull::dgamma);
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

    // // --------------------------------------------------
    // // FrictionQPotFEM.UniformMultiLayerIndividualDrive2d
    // // --------------------------------------------------

    // {

    //     py::module sub = mod.def_submodule(
    //         "UniformMultiLayerIndividualDrive2d", "UniformMultiLayerIndividualDrive2d");

    //     namespace SUB = FrictionQPotFEM::UniformMultiLayerIndividualDrive2d;

    //     sub.def(
    //         "version_dependencies",
    //         &SUB::version_dependencies,
    //         "Return version information of library and its dependencies.");

    //     py::class_<SUB::System, FrictionQPotFEM::Generic2d::System> cls(sub, "System");

    //     cls.def(
    //         py::init<
    //             const xt::pytensor<double, 2>&,
    //             const xt::pytensor<size_t, 2>&,
    //             const xt::pytensor<size_t, 2>&,
    //             const xt::pytensor<size_t, 1>&,
    //             const std::vector<xt::pytensor<size_t, 1>>&,
    //             const std::vector<xt::pytensor<size_t, 1>>&,
    //             const xt::pytensor<bool, 1>&>(),
    //         "System",
    //         py::arg("coor"),
    //         py::arg("conn"),
    //         py::arg("dofs"),
    //         py::arg("iip"),
    //         py::arg("elem"),
    //         py::arg("node"),
    //         py::arg("layer_is_plastic"));

    //     cls.def("nlayer", &SUB::System::nlayer, "nlayer");

    //     cls.def("layerNodes", &SUB::System::layerNodes, "layerNodes", py::arg("i"));

    //     cls.def("layerElements", &SUB::System::layerElements, "layerElements", py::arg("i"));

    //     cls.def_property_readonly("layerIsPlastic", &SUB::System::layerIsPlastic, "layerIsPlastic");

    //     cls.def(
    //         "layerSetDriveStiffness",
    //         &SUB::System::layerSetDriveStiffness,
    //         "layerSetDriveStiffness",
    //         py::arg("k"),
    //         py::arg("symmetric") = true);

    //     cls.def(
    //         "initEventDriven",
    //         py::overload_cast<const xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&>(
    //             &SUB::System::initEventDriven<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>),
    //         "initEventDriven",
    //         py::arg("delta_ubar"),
    //         py::arg("active"));

    //     cls.def(
    //         "initEventDriven",
    //         py::overload_cast<
    //             const xt::pytensor<double, 2>&,
    //             const xt::pytensor<bool, 2>&,
    //             const xt::pytensor<double, 2>&>(&SUB::System::initEventDriven<
    //                                             xt::pytensor<double, 2>,
    //                                             xt::pytensor<bool, 2>,
    //                                             xt::pytensor<double, 2>>),
    //         "initEventDriven",
    //         py::arg("delta_ubar"),
    //         py::arg("active"),
    //         py::arg("delta_u"));

    //     cls.def_property_readonly(
    //         "eventDriven_deltaUbar", &SUB::System::eventDriven_deltaUbar, "eventDriven_deltaUbar");

    //     cls.def_property_readonly(
    //         "eventDriven_targetActive",
    //         &SUB::System::eventDriven_targetActive,
    //         "eventDriven_targetActive");

    //     cls.def(
    //         "layerSetTargetActive",
    //         &SUB::System::layerSetTargetActive<xt::pytensor<double, 2>>,
    //         "layerSetTargetActive",
    //         py::arg("active"));

    //     cls.def("layerUbar", &SUB::System::layerUbar, "layerUbar");
    //     cls.def_property_readonly(
    //         "layerTargetUbar", &SUB::System::layerTargetUbar, "layerTargetUbar");
    //     cls.def_property_readonly(
    //         "layerTargetActive", &SUB::System::layerTargetActive, "layerTargetActive");

    //     cls.def(
    //         "layerSetTargetUbar",
    //         &SUB::System::layerSetTargetUbar<xt::pytensor<double, 2>>,
    //         "layerSetTargetUbar",
    //         py::arg("ubar"));

    //     cls.def(
    //         "layerSetUbar",
    //         &SUB::System::layerSetUbar<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
    //         "layerSetUbar",
    //         py::arg("ubar"),
    //         py::arg("prescribe"));

    //     cls.def(
    //         "layerTargetUbar_affineSimpleShear",
    //         &SUB::System::layerTargetUbar_affineSimpleShear<xt::pytensor<double, 1>>,
    //         "layerTargetUbar_affineSimpleShear",
    //         py::arg("delta_gamma"),
    //         py::arg("height"));

    //     cls.def_property_readonly("fdrive", &SUB::System::fdrive, "fdrive");

    //     cls.def("layerFdrive", &SUB::System::layerFdrive, "layerFdrive");

    //     cls.def("__repr__", [](const SUB::System&) {
    //         return "<FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System>";
    //     });
    // }

    // // ---------------------------------------------
    // // FrictionQPotFEM.UniformMultiLayerLeverDrive2d
    // // ---------------------------------------------

    // {

    //     py::module sub =
    //         mod.def_submodule("UniformMultiLayerLeverDrive2d", "UniformMultiLayerLeverDrive2d");

    //     namespace SUB = FrictionQPotFEM::UniformMultiLayerLeverDrive2d;

    //     sub.def(
    //         "version_dependencies",
    //         &SUB::version_dependencies,
    //         "Return version information of library and its dependencies.");

    //     py::class_<SUB::System, FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System> cls(
    //         sub, "System");

    //     cls.def(
    //         py::init<
    //             const xt::pytensor<double, 2>&,
    //             const xt::pytensor<size_t, 2>&,
    //             const xt::pytensor<size_t, 2>&,
    //             const xt::pytensor<size_t, 1>&,
    //             const std::vector<xt::pytensor<size_t, 1>>&,
    //             const std::vector<xt::pytensor<size_t, 1>>&,
    //             const xt::pytensor<bool, 1>&>(),
    //         "System",
    //         py::arg("coor"),
    //         py::arg("conn"),
    //         py::arg("dofs"),
    //         py::arg("iip"),
    //         py::arg("elem"),
    //         py::arg("node"),
    //         py::arg("layer_is_plastic"));

    //     cls.def(
    //         "setLeverProperties",
    //         &SUB::System::setLeverProperties<xt::pytensor<double, 1>>,
    //         "setLeverProperties",
    //         py::arg("H"),
    //         py::arg("hi"));

    //     cls.def("setLeverTarget", &SUB::System::setLeverTarget, "setLeverTarget", py::arg("u"));
    //     cls.def("leverTarget", &SUB::System::leverTarget, "leverTarget");
    //     cls.def("leverPosition", &SUB::System::leverPosition, "leverPosition");

    //     cls.def(
    //         "initEventDriven",
    //         py::overload_cast<double, const xt::pytensor<bool, 2>&>(
    //             &SUB::System::initEventDriven<xt::pytensor<bool, 2>>),
    //         "initEventDriven",
    //         py::arg("delta_ubar"),
    //         py::arg("active"));

    //     cls.def(
    //         "initEventDriven",
    //         py::overload_cast<
    //             double,
    //             const xt::pytensor<bool, 2>&,
    //             const xt::pytensor<double, 2>&,
    //             const xt::pytensor<double, 2>&>(&SUB::System::initEventDriven<
    //                                             xt::pytensor<bool, 2>,
    //                                             xt::pytensor<double, 2>,
    //                                             xt::pytensor<double, 2>>),
    //         "initEventDriven",
    //         py::arg("xdrive"),
    //         py::arg("active"),
    //         py::arg("delta_u"),
    //         py::arg("delta_ubar"));

    //     cls.def(
    //         "eventDriven_deltaLeverPosition",
    //         &SUB::System::eventDriven_deltaLeverPosition,
    //         "eventDriven_deltaLeverPosition");

    //     cls.def("__repr__", [](const SUB::System&) {
    //         return "<FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System>";
    //     });
    // }
}
