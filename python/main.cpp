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
#include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>
#include <FrictionQPotFEM/UniformMultiLayerLeverDrive2d.h>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <FrictionQPotFEM/UniformSingleLayerThermal2d.h>

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

PYBIND11_MODULE(_FrictionQPotFEM, mod)
{
    // Ensure members to display as `FrictionQPotFEM.X` (not `FrictionQPotFEM._FrictionQPotFEM.X`)
    ScopedModuleNameOverride name_override(mod, "FrictionQPotFEM");

    xt::import_numpy();

    mod.doc() = "Friction model based on GooseFEM and FrictionQPotFEM";

    namespace M = FrictionQPotFEM;

    mod.def("version", &M::version, "Return version string.");

    mod.def(
        "epsy_initelastic_toquad",
        &M::epsy_initelastic_toquad,
        "Convert array of yield strains per element: init elastic, same for all quadrature points",
        py::arg("arg"),
        py::arg("nip") = GooseFEM::Element::Quad4::Gauss::nip());

    mod.def(
        "moduli_toquad",
        &M::moduli_toquad,
        "Convert array of moduli per element to be the same for all quadrature points",
        py::arg("arg"),
        py::arg("nip") = GooseFEM::Element::Quad4::Gauss::nip());

    mod.def(
        "getuniform",
        &M::getuniform<xt::pyarray<double>>,
        "Extract uniform value (throw if not uniform)",
        py::arg("arg"));

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

        sub.def(
            "version_compiler", &SUB::version_compiler, "Version information of the compilers.");

        {

            py::class_<SUB::System> cls(sub, "System");

            cls.def(
                py::init<
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 3>&,
                    double,
                    double,
                    double,
                    double>(),
                "System",
                py::arg("coor"),
                py::arg("conn"),
                py::arg("dofs"),
                py::arg("iip"),
                py::arg("elastic_elem"),
                py::arg("elastic_K"),
                py::arg("elastic_G"),
                py::arg("plastic_elem"),
                py::arg("plastic_K"),
                py::arg("plastic_G"),
                py::arg("plastic_epsy"),
                py::arg("dt"),
                py::arg("rho"),
                py::arg("alpha"),
                py::arg("eta"));

            cls.def_property_readonly("N", &SUB::System::N, "Number of plastic elements");
            cls.def_property_readonly("type", &SUB::System::type, "Class identifier");

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

            cls.def_property_readonly(
                "isHomogeneousElastic",
                &SUB::System::isHomogeneousElastic,
                "``True`` is elasticity is the same everywhere");

            cls.def_property(
                "eta",
                &SUB::System::eta,
                &SUB::System::setEta,
                "Damping proportional to the strain-rate at the interface");

            cls.def_property(
                "alpha",
                &SUB::System::alpha,
                &SUB::System::setAlpha,
                "Homogeneous background damping density. Otherwise use setDampingMatrix.");

            cls.def_property(
                "rho",
                &SUB::System::rho,
                &SUB::System::setRho,
                "Homogeneous mass density. Otherwise use setMassMatrix.");

            cls.def_property("dt", &SUB::System::dt, &SUB::System::setDt, "Time step");
            cls.def_property("t", &SUB::System::t, &SUB::System::setT, "Current time == inc * dt");
            cls.def_property("inc", &SUB::System::inc, &SUB::System::setInc, "Current increment");

            cls.def("quench", &SUB::System::quench, "quench");

            cls.def_property_readonly(
                "elastic_elem", &SUB::System::elastic_elem, "Element numbers of elastic elements");

            cls.def_property_readonly(
                "plastic_elem", &SUB::System::plastic_elem, "Element numbers of plastic elements");

            cls.def_property_readonly("conn", &SUB::System::conn, "conn");
            cls.def_property_readonly("coor", &SUB::System::coor, "coor");
            cls.def_property_readonly("dofs", &SUB::System::dofs, "dofs");

            cls.def("updated_u", &SUB::System::updated_u, "Update relevant fields after u update");
            cls.def("updated_v", &SUB::System::updated_v, "Update relevant fields after v update");
            cls.def("refresh", &SUB::System::refresh, "Recompute all forces");

            cls.def_property(
                "u", &SUB::System::u, &SUB::System::setU<xt::pytensor<double, 2>>, "u");

            cls.def_property(
                "v", &SUB::System::v, &SUB::System::setV<xt::pytensor<double, 2>>, "v");

            cls.def_property(
                "a", &SUB::System::a, &SUB::System::setA<xt::pytensor<double, 2>>, "a");

            cls.def_property_readonly("mass", &SUB::System::mass, "mass");

            cls.def_property_readonly("damping", &SUB::System::damping, "damping");

            cls.def_property(
                "fext", &SUB::System::fext, &SUB::System::setFext<xt::pytensor<double, 2>>, "fext");

            cls.def_property_readonly("fint", &SUB::System::fint, "fint");
            cls.def_property_readonly("fmaterial", &SUB::System::fmaterial, "fmaterial");
            cls.def_property_readonly("fdamp", &SUB::System::fdamp, "fdamp");
            cls.def_property_readonly("dV", &SUB::System::dV, "dV");
            cls.def_property_readonly("vector", &SUB::System::vector, "vector");
            cls.def_property_readonly("quad", &SUB::System::quad, "quad");
            cls.def_property_readonly("elastic", &SUB::System::elastic, "elastic");
            cls.def_property_readonly("plastic", &SUB::System::plastic, "plastic");

            cls.def("residual", &SUB::System::residual, "residual");
            cls.def("stiffness", &SUB::System::stiffness, "stiffness");
            cls.def("K", &SUB::System::K, "K");
            cls.def("G", &SUB::System::G, "G");
            cls.def("Sig", &SUB::System::Sig, "Sig");
            cls.def("Eps", &SUB::System::Eps, "Eps");
            cls.def("Epsdot", &SUB::System::Epsdot, "Epsdot");
            cls.def("Epsddot", &SUB::System::Epsddot, "Epsddot");
            cls.def("plastic_Epsdot", &SUB::System::plastic_Epsdot, "plastic_Epsdot");

            cls.def(
                "eventDriven_setDeltaU",
                &SUB::System::eventDriven_setDeltaU<xt::pytensor<double, 2>>,
                "eventDriven_setDeltaU",
                py::arg("delta_u"),
                py::arg("autoscale") = true);

            cls.def_property_readonly(
                "eventDriven_deltaU", &SUB::System::eventDriven_deltaU, "eventDriven_deltaU");

            cls.def(
                "eventDrivenStep",
                &SUB::System::eventDrivenStep,
                "eventDrivenStep",
                py::arg("deps"),
                py::arg("kick"),
                py::arg("direction") = +1,
                py::arg("yield_element") = false,
                py::arg("iterative") = false);

            cls.def("timeStep", &SUB::System::timeStep, "timeStep");

            cls.def(
                "timeSteps",
                &SUB::System::timeSteps,
                "timeSteps",
                py::arg("n"),
                py::arg("nmargin") = 0);

            cls.def(
                "flowSteps",
                &SUB::System::flowSteps<xt::pytensor<double, 2>>,
                "flowSteps",
                py::arg("n"),
                py::arg("v"),
                py::arg("nmargin") = 0);

            cls.def(
                "timeStepsUntilEvent",
                &SUB::System::timeStepsUntilEvent,
                "timeStepsUntilEvent",
                py::arg("nmargin") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = size_t(1e7));

            cls.def(
                "minimise",
                &SUB::System::minimise,
                "minimise",
                py::arg("nmargin") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = size_t(1e7),
                py::arg("time_activity") = false,
                py::arg("max_iter_is_error") = true);

            cls.def(
                "minimise_truncate",
                static_cast<size_t (SUB::System::*)(size_t, size_t, double, size_t, size_t)>(
                    &SUB::System::minimise_truncate),
                "minimise_truncate",
                py::arg("A_truncate") = 0,
                py::arg("S_truncate") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = size_t(1e7));

            cls.def(
                "minimise_truncate",
                static_cast<size_t (SUB::System::*)(
                    const xt::pytensor<size_t, 1>&, size_t, size_t, double, size_t, size_t)>(
                    &SUB::System::minimise_truncate<xt::pytensor<size_t, 1>>),
                "minimise_truncate",
                py::arg("idx_n"),
                py::arg("A_truncate") = 0,
                py::arg("S_truncate") = 0,
                py::arg("tol") = 1e-5,
                py::arg("niter_tol") = 20,
                py::arg("max_iter") = size_t(1e7));

            cls.def(
                "affineSimpleShear",
                &SUB::System::affineSimpleShear,
                "affineSimpleShear",
                py::arg("delta_gamma"));

            cls.def(
                "affineSimpleShearCentered",
                &SUB::System::affineSimpleShearCentered,
                "affineSimpleShearCentered",
                py::arg("delta_gamma"));

            cls.def("__repr__", [](const SUB::System&) {
                return "<FrictionQPotFEM.Generic2d.System>";
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

        sub.def(
            "version_compiler", &SUB::version_compiler, "Version information of the compilers.");

        {

            py::class_<SUB::System, FrictionQPotFEM::Generic2d::System> cls(sub, "System");

            cls.def(
                py::init<
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 3>&,
                    double,
                    double,
                    double,
                    double>(),
                "System",
                py::arg("coor"),
                py::arg("conn"),
                py::arg("dofs"),
                py::arg("iip"),
                py::arg("elastic_elem"),
                py::arg("elastic_K"),
                py::arg("elastic_G"),
                py::arg("plastic_elem"),
                py::arg("plastic_K"),
                py::arg("plastic_G"),
                py::arg("plastic_epsy"),
                py::arg("dt"),
                py::arg("rho"),
                py::arg("alpha"),
                py::arg("eta"));

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

    // -------------------------------------------
    // FrictionQPotFEM.UniformSingleLayerThermal2d
    // -------------------------------------------

    {

        py::module sub =
            mod.def_submodule("UniformSingleLayerThermal2d", "UniformSingleLayerThermal2d");

        namespace SUB = FrictionQPotFEM::UniformSingleLayerThermal2d;

        sub.def(
            "version_dependencies",
            &SUB::version_dependencies,
            "Return version information of library and its dependencies.");

        sub.def(
            "version_compiler", &SUB::version_compiler, "Version information of the compilers.");

        {

            py::class_<SUB::System, FrictionQPotFEM::Generic2d::System> cls(sub, "System");

            cls.def(
                py::init<
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<size_t, 1>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 2>&,
                    const xt::pytensor<double, 3>&,
                    double,
                    double,
                    double,
                    double,
                    size_t,
                    size_t,
                    double>(),
                "System",
                py::arg("coor"),
                py::arg("conn"),
                py::arg("dofs"),
                py::arg("iip"),
                py::arg("elastic_elem"),
                py::arg("elastic_K"),
                py::arg("elastic_G"),
                py::arg("plastic_elem"),
                py::arg("plastic_K"),
                py::arg("plastic_G"),
                py::arg("plastic_epsy"),
                py::arg("dt"),
                py::arg("rho"),
                py::arg("alpha"),
                py::arg("eta"),
                py::arg("temperature_dinc"),
                py::arg("temperature_seed"),
                py::arg("temperature"));

            cls.def("__repr__", [](const SUB::System&) {
                return "<FrictionQPotFEM.UniformSingleLayerThermal2d.System>";
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

        sub.def(
            "version_compiler", &SUB::version_compiler, "Version information of the compilers.");

        py::class_<SUB::System, FrictionQPotFEM::Generic2d::System> cls(sub, "System");

        cls.def(
            py::init<
                const xt::pytensor<double, 2>&,
                const xt::pytensor<size_t, 2>&,
                const xt::pytensor<size_t, 2>&,
                const xt::pytensor<size_t, 1>&,
                const std::vector<xt::pytensor<size_t, 1>>&,
                const std::vector<xt::pytensor<size_t, 1>>&,
                const xt::pytensor<bool, 1>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 3>&,
                double,
                double,
                double,
                double,
                const xt::pytensor<bool, 2>&,
                double>(),
            "System",
            py::arg("coor"),
            py::arg("conn"),
            py::arg("dofs"),
            py::arg("iip"),
            py::arg("elem"),
            py::arg("node"),
            py::arg("layer_is_plastic"),
            py::arg("elastic_K"),
            py::arg("elastic_G"),
            py::arg("plastic_K"),
            py::arg("plastic_G"),
            py::arg("plastic_epsy"),
            py::arg("dt"),
            py::arg("rho"),
            py::arg("alpha"),
            py::arg("eta"),
            py::arg("drive_is_active"),
            py::arg("k_drive"));

        cls.def("nlayer", &SUB::System::nlayer, "nlayer");

        cls.def("layerNodes", &SUB::System::layerNodes, "layerNodes", py::arg("i"));

        cls.def("layerElements", &SUB::System::layerElements, "layerElements", py::arg("i"));

        cls.def_property_readonly("layerIsPlastic", &SUB::System::layerIsPlastic, "layerIsPlastic");

        cls.def(
            "layerSetDriveStiffness",
            &SUB::System::layerSetDriveStiffness,
            "layerSetDriveStiffness",
            py::arg("k"),
            py::arg("symmetric") = true);

        cls.def(
            "initEventDriven",
            py::overload_cast<const xt::pytensor<double, 2>&, const xt::pytensor<bool, 2>&>(
                &SUB::System::initEventDriven<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>),
            "initEventDriven",
            py::arg("delta_ubar"),
            py::arg("active"));

        cls.def(
            "initEventDriven",
            py::overload_cast<
                const xt::pytensor<double, 2>&,
                const xt::pytensor<bool, 2>&,
                const xt::pytensor<double, 2>&>(&SUB::System::initEventDriven<
                                                xt::pytensor<double, 2>,
                                                xt::pytensor<bool, 2>,
                                                xt::pytensor<double, 2>>),
            "initEventDriven",
            py::arg("delta_ubar"),
            py::arg("active"),
            py::arg("delta_u"));

        cls.def_property_readonly(
            "eventDriven_deltaUbar", &SUB::System::eventDriven_deltaUbar, "eventDriven_deltaUbar");

        cls.def_property_readonly(
            "eventDriven_targetActive",
            &SUB::System::eventDriven_targetActive,
            "eventDriven_targetActive");

        cls.def(
            "layerSetTargetActive",
            &SUB::System::layerSetTargetActive<xt::pytensor<double, 2>>,
            "layerSetTargetActive",
            py::arg("active"));

        cls.def("layerUbar", &SUB::System::layerUbar, "layerUbar");
        cls.def_property_readonly(
            "layerTargetUbar", &SUB::System::layerTargetUbar, "layerTargetUbar");
        cls.def_property_readonly(
            "layerTargetActive", &SUB::System::layerTargetActive, "layerTargetActive");

        cls.def(
            "layerSetTargetUbar",
            &SUB::System::layerSetTargetUbar<xt::pytensor<double, 2>>,
            "layerSetTargetUbar",
            py::arg("ubar"));

        cls.def(
            "layerSetUbar",
            &SUB::System::layerSetUbar<xt::pytensor<double, 2>, xt::pytensor<bool, 2>>,
            "layerSetUbar",
            py::arg("ubar"),
            py::arg("prescribe"));

        cls.def(
            "layerTargetUbar_affineSimpleShear",
            &SUB::System::layerTargetUbar_affineSimpleShear<xt::pytensor<double, 1>>,
            "layerTargetUbar_affineSimpleShear",
            py::arg("delta_gamma"),
            py::arg("height"));

        cls.def_property_readonly("fdrive", &SUB::System::fdrive, "fdrive");

        cls.def("layerFdrive", &SUB::System::layerFdrive, "layerFdrive");

        cls.def("__repr__", [](const SUB::System&) {
            return "<FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System>";
        });
    }

    // ---------------------------------------------
    // FrictionQPotFEM.UniformMultiLayerLeverDrive2d
    // ---------------------------------------------

    {

        py::module sub =
            mod.def_submodule("UniformMultiLayerLeverDrive2d", "UniformMultiLayerLeverDrive2d");

        namespace SUB = FrictionQPotFEM::UniformMultiLayerLeverDrive2d;

        sub.def(
            "version_dependencies",
            &SUB::version_dependencies,
            "Return version information of library and its dependencies.");

        sub.def(
            "version_compiler", &SUB::version_compiler, "Version information of the compilers.");

        py::class_<SUB::System, FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System> cls(
            sub, "System");

        cls.def(
            py::init<
                const xt::pytensor<double, 2>&,
                const xt::pytensor<size_t, 2>&,
                const xt::pytensor<size_t, 2>&,
                const xt::pytensor<size_t, 1>&,
                const std::vector<xt::pytensor<size_t, 1>>&,
                const std::vector<xt::pytensor<size_t, 1>>&,
                const xt::pytensor<bool, 1>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 3>&,
                double,
                double,
                double,
                double,
                const xt::pytensor<bool, 2>&,
                double,
                double,
                const xt::pytensor<double, 1>&>(),
            "System",
            py::arg("coor"),
            py::arg("conn"),
            py::arg("dofs"),
            py::arg("iip"),
            py::arg("elem"),
            py::arg("node"),
            py::arg("layer_is_plastic"),
            py::arg("elastic_K"),
            py::arg("elastic_G"),
            py::arg("plastic_K"),
            py::arg("plastic_G"),
            py::arg("plastic_epsy"),
            py::arg("dt"),
            py::arg("rho"),
            py::arg("alpha"),
            py::arg("eta"),
            py::arg("drive_is_active"),
            py::arg("k_drive"),
            py::arg("H"),
            py::arg("hi"));

        cls.def(
            "setLeverProperties",
            &SUB::System::setLeverProperties<xt::pytensor<double, 1>>,
            "setLeverProperties",
            py::arg("H"),
            py::arg("hi"));

        cls.def("setLeverTarget", &SUB::System::setLeverTarget, "setLeverTarget", py::arg("u"));
        cls.def("leverTarget", &SUB::System::leverTarget, "leverTarget");
        cls.def("leverPosition", &SUB::System::leverPosition, "leverPosition");

        cls.def(
            "initEventDriven",
            py::overload_cast<double, const xt::pytensor<bool, 2>&>(
                &SUB::System::initEventDriven<xt::pytensor<bool, 2>>),
            "initEventDriven",
            py::arg("delta_ubar"),
            py::arg("active"));

        cls.def(
            "initEventDriven",
            py::overload_cast<
                double,
                const xt::pytensor<bool, 2>&,
                const xt::pytensor<double, 2>&,
                const xt::pytensor<double, 2>&>(&SUB::System::initEventDriven<
                                                xt::pytensor<bool, 2>,
                                                xt::pytensor<double, 2>,
                                                xt::pytensor<double, 2>>),
            "initEventDriven",
            py::arg("xdrive"),
            py::arg("active"),
            py::arg("delta_u"),
            py::arg("delta_ubar"));

        cls.def(
            "eventDriven_deltaLeverPosition",
            &SUB::System::eventDriven_deltaLeverPosition,
            "eventDriven_deltaLeverPosition");

        cls.def("__repr__", [](const SUB::System&) {
            return "<FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System>";
        });
    }
}
