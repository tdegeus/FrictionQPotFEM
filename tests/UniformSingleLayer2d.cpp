#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <catch2/catch_all.hpp>
#include <iostream>
#include <xtensor/xrandom.hpp>

TEST_CASE("FrictionQPotFEM::UniformSingleLayer2d", "UniformSingleLayer2d.h")
{
    SECTION("version")
    {
        std::cout << FrictionQPotFEM::version() << std::endl;

        auto deps = FrictionQPotFEM::UniformSingleLayer2d::version_dependencies();

        for (auto& i : deps) {
            std::cout << i << std::endl;
        }
    }

    SECTION("System::addAffineSimpleShear")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

        xt::xtensor<size_t, 1> elas = {0, 1, 2, 6, 7, 8};
        xt::xtensor<size_t, 1> plas = {3, 4, 5};
        xt::xtensor<double, 2> epsy = {
            {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}};

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::eval(xt::arange<size_t>(mesh.nnode() * mesh.ndim())),
            elas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            plas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::epsy_initelastic_toquad(epsy),
            1,
            1,
            1,
            0);

        for (size_t i = 0; i < 10; ++i) {
            double delta_gamma = 0.01;
            double gamma = delta_gamma * static_cast<double>(i + 1);
            auto du = sys.affineSimpleShear(delta_gamma);
            sys.setU(xt::eval(sys.u() + du));
            REQUIRE(xt::allclose(xt::view(sys.Eps(), xt::all(), xt::all(), 0, 1), gamma));
        }
    }

    SECTION("System::addAffineSimpleShearCentered")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

        xt::xtensor<size_t, 1> elas = {0, 1, 2, 6, 7, 8};
        xt::xtensor<size_t, 1> plas = {3, 4, 5};
        xt::xtensor<double, 2> epsy = {
            {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}};

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::eval(xt::arange<size_t>(mesh.nnode() * mesh.ndim())),
            elas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            plas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::epsy_initelastic_toquad(epsy),
            1,
            1,
            1,
            0);

        auto plastic = sys.plastic();
        auto conn = sys.conn();
        auto bot = xt::view(conn, xt::keep(plastic), 0); // missing last node, but ok for test
        auto top = xt::view(conn, xt::keep(plastic), 3);

        for (size_t i = 0; i < 10; ++i) {
            double delta_gamma = 0.01;
            double gamma = delta_gamma * static_cast<double>(i + 1);
            auto dus = sys.affineSimpleShearCentered(delta_gamma);
            sys.setU(xt::eval(sys.u() + dus));
            auto u = sys.u();
            auto du = xt::eval(xt::view(u, xt::keep(top), 1) + xt::view(u, xt::keep(bot), 1));
            REQUIRE(xt::allclose(xt::view(sys.Eps(), xt::all(), xt::all(), 0, 1), gamma));
            REQUIRE(xt::allclose(du, 0.0));
        }
    }

    SECTION("System::addSimpleShearEventDriven")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

        xt::xtensor<size_t, 1> elas = {};
        xt::xtensor<size_t, 1> plas = {0};
        xt::xtensor<double, 2> epsy = {{1.0, 2.0, 3.0, 4.0}};

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::eval(xt::arange<size_t>(mesh.nnode() * mesh.ndim())),
            elas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            plas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::epsy_initelastic_toquad(epsy),
            1,
            1,
            1,
            0);

        REQUIRE(sys.isHomogeneousElastic());

        double delta_eps = 1e-3;
        auto dV = sys.quad().AsTensor<2>(sys.dV());
        xt::xtensor<double, 2> Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 0.0));

        sys.initEventDrivenSimpleShear();

        sys.eventDrivenStep(delta_eps, false, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 - delta_eps / 2.0));

        sys.eventDrivenStep(delta_eps, true, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 + delta_eps / 2.0));

        sys.eventDrivenStep(delta_eps, false, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 - delta_eps / 2.0));

        sys.eventDrivenStep(delta_eps, true, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 + delta_eps / 2.0));

        sys.eventDrivenStep(delta_eps, false, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 + delta_eps / 2.0));

        sys.eventDrivenStep(delta_eps, true, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 - delta_eps / 2.0));

        sys.eventDrivenStep(delta_eps, false, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 + delta_eps / 2.0));

        sys.eventDrivenStep(delta_eps, true, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(
            GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 - delta_eps / 2.0));
    }

    SECTION("System::addSimpleShearToFixedStress")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

        xt::xtensor<size_t, 1> elas = {};
        xt::xtensor<size_t, 1> plas = {0};
        xt::xtensor<double, 2> epsy = {{1.0, 2.0, 3.0, 4.0}};

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::eval(xt::arange<size_t>(mesh.nnode() * mesh.ndim())),
            elas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            plas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::epsy_initelastic_toquad(epsy),
            1,
            1,
            1,
            0);

        REQUIRE(sys.isHomogeneousElastic());

        auto dV = sys.quad().AsTensor<2>(sys.dV());
        xt::xtensor<double, 2> Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        xt::xtensor<double, 2> Sigbar = xt::average(sys.Sig(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 0.0));
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)(), 0.0));

        double target_stress = 1.0;
        sys.addSimpleShearToFixedStress(target_stress);
        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        Sigbar = xt::average(sys.Sig(), dV, {0, 1});
        REQUIRE(
            xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), target_stress / 2.0));
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)(), target_stress));

        target_stress = 1.5;
        sys.addSimpleShearToFixedStress(target_stress);
        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        Sigbar = xt::average(sys.Sig(), dV, {0, 1});
        REQUIRE(
            xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), target_stress / 2.0));
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)(), target_stress));

        target_stress = 0.5;
        sys.addSimpleShearToFixedStress(target_stress);
        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        Sigbar = xt::average(sys.Sig(), dV, {0, 1});
        REQUIRE(
            xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), target_stress / 2.0));
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)(), target_stress));
    }

    SECTION("System::triggerElementWithLocalSimpleShear")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

        auto dofs = mesh.dofs();
        auto top = mesh.nodesTopEdge();
        auto bottom = mesh.nodesBottomEdge();
        size_t nfix = top.size();
        xt::xtensor<size_t, 1> iip = xt::empty<size_t>({2 * mesh.ndim() * nfix});
        xt::view(iip, xt::range(0 * nfix, 1 * nfix)) = xt::view(dofs, xt::keep(bottom), 0);
        xt::view(iip, xt::range(1 * nfix, 2 * nfix)) = xt::view(dofs, xt::keep(bottom), 1);
        xt::view(iip, xt::range(2 * nfix, 3 * nfix)) = xt::view(dofs, xt::keep(top), 0);
        xt::view(iip, xt::range(3 * nfix, 4 * nfix)) = xt::view(dofs, xt::keep(top), 1);

        xt::xtensor<size_t, 1> elas = {0, 1, 2, 3, 5, 6, 7, 8};
        xt::xtensor<size_t, 1> plas = {4};
        xt::xtensor<double, 2> epsy = {{1.0, 2.0, 3.0, 4.0}};

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            dofs,
            iip,
            elas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({elas.size()}))),
            plas,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({plas.size()}))),
            FrictionQPotFEM::epsy_initelastic_toquad(epsy),
            1,
            1,
            1,
            0);

        REQUIRE(sys.isHomogeneousElastic());

        double delta_eps = 1e-3;
        sys.triggerElementWithLocalSimpleShear(delta_eps, 0);

        auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(sys.Eps());
        auto plastic = sys.plastic();
        auto eps_p = xt::view(eps, xt::keep(plastic), xt::all());

        REQUIRE(xt::allclose(eps_p, 1.0 + delta_eps / 2.0));
    }
}
