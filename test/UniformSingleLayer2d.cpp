
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>

TEST_CASE("FrictionQPotFEM::UniformSingleLayer2d", "UniformSingleLayer2d.h")
{
    SECTION("System::plastic_signOfPerturbation")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
            xt::xtensor<size_t, 1>{0, 1, 2, 6, 7, 8},
            xt::xtensor<size_t, 1>{3, 4, 5});

        sys.setMassMatrix(xt::ones<double>({mesh.nelem()}));
        sys.setDampingMatrix(xt::ones<double>({mesh.nelem()}));

        sys.setElastic(
            xt::ones<double>({6}),
            xt::ones<double>({6}));

        sys.setPlastic(
            xt::xtensor<double, 1>{1.0, 1.0, 1.0},
            xt::xtensor<double, 1>{1.0, 1.0, 1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}});

        sys.setDt(1.0);

        sys.addAffineSimpleShear(0.1);
        auto u0 = sys.u();
        sys.addAffineSimpleShear(0.1);
        auto u = sys.u();
        auto delta_u = u - u0;

        REQUIRE(xt::all(xt::equal(sys.plastic_signOfPerturbation(delta_u), xt::ones<int>({3, 4}))));
        REQUIRE(xt::all(xt::equal(sys.plastic_signOfPerturbation(- delta_u), -1 * xt::ones<int>({3, 4}))));
    }

    SECTION("System::addAffineSimpleShear")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
            xt::xtensor<size_t, 1>{0, 1, 2, 6, 7, 8},
            xt::xtensor<size_t, 1>{3, 4, 5});

        sys.setMassMatrix(xt::ones<double>({mesh.nelem()}));
        sys.setDampingMatrix(xt::ones<double>({mesh.nelem()}));

        sys.setElastic(
            xt::ones<double>({6}),
            xt::ones<double>({6}));

        sys.setPlastic(
            xt::xtensor<double, 1>{1.0, 1.0, 1.0},
            xt::xtensor<double, 1>{1.0, 1.0, 1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}});

        sys.setDt(1.0);

        for (size_t i = 0; i < 10; ++i) {
            double delta_gamma = 0.01;
            double gamma = delta_gamma * static_cast<double>(i + 1);
            sys.addAffineSimpleShear(delta_gamma);
            REQUIRE(xt::allclose(xt::view(sys.Eps(), xt::all(), xt::all(), 0, 1), gamma));
        }
    }

    SECTION("System::addAffineSimpleShearCentered")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
            xt::xtensor<size_t, 1>{0, 1, 2, 6, 7, 8},
            xt::xtensor<size_t, 1>{3, 4, 5});

        sys.setMassMatrix(xt::ones<double>({mesh.nelem()}));
        sys.setDampingMatrix(xt::ones<double>({mesh.nelem()}));

        sys.setElastic(
            xt::ones<double>({6}),
            xt::ones<double>({6}));

        sys.setPlastic(
            xt::xtensor<double, 1>{1.0, 1.0, 1.0},
            xt::xtensor<double, 1>{1.0, 1.0, 1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}, {1.0, 2.0, 3.0, 4.0}});

        sys.setDt(1.0);

        auto plastic = sys.plastic();
        auto conn = sys.conn();
        auto bot = xt::view(conn, xt::keep(plastic), 0); // missing last node, but ok for test
        auto top = xt::view(conn, xt::keep(plastic), 3);

        for (size_t i = 0; i < 10; ++i) {
            double delta_gamma = 0.01;
            double gamma = delta_gamma * static_cast<double>(i + 1);
            sys.addAffineSimpleShearCentered(delta_gamma);
            auto u = sys.u();
            auto du = xt::eval(xt::view(u, xt::keep(top), 1) + xt::view(u, xt::keep(bot), 1));
            REQUIRE(xt::allclose(xt::view(sys.Eps(), xt::all(), xt::all(), 0, 1), gamma));
            REQUIRE(xt::allclose(du, 0.0));
        }
    }

    SECTION("System::addSimpleShearEventDriven")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
            xt::xtensor<size_t, 1>{},
            xt::xtensor<size_t, 1>{0});

        sys.setMassMatrix(xt::ones<double>({1}));
        sys.setDampingMatrix(xt::ones<double>({1}));

        sys.setElastic(
            xt::xtensor<double, 1>{},
            xt::xtensor<double, 1>{});

        sys.setPlastic(
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

        sys.setDt(1.0);

        REQUIRE(sys.isHomogeneousElastic());

        double delta_eps = 1e-3;
        auto dV = sys.quad().AsTensor<2>(sys.dV());
        xt::xtensor<double, 2> Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 0.0));

        sys.addSimpleShearEventDriven(delta_eps, false, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 - delta_eps / 2.0));

        sys.addSimpleShearEventDriven(delta_eps, true, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 + delta_eps / 2.0));

        sys.addSimpleShearEventDriven(delta_eps, false, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 - delta_eps / 2.0));

        sys.addSimpleShearEventDriven(delta_eps, true, +1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 + delta_eps / 2.0));

        sys.addSimpleShearEventDriven(delta_eps, false, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 + delta_eps / 2.0));

        sys.addSimpleShearEventDriven(delta_eps, true, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 2.0 - delta_eps / 2.0));

        sys.addSimpleShearEventDriven(delta_eps, false, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 + delta_eps / 2.0));

        sys.addSimpleShearEventDriven(delta_eps, true, -1.0);

        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), 1.0 - delta_eps / 2.0));
    }

    SECTION("System::addSimpleShearToFixedStress")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
            xt::xtensor<size_t, 1>{},
            xt::xtensor<size_t, 1>{0});

        sys.setMassMatrix(xt::ones<double>({1}));
        sys.setDampingMatrix(xt::ones<double>({1}));

        sys.setElastic(
            xt::xtensor<double, 1>{},
            xt::xtensor<double, 1>{});

        sys.setPlastic(
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

        sys.setDt(1.0);

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
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), target_stress / 2.0));
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)(), target_stress));

        target_stress = 1.5;
        sys.addSimpleShearToFixedStress(target_stress);
        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        Sigbar = xt::average(sys.Sig(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), target_stress / 2.0));
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Sigd(Sigbar)(), target_stress));

        target_stress = 0.5;
        sys.addSimpleShearToFixedStress(target_stress);
        Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        Sigbar = xt::average(sys.Sig(), dV, {0, 1});
        REQUIRE(xt::allclose(GMatElastoPlasticQPot::Cartesian2d::Epsd(Epsbar)(), target_stress / 2.0));
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

        FrictionQPotFEM::UniformSingleLayer2d::System sys(
            mesh.coor(),
            mesh.conn(),
            dofs,
            iip,
            xt::xtensor<size_t, 1>{0, 1, 2, 3, 5, 6, 7, 8},
            xt::xtensor<size_t, 1>{4});

        sys.setMassMatrix(xt::ones<double>({mesh.nelem()}));
        sys.setDampingMatrix(xt::ones<double>({mesh.nelem()}));

        sys.setElastic(
            1.0 * xt::ones<double>({8}),
            1.0 * xt::ones<double>({8}));

        sys.setPlastic(
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

        sys.setDt(1.0);

        REQUIRE(sys.isHomogeneousElastic());

        double delta_eps = 1e-3;
        sys.triggerElementWithLocalSimpleShear(delta_eps, 0);

        auto eps = GMatElastoPlasticQPot::Cartesian2d::Epsd(sys.Eps());
        auto plastic = sys.plastic();
        auto eps_p = xt::view(eps, xt::keep(plastic), xt::all());

        REQUIRE(xt::allclose(eps_p, 1.0 + delta_eps / 2.0));
    }

    SECTION("System::plastic_*")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

        FrictionQPotFEM::UniformSingleLayer2d::System full(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
            xt::xtensor<size_t, 1>{},
            xt::xtensor<size_t, 1>{0});

        full.setMassMatrix(xt::ones<double>({1}));
        full.setDampingMatrix(xt::ones<double>({1}));

        full.setElastic(
            xt::xtensor<double, 1>{},
            xt::xtensor<double, 1>{});

        full.setPlastic(
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

        full.setDt(1.0);

        FrictionQPotFEM::UniformSingleLayer2d::System reduced(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
            xt::xtensor<size_t, 1>{},
            xt::xtensor<size_t, 1>{0});

        reduced.setMassMatrix(xt::ones<double>({1}));
        reduced.setDampingMatrix(xt::ones<double>({1}));

        reduced.setElastic(
            xt::xtensor<double, 1>{},
            xt::xtensor<double, 1>{});

        reduced.setPlastic(
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 1>{1.0},
            xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

        reduced.setDt(1.0);

        double delta_eps = 1e-3;
        full.addSimpleShearEventDriven(delta_eps, false);
        reduced.addSimpleShearEventDriven(delta_eps, false);

        REQUIRE(xt::allclose(full.Eps(), reduced.Eps()));
        REQUIRE(xt::allclose(full.Eps(), full.plastic_Eps()));
        REQUIRE(xt::allclose(reduced.Eps(), reduced.plastic_Eps()));

        REQUIRE(xt::allclose(full.Sig(), reduced.Sig()));
        REQUIRE(xt::allclose(full.Sig(), full.plastic_Sig()));
        REQUIRE(xt::allclose(reduced.Sig(), reduced.plastic_Sig()));

        REQUIRE(xt::allclose(full.plastic_CurrentYieldLeft(), reduced.plastic_CurrentYieldLeft()));
        REQUIRE(xt::allclose(full.plastic_CurrentYieldRight(), reduced.plastic_CurrentYieldRight()));
        REQUIRE(xt::all(xt::equal(full.plastic_CurrentIndex(), reduced.plastic_CurrentIndex())));
    }
}
