
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/Generic2d.h>
#include <iostream>

TEST_CASE("FrictionQPotFEM::Generic2d", "Generic2d.h")
{
    SECTION("Compare System and HybridSystem")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

        FrictionQPotFEM::Generic2d::System full(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::eval(xt::arange<size_t>(mesh.nnode() * mesh.ndim())),
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

        FrictionQPotFEM::Generic2d::HybridSystem reduced(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::eval(xt::arange<size_t>(mesh.nnode() * mesh.ndim())),
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

        auto u = full.u();
        auto coor = full.coor();
        double delta_gamma = 2.0;

        for (size_t n = 0; n < coor.shape(0); ++n) {
            u(n, 0) += 2.0 * delta_gamma * (coor(n, 1) - coor(0, 1));
        }

        full.setU(u);
        reduced.setU(u);

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
