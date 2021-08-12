
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/Generic2d.h>
#include <iostream>

TEST_CASE("FrictionQPotFEM::Generic2d", "Generic2d.h")
{
    SECTION("Compare System and HybridSystem")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

        FrictionQPotFEM::Generic2d::System full(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            xt::eval(xt::arange<size_t>(mesh.nnode() * mesh.ndim())),
            xt::xtensor<size_t, 1>{0, 1, 2, 3, 5, 6, 7, 8},
            xt::xtensor<size_t, 1>{4});

        full.setMassMatrix(xt::ones<double>({mesh.nelem()}));
        full.setDampingMatrix(xt::ones<double>({mesh.nelem()}));

        full.setElastic(
            xt::ones<double>({8}),
            xt::ones<double>({8}));

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
            xt::xtensor<size_t, 1>{0, 1, 2, 3, 5, 6, 7, 8},
            xt::xtensor<size_t, 1>{4});

        reduced.setMassMatrix(xt::ones<double>({mesh.nelem()}));
        reduced.setDampingMatrix(xt::ones<double>({mesh.nelem()}));

        reduced.setElastic(
            xt::ones<double>({8}),
            xt::ones<double>({8}));

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
        REQUIRE(xt::allclose(full.plastic_Eps(), reduced.plastic_Eps()));

        REQUIRE(xt::allclose(full.Sig(), reduced.Sig()));
        REQUIRE(xt::allclose(full.plastic_Sig(), reduced.plastic_Sig()));

        REQUIRE(xt::allclose(full.plastic_CurrentYieldLeft(), reduced.plastic_CurrentYieldLeft()));
        REQUIRE(xt::allclose(full.plastic_CurrentYieldRight(), reduced.plastic_CurrentYieldRight()));
        REQUIRE(xt::all(xt::equal(full.plastic_CurrentIndex(), reduced.plastic_CurrentIndex())));
    }
}
