
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>

TEST_CASE("FrictionQPotFEM::UniformSingleLayer2d_LocalTrigger", "UniformSingleLayer2d.h")
{
    SECTION("LocalTrigger")
    {
        size_t N = 6;
        double h = xt::numeric_constants<double>::PI;
        double L = h * static_cast<double>(N);

        GooseFEM::Mesh::Quad4::FineLayer mesh(N, N, h);

        auto coor = mesh.coor();
        auto conn = mesh.conn();
        auto dofs = mesh.dofs();

        xt::xtensor<size_t, 1> plastic = mesh.elementsMiddleLayer();
        xt::xtensor<size_t, 1> elastic = xt::setdiff1d(xt::arange(mesh.nelem()), plastic);

        auto left = mesh.nodesLeftOpenEdge();
        auto right = mesh.nodesRightOpenEdge();
        xt::view(dofs, xt::keep(right), 0) = xt::view(dofs, xt::keep(left), 0);
        xt::view(dofs, xt::keep(right), 1) = xt::view(dofs, xt::keep(left), 1);

        auto top = mesh.nodesTopEdge();
        auto bottom = mesh.nodesBottomEdge();
        size_t nfix = top.size();
        xt::xtensor<size_t, 1> iip = xt::empty<size_t>({2 * mesh.ndim() * nfix});
        xt::view(iip, xt::range(0 * nfix, 1 * nfix)) = xt::view(dofs, xt::keep(bottom), 0);
        xt::view(iip, xt::range(1 * nfix, 2 * nfix)) = xt::view(dofs, xt::keep(bottom), 1);
        xt::view(iip, xt::range(2 * nfix, 3 * nfix)) = xt::view(dofs, xt::keep(top), 0);
        xt::view(iip, xt::range(3 * nfix, 4 * nfix)) = xt::view(dofs, xt::keep(top), 1);

        double c = 1.0;
        double G = 1.0;
        double K = 10.0 * G;
        double rho = G / std::pow(c, 2.0);
        double qL = 2.0 * xt::numeric_constants<double>::PI / L;
        double qh = 2.0 * xt::numeric_constants<double>::PI / h;
        double alpha = std::sqrt(2.0) * qL * c * rho;
        double dt = 1.0 / (c * qh) / 10.0;

        xt::xtensor<double, 2> epsy = xt::ones<double>(std::array<size_t, 2>{N, 1000});
        epsy = xt::cumsum(epsy, 1);

        // Initialise system

        FrictionQPotFEM::UniformSingleLayer2d::System sys(coor, conn, dofs, iip, elastic, plastic);
        sys.setMassMatrix(rho * xt::ones<double>({mesh.nelem()}));
        sys.setDampingMatrix(alpha * xt::ones<double>({mesh.nelem()}));
        sys.setElastic(K * xt::ones<double>({elastic.size()}), G * xt::ones<double>({elastic.size()}));
        sys.setPlastic(K * xt::ones<double>({plastic.size()}), G * xt::ones<double>({plastic.size()}), epsy);
        sys.setDt(dt);

        {
            FrictionQPotFEM::UniformSingleLayer2d::LocalTriggerFineLayerFull trigger(sys);
            xt::xtensor<double, 2> epsy = xt::ones<double>({plastic.size(), size_t(4)});
            trigger.setState(sys.Eps(), sys.Sig(), epsy);
            xt::xtensor<double, 2> dE = 0.150701 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> dgamma = 0.373301 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> barriers = 5.357607 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> p = xt::zeros<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> s = 2.678804 * xt::ones<double>({plastic.size(), size_t(4)});
            REQUIRE(xt::allclose(dE, trigger.dE()));
            REQUIRE(xt::allclose(dgamma, trigger.dgamma()));
            REQUIRE(xt::allclose(barriers, trigger.barriers()));
            REQUIRE(xt::allclose(p, trigger.p()));
            REQUIRE(xt::allclose(s, xt::abs(trigger.s())));
        }

        {
            FrictionQPotFEM::UniformSingleLayer2d::LocalTriggerFineLayer trigger(sys);
            xt::xtensor<double, 2> epsy = xt::ones<double>({plastic.size(), size_t(4)});
            trigger.setState(sys.Eps(), sys.Sig(), epsy);
            xt::xtensor<double, 2> dE = 0.150701 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> dgamma = 0.373301 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> barriers = 5.357607 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> p = xt::zeros<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> s = 2.678804 * xt::ones<double>({plastic.size(), size_t(4)});
            REQUIRE(xt::allclose(dE, trigger.dE()));
            REQUIRE(xt::allclose(dgamma, trigger.dgamma()));
            REQUIRE(xt::allclose(barriers, trigger.barriers()));
            REQUIRE(xt::allclose(p, trigger.p()));
            REQUIRE(xt::allclose(s, xt::abs(trigger.s())));
        }

        {
            FrictionQPotFEM::UniformSingleLayer2d::LocalTriggerFineLayerFull trigger(sys);
            xt::xtensor<double, 2> epsy = xt::ones<double>({plastic.size(), size_t(4)});
            trigger.setStateMinimalSearch(sys.Eps(), sys.Sig(), epsy);
            xt::xtensor<double, 2> dE = 0.150701 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> dgamma = 0.373301 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> barriers = 5.357607 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> p = xt::zeros<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> s = 2.678804 * xt::ones<double>({plastic.size(), size_t(4)});
            REQUIRE(xt::allclose(dE, trigger.dE()));
            REQUIRE(xt::allclose(dgamma, trigger.dgamma()));
            REQUIRE(xt::allclose(barriers, trigger.barriers()));
            REQUIRE(xt::allclose(p, trigger.p()));
            REQUIRE(xt::allclose(s, xt::abs(trigger.s())));
        }

        {
            FrictionQPotFEM::UniformSingleLayer2d::LocalTriggerFineLayerFull trigger(sys);
            xt::xtensor<double, 2> epsy = xt::ones<double>({plastic.size(), size_t(4)});
            trigger.setStateSimpleShear(sys.Eps(), sys.Sig(), epsy);
            xt::xtensor<double, 2> dE = 0.150701 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> dgamma = 0.373301 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> barriers = 5.357607 * xt::ones<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> p = xt::zeros<double>({plastic.size(), size_t(4)});
            xt::xtensor<double, 2> s = 2.678804 * xt::ones<double>({plastic.size(), size_t(4)});
            REQUIRE(xt::allclose(dE, trigger.dE()));
            REQUIRE(xt::allclose(dgamma, trigger.dgamma()));
            REQUIRE(xt::allclose(barriers, trigger.barriers()));
            REQUIRE(xt::allclose(p, trigger.p()));
            REQUIRE(xt::allclose(s, xt::abs(trigger.s())));
        }
    }
}
