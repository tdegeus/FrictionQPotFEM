
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>

#define ISCLOSE(a, b) REQUIRE_THAT((a), Catch::WithinAbs((b), 1e-8));

TEST_CASE("FrictionQPotFEM::UniformSingleLayer2d", "UniformSingleLayer2d.h")
{

SECTION("System vs. HybridSystem")
{
    // Define a geometry

    size_t N = std::pow(3, 2);
    double h = xt::numeric_constants<double>::PI;
    double L = h * static_cast<double>(N);

    GF::Mesh::Quad4::FineLayer mesh(N, N, h);

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

    double k = 2.0;
    xt::xtensor<double, 2> epsy = 1e-5 + 1e-3 * xt::random::weibull<double>(std::array<size_t, 2>{N, 1000}, k, 1.0);
    xt::view(epsy, xt::all(), 0) = 1e-5 + 1e-3 * xt::random::rand<double>({N});
    epsy = xt::cumsum(epsy, 1);

    // Initialise system

    FrictionQPotFEM::UniformSingleLayer2d::System full(coor, conn, dofs, iip, elastic, plastic);
    FrictionQPotFEM::UniformSingleLayer2d::HybridSystem reduced(coor, conn, dofs, iip, elastic, plastic);

    full.setMassMatrix(rho * xt::ones<double>({mesh.nelem()}));
    reduced.setMassMatrix(rho * xt::ones<double>({mesh.nelem()}));

    full.setDampingMatrix(alpha * xt::ones<double>({mesh.nelem()}));
    reduced.setDampingMatrix(alpha * xt::ones<double>({mesh.nelem()}));

    full.setElastic(K * xt::ones<double>({elastic.size()}), G * xt::ones<double>({elastic.size()}));
    reduced.setElastic(K * xt::ones<double>({elastic.size()}), G * xt::ones<double>({elastic.size()}));

    full.setPlastic(K * xt::ones<double>({plastic.size()}), G * xt::ones<double>({plastic.size()}), epsy);
    reduced.setPlastic(K * xt::ones<double>({plastic.size()}), G * xt::ones<double>({plastic.size()}), epsy);

    full.setDt(dt);
    reduced.setDt(dt);

    // Run

    xt::xtensor<double, 3> dF = xt::zeros<double>({1001, 2, 2});
    xt::view(dF, xt::range(1, dF.shape(0)), 0, 1) = 0.004 / 1000.0;

    xt::xtensor<double, 2> ret = xt::zeros<double>(std::array<size_t, 2>{dF.shape(0), 2});
    auto dV = full.AsTensor<2>(full.dV());
    REQUIRE(xt::allclose(dV, reduced.AsTensor<2>(reduced.dV())));

    GF::Iterate::StopList stop(20);

    for (size_t inc = 0 ; inc < dF.shape(0); ++inc) {

        REQUIRE(xt::allclose(full.u(), reduced.u()));

        auto u = full.u();

        for (size_t i = 0; i < mesh.nnode(); ++i) {
            for (size_t j = 0; j < mesh.ndim(); ++j) {
                for (size_t k = 0; k < mesh.ndim(); ++k) {
                    u(i, j) += dF(inc, j, k) * (coor(i, k) - coor(0, k));
                }
            }
        }

        full.setU(u);
        reduced.setU(u);

        REQUIRE(xt::allclose(full.u(), reduced.u()));

        for (size_t iiter = 0; iiter < 99999 ; ++iiter) {

            full.timeStep();
            reduced.timeStep();

            REQUIRE(xt::allclose(full.fmaterial(), reduced.fmaterial()));
            REQUIRE(xt::allclose(full.u(), reduced.u()));
            REQUIRE(full.t() == Approx(reduced.t()));
            ISCLOSE(full.residual(), reduced.residual());

            if (stop.stop(full.residual(), 1e-5)) {
                std::cout << inc << ", " << iiter << std::endl;
                break;
            }
        }

        full.quench();
        reduced.quench();

        stop.reset();

        // xt::xtensor<double, 2> Epsbar = xt::average(m_Eps, dV, {0, 1});
        // xt::xtensor<double, 2> Sigbar = xt::average(m_Sig, dV, {0, 1});

        // ret(inc, 0) = GM::Epsd(Epsbar)();
        // ret(inc, 1) = GM::Epsd(Sigbar)();
    }



}

}
