
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <fmt/core.h>
#include <xtensor/xrandom.hpp>
#include <xtensor/xcsv.hpp>
#include <fstream>

int main()
{
    // Define a geometry

    size_t N = std::pow(3, 4);
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

    double k = 2.0;
    xt::xtensor<double, 2> epsy = 1e-5 + 1e-3 * xt::random::weibull<double>(std::array<size_t, 2>{N, 1000}, k, 1.0);
    xt::view(epsy, xt::all(), 0) = 1e-5 + 1e-3 * xt::random::rand<double>({N});
    epsy = xt::cumsum(epsy, 1);

    // Initialise system

    FrictionQPotFEM::UniformSingleLayer2d::HybridSystem sys(coor, conn, dofs, iip, elastic, plastic);
    sys.setMassMatrix(rho * xt::ones<double>({mesh.nelem()}));
    sys.setDampingMatrix(alpha * xt::ones<double>({mesh.nelem()}));
    sys.setElastic(K * xt::ones<double>({elastic.size()}), G * xt::ones<double>({elastic.size()}));
    sys.setPlastic(K * xt::ones<double>({plastic.size()}), G * xt::ones<double>({plastic.size()}), epsy);
    sys.setDt(dt);

    // Run

    size_t ninc = 200;
    xt::xtensor<double, 2> ret = xt::zeros<double>(std::array<size_t, 2>{ninc, 2});
    auto dV = sys.AsTensor<2>(sys.dV());
    bool kick = true;

    for (size_t inc = 0; inc < ninc; ++inc) {
        sys.addSimpleShearEventDriven(1e-5, kick, +1.0);
        if (kick) {
            size_t niter = sys.minimise();
            fmt::print("inc = {0:3d}, niter = {1:d}\n", inc, niter);
        }
        kick = !kick;

        xt::xtensor<double, 2> Epsbar = xt::average(sys.Eps(), dV, {0, 1});
        xt::xtensor<double, 2> Sigbar = xt::average(sys.Sig(), dV, {0, 1});

        ret(inc, 0) = Epsbar(0, 1);
        ret(inc, 1) = Sigbar(0, 1);
    }

    std::ofstream outfile("Load.txt");
    xt::dump_csv(outfile, ret);
    outfile.close();

    return 0;
}
