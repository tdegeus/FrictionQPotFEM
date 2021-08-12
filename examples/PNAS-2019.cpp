#include <stdexcept> // can be removed for QPot > 0.9.6
#include <string> // can be removed for QPot > 0.9.6
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <highfive/H5Easy.hpp>
#include <fmt/core.h>

#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

int main(int argc, char *argv[])
{
    MYASSERT(argc == 2);

    H5Easy::File data(argv[1], H5Easy::File::ReadOnly);

    FrictionQPotFEM::UniformSingleLayer2d::System sys(
        H5Easy::load<xt::xtensor<double, 2>>(data, "/coor"),
        H5Easy::load<xt::xtensor<size_t, 2>>(data, "/conn"),
        H5Easy::load<xt::xtensor<size_t, 2>>(data, "/dofs"),
        H5Easy::load<xt::xtensor<size_t, 1>>(data, "/dofsP"),
        H5Easy::load<xt::xtensor<size_t, 1>>(data, "/elastic/elem"),
        H5Easy::load<xt::xtensor<size_t, 1>>(data, "/cusp/elem"));

    sys.setMassMatrix(H5Easy::load<xt::xtensor<double, 1>>(data, "/rho"));
    sys.setDampingMatrix(H5Easy::load<xt::xtensor<double, 1>>(data, "/damping/alpha"));

    sys.setElastic(
        H5Easy::load<xt::xtensor<double, 1>>(data, "/elastic/K"),
        H5Easy::load<xt::xtensor<double, 1>>(data, "/elastic/G"));

    sys.setPlastic(
        H5Easy::load<xt::xtensor<double, 1>>(data, "/cusp/K"),
        H5Easy::load<xt::xtensor<double, 1>>(data, "/cusp/G"),
        H5Easy::load<xt::xtensor<double, 2>>(data, "/cusp/epsy"));

    sys.setDt(H5Easy::load<double>(data, "/run/dt"));

    size_t ninc = xt::amax(H5Easy::load<xt::xtensor<size_t, 1>>(data, "/stored"))();
    bool kick = true;
    double deps = H5Easy::load<double>(data, "/run/epsd/kick");
    size_t niter;

    for (size_t inc = 1; inc < ninc; ++inc) {

        sys.addSimpleShearEventDriven(deps, kick);

        if (kick) {
            niter = sys.minimise();
        }
        else {
            niter = 0;
            MYASSERT(sys.residual() < 1e-2);
        }

        kick = !kick;

        auto u = sys.u();
        auto u_stored = H5Easy::load<xt::xtensor<double, 2>>(data, fmt::format("/disp/{0:d}", inc));
        double u_res = xt::norm_l2(u - u_stored)() / xt::norm_l2(u)();
        MYASSERT(xt::norm_l2(u - u_stored)() / xt::norm_l2(u)() < 1e-5);

        fmt::print("inc = {0:8d}, kick = {1:1d}, iiter = {2:8d}, disp_error = {3:.2e}\n",
            inc, kick, niter, u_res);
    }

    return 0;
}
