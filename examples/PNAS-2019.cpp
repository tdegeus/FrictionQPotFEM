
#include <FrictionQPotFEM/UniformSingleLayer2d.h>
#include <fmt/core.h>
#include <highfive/H5Easy.hpp>

#define MYASSERT(expr) MYASSERT_IMPL(expr, __FILE__, __LINE__)
#define MYASSERT_IMPL(expr, file, line) \
    if (!(expr)) { \
        throw std::runtime_error( \
            std::string(file) + ':' + std::to_string(line) + \
            ": assertion failed (" #expr ") \n\t"); \
    }

int main(int argc, char* argv[])
{
    MYASSERT(argc == 2);

    H5Easy::File data(argv[1], H5Easy::File::ReadOnly);

    FrictionQPotFEM::UniformSingleLayer2d::System sys(
        H5Easy::load<xt::xtensor<double, 2>>(data, "/coor"),
        H5Easy::load<xt::xtensor<size_t, 2>>(data, "/conn"),
        H5Easy::load<xt::xtensor<size_t, 2>>(data, "/dofs"),
        H5Easy::load<xt::xtensor<size_t, 1>>(data, "/dofsP"),
        H5Easy::load<xt::xtensor<size_t, 1>>(data, "/elastic/elem"),
        FrictionQPotFEM::moduli_toquad(H5Easy::load<xt::xtensor<double, 1>>(data, "/elastic/K")),
        FrictionQPotFEM::moduli_toquad(H5Easy::load<xt::xtensor<double, 1>>(data, "/elastic/G")),
        H5Easy::load<xt::xtensor<size_t, 1>>(data, "/cusp/elem"),
        FrictionQPotFEM::moduli_toquad(H5Easy::load<xt::xtensor<double, 1>>(data, "/cusp/K")),
        FrictionQPotFEM::moduli_toquad(H5Easy::load<xt::xtensor<double, 1>>(data, "/cusp/G")),
        FrictionQPotFEM::epsy_initelastic_toquad(
            H5Easy::load<xt::xtensor<double, 2>>(data, "/cusp/epsy")),
        H5Easy::load<double>(data, "/run/dt"),
        FrictionQPotFEM::getuniform(H5Easy::load<xt::xtensor<double, 1>>(data, "/rho")),
        FrictionQPotFEM::getuniform(H5Easy::load<xt::xtensor<double, 1>>(data, "/damping/alpha")),
        0);

    size_t ninc = xt::amax(H5Easy::load<xt::xtensor<size_t, 1>>(data, "/stored"))();
    bool kick = true;
    double deps = H5Easy::load<double>(data, "/run/epsd/kick");

    sys.initEventDrivenSimpleShear();

    for (size_t step = 1; step < ninc; ++step) {

        size_t inc_n = sys.inc();
        sys.eventDrivenStep(deps, kick);

        if (kick) {
            auto niter = sys.minimise(1);
            MYASSERT(niter >= 0);
        }
        else {
            MYASSERT(sys.residual() < 1e-2);
        }

        kick = !kick;

        auto u = sys.u();
        auto uref = H5Easy::load<xt::xtensor<double, 2>>(data, fmt::format("/disp/{0:d}", step));
        double u_res = xt::norm_l2(u - uref)() / xt::norm_l2(u)();
        MYASSERT(xt::norm_l2(u - uref)() / xt::norm_l2(u)() < 1e-5);

        fmt::print(
            "step = {0:8d}, kick = {1:1d}, iiter = {2:8d}, disp_error = {3:.2e}\n",
            step,
            kick,
            sys.inc() - inc_n,
            u_res);
    }

    return 0;
}
