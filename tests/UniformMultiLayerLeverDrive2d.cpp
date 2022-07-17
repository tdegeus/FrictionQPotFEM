#include <FrictionQPotFEM/UniformMultiLayerLeverDrive2d.h>
#include <GooseFEM/MeshQuad4.h>
#include <catch2/catch_all.hpp>
#include <xtensor/xrandom.hpp>

TEST_CASE("FrictionQPotFEM::UniformMultiLayerLeverDrive2d", "UniformMultiLayerLeverDrive2d.h")
{
    SECTION("Force from drive: prescribe mean of each layer")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(5, 1);

        xt::xtensor<bool, 1> is_plastic = {false, true, false, true, false};
        size_t nlayer = is_plastic.size();
        GooseFEM::Mesh::Vstack stitch;
        for (size_t i = 0; i < nlayer; ++i) {
            stitch.push_back(mesh.coor(), mesh.conn(), mesh.nodesBottomEdge(), mesh.nodesTopEdge());
        }

        auto dofs = stitch.dofs();
        auto bottom = stitch.nodeset(mesh.nodesBottomEdge(), 0);
        auto iip = xt::ravel(xt::view(dofs, xt::keep(bottom)));
        auto layers = stitch.elemmap();

        size_t nelas = 0;
        size_t nplas = 0;

        for (size_t i = 0; i < layers.size(); ++i) {
            if (is_plastic[i]) {
                nplas += layers[i].size();
            }
            else {
                nelas += layers[i].size();
            }
        }

        xt::xtensor<bool, 2> drive = xt::zeros<bool>({5, 2});
        xt::xtensor<double, 2> u_target = xt::zeros<double>({5, 2});
        xt::xtensor<double, 1> height = 0.5 * xt::arange<double>(5);

        drive(2, 0) = true;
        drive(4, 0) = true;

        u_target(2, 0) = height(2);
        u_target(4, 0) = height(4);

        FrictionQPotFEM::UniformMultiLayerLeverDrive2d::System sys(
            stitch.coor(),
            stitch.conn(),
            dofs,
            iip,
            layers,
            stitch.nodemap(),
            is_plastic,
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({nelas}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({nelas}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({nplas}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(xt::ones<double>({nplas}))),
            FrictionQPotFEM::epsy_initelastic_toquad(100 * xt::ones<double>({nplas, size_t(1)})),
            1,
            1,
            1,
            0,
            drive,
            1.0,
            height.back(),
            height);

        sys.layerSetUbar(u_target, drive);
        sys.setLeverTarget(1.0 * height.back());

        REQUIRE(sys.leverPosition() == Catch::Approx(sys.leverTarget()));
        REQUIRE(xt::allclose(sys.fdrive(), 0.0));
        REQUIRE(xt::allclose(sys.layerFdrive(), 0.0));
    }
}
