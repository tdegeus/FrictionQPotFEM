
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/UniformMultiLayerLeverDrive2d.h>
#include <GooseFEM/MeshQuad4.h>

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

        FrictionQPotFEM::UniformMultiLayerLeverDrive2d::System sys(
            stitch.coor(), stitch.conn(), dofs, iip, stitch.elemmap(), stitch.nodemap(), is_plastic);

        size_t nelas = sys.elastic().size();
        size_t nplas = sys.plastic().size();

        sys.setDt(1.0);
        sys.setMassMatrix(xt::eval(xt::ones<double>({stitch.nelem()})));
        sys.setDampingMatrix(xt::eval(xt::ones<double>({stitch.nelem()})));

        sys.setElastic(
            xt::eval(xt::ones<double>({nelas})),
            xt::eval(xt::ones<double>({nelas})));

        sys.setPlastic(
            xt::eval(xt::ones<double>({nplas})),
            xt::eval(xt::ones<double>({nplas})),
            xt::eval(xt::ones<double>({nplas, size_t(1)}) * 100.0));

        xt::xtensor<bool, 2> drive = xt::zeros<bool>({5, 2});
        xt::xtensor<double, 2> u_target = xt::zeros<double>({5, 2});
        xt::xtensor<double, 1> height = 0.5 * xt::arange<double>(5);

        drive(2, 0) = true;
        drive(4, 0) = true;

        u_target(2, 0) = height(2);
        u_target(4, 0) = height(4);

        sys.setLeverProperties(height.back(), height);
        sys.layerSetDriveStiffness(1.0);
        sys.layerSetTargetActive(drive);
        sys.layerSetUbar(u_target, drive);
        sys.setLeverTarget(1.0 * height.back());

        REQUIRE(sys.leverPosition() == Approx(sys.leverTarget()));
        REQUIRE(xt::allclose(sys.fdrive(), 0.0));
        REQUIRE(xt::allclose(sys.layerFdrive(), 0.0));
    }
}
