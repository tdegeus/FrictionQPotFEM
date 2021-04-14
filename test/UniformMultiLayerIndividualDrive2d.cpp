
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>

TEST_CASE("FrictionQPotFEM::UniformMultiLayerIndividualDrive2d", "UniformMultiLayerIndividualDrive2d.h")
{
    SECTION("Basic")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(5, 1);

        size_t nlayer = 5;

        auto x0 = mesh.coor();
        auto x1 = mesh.coor();
        auto x2 = mesh.coor();
        auto x3 = mesh.coor();
        auto x4 = mesh.coor();

        xt::view(x1, xt::all(), 1) += 1.0;
        xt::view(x2, xt::all(), 1) += 2.0;
        xt::view(x3, xt::all(), 1) += 3.0;
        xt::view(x4, xt::all(), 1) += 4.0;

        GooseFEM::Mesh::Stitch stitch;
        stitch.push_back(x0, mesh.conn());
        stitch.push_back(x1, mesh.conn());
        stitch.push_back(x2, mesh.conn());
        stitch.push_back(x3, mesh.conn());
        stitch.push_back(x4, mesh.conn());

        xt::xtensor<bool, 1> is_plastic = {false, true, false, true, false};
        xt::xtensor<bool, 1> is_virtual = xt::zeros<bool>({stitch.nnode()});
        std::vector<xt::xtensor<size_t, 1>> elem;
        std::vector<xt::xtensor<size_t, 1>> node;
        for (size_t i = 0; i < nlayer; ++i) {
            elem.push_back(stitch.elemmap(i));
            node.push_back(stitch.nodemap(i));
        }

        auto dofs = GooseFEM::Mesh::dofs(stitch.coor().shape(0), stitch.coor().shape(1));
        auto bottom = stitch.nodeset(mesh.nodesBottomEdge(), 0);
        auto iip = xt::ravel(xt::view(dofs, xt::keep(bottom)));

        FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System sys(
            stitch.coor(), stitch.conn(), dofs, iip, elem, node, is_plastic, is_virtual);

        for (size_t i = 0; i < is_plastic.size(); ++i) {
            REQUIRE(is_plastic(i) == sys.layerIsPlastic(i));
        }
    }
}
