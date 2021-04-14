
// #include <catch2/catch.hpp>
// #include <xtensor/xrandom.hpp>
// #include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>

// TEST_CASE("FrictionQPotFEM::UniformMultiLayerIndividualDrive2d", "UniformMultiLayerIndividualDrive2d.h")
// {
//     SECTION("Basic")
//     {
//         GooseFEM::Mesh::Quad4::Regular mesh(5, 1);

//         auto x0 = mesh.coor();
//         auto x1 = mesh.coor();
//         auto x2 = mesh.coor();
//         auto x3 = mesh.coor();
//         auto x4 = mesh.coor();

//         xt::view(x1, xt::all(), 1) += 1.0;
//         xt::view(x2, xt::all(), 1) += 2.0;
//         xt::view(x3, xt::all(), 1) += 3.0;
//         xt::view(x4, xt::all(), 1) += 4.0;

//         GooseFEM::Mesh::Stich stich();
//         stich.push_back(x0, mesh.conn());
//         stich.push_back(x1, mesh.conn());
//         stich.push_back(x2, mesh.conn());
//         stich.push_back(x3, mesh.conn());
//         stich.push_back(x4, mesh.conn());

//         xt::xtensor<bool, 1> is_plastic = {false, true, false, true, false};
//         std::vector<xt::xtensor<size_t, 1>> elem;
//         for (size_t i = 0; i < is_plastic.size(); ++i) {
//             elem.push_back(stich.elemmap(i));
//         }

//         auto dofs = GooseFEM::Mesh::dofs(stich.coor().shape(0), stich.coor().shape(1));
//         auto bottom = stich.nodeset(mesh.nodesBottomEdge(), 0);
//         auto iip = xt::ravel(xt::view(dofs, xt::keep(bottom)));

//         FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::Sytem sys(
//             stich.coor(), stich.conn(), dofs, iip, elem, is_plastic)

//         for (size_t i = 0; i < is_plastic.size(); ++i) {
//             REQUIRE(is_plastic(i) == sys.layerIsPlastic(i));
//             REQUIRE(is_plastic(i) == sys.layerNodes(i));
//         }
//     }
// }
