
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>
#include <GooseFEM/MeshQuad4.h>

TEST_CASE("FrictionQPotFEM::UniformMultiLayerIndividualDrive2d", "UniformMultiLayerIndividualDrive2d.h")
{
    SECTION("Basic")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(5, 1);

        size_t nlayer = 5;
        GooseFEM::Mesh::Vstack stitch;
        for (size_t i = 0; i < nlayer; ++i) {
            stitch.push_back(mesh.coor(), mesh.conn(), mesh.nodesBottomEdge(), mesh.nodesTopEdge());
        }

        xt::xtensor<bool, 1> is_plastic = {false, true, false, true, false};
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
            stitch.coor(), stitch.conn(), dofs, iip, elem, node, is_plastic);

        for (size_t i = 0; i < is_plastic.size(); ++i) {
            REQUIRE(is_plastic(i) == sys.layerIsPlastic(i));
        }
    }

    SECTION("Drive")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(5, 1);

        size_t nlayer = 5;
        GooseFEM::Mesh::Vstack stitch;
        for (size_t i = 0; i < nlayer; ++i) {
            stitch.push_back(mesh.coor(), mesh.conn(), mesh.nodesBottomEdge(), mesh.nodesTopEdge());
        }

        xt::xtensor<bool, 1> is_plastic = {false, true, false, true, false};
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
            stitch.coor(), stitch.conn(), dofs, iip, elem, node, is_plastic);

        xt::xtensor<bool, 2> drive = xt::zeros<bool>({5, 2});
        xt::xtensor<double, 2> u_target = xt::zeros<double>({5, 2});

        drive(2, 0) = true;
        drive(4, 0) = true;

        u_target(2, 0) = 1.0;
        u_target(4, 0) = 2.0;

        sys.layerSetUbar(u_target, drive);

        sys.setDriveStiffness(1.0);

        xt::xtensor<double, 2> f = xt::zeros<double>({stitch.nnode(), stitch.ndim()});

        {
            xt::xtensor<size_t, 1> c0 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesBottomLeftCorner()}, 2);
            xt::xtensor<size_t, 1> c1 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesBottomRightCorner()}, 2);
            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesBottomOpenEdge(), 2);

            xt::view(f, xt::keep(c0), 0) -= 0.25;
            xt::view(f, xt::keep(c1), 0) -= 0.25;
            xt::view(f, xt::keep(e), 0) -= 0.5;
        }

        {
            xt::xtensor<size_t, 1> c0 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesTopLeftCorner()}, 2);
            xt::xtensor<size_t, 1> c1 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesTopRightCorner()}, 2);
            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesTopOpenEdge(), 2);

            xt::view(f, xt::keep(c0), 0) -= 0.25;
            xt::view(f, xt::keep(c1), 0) -= 0.25;
            xt::view(f, xt::keep(e), 0) -= 0.5;
        }

        {
            xt::xtensor<size_t, 1> c0 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesBottomLeftCorner()}, 4);
            xt::xtensor<size_t, 1> c1 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesBottomRightCorner()}, 4);
            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesBottomOpenEdge(), 4);

            xt::view(f, xt::keep(c0), 0) -= 0.5;
            xt::view(f, xt::keep(c1), 0) -= 0.5;
            xt::view(f, xt::keep(e), 0) -= 1.0;
        }

        {
            xt::xtensor<size_t, 1> c0 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesTopLeftCorner()}, 4);
            xt::xtensor<size_t, 1> c1 = stitch.nodeset(xt::xtensor<size_t, 1>{mesh.nodesTopRightCorner()}, 4);
            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesTopOpenEdge(), 4);

            xt::view(f, xt::keep(c0), 0) -= 0.5;
            xt::view(f, xt::keep(c1), 0) -= 0.5;
            xt::view(f, xt::keep(e), 0) -= 1.0;
        }

        REQUIRE(xt::allclose(f, sys.fdrive()));
    }

    SECTION("Drive, distribute")
    {
        GooseFEM::Mesh::Quad4::Regular mesh(5, 1);

        size_t nlayer = 5;
        GooseFEM::Mesh::Vstack stitch;
        for (size_t i = 0; i < nlayer; ++i) {
            stitch.push_back(mesh.coor(), mesh.conn(), mesh.nodesBottomEdge(), mesh.nodesTopEdge());
        }

        xt::xtensor<bool, 1> is_plastic = {false, true, false, true, false};
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
            stitch.coor(), stitch.conn(), dofs, iip, elem, node, is_plastic);

        auto nelas = sys.elastic().size();
        auto nplas = sys.plastic().size();

        sys.setDt(1.0);
        sys.setMassMatrix(xt::eval(xt::ones<double>({stitch.nelem()})));
        sys.setDampingMatrix(xt::eval(xt::ones<double>({stitch.nelem()})));

        sys.setElastic(
            xt::eval(xt::ones<double>({nelas})),
            xt::eval(xt::ones<double>({nelas})));

        sys.setPlastic(
            xt::eval(xt::ones<double>({nplas})),
            xt::eval(xt::ones<double>({nplas})),
            xt::eval(xt::ones<double>({nplas, 1ul}) * 100.0));

        xt::xtensor<bool, 2> drive = xt::zeros<bool>({5, 2});
        xt::xtensor<double, 2> u_target = xt::zeros<double>({5, 2});

        drive(2, 0) = true;
        drive(4, 0) = true;

        u_target(2, 0) = 1.0;
        u_target(4, 0) = 2.0;

        sys.setDriveStiffness(1.0);
        sys.layerSetDistributeUbar(u_target, drive);

        xt::xtensor<double, 2> u = xt::zeros<double>({stitch.nnode(), stitch.ndim()});
        auto n2 = stitch.nodemap(2);
        auto n4 = stitch.nodemap(4);
        xt::view(u, xt::keep(n2), 0) = 1.0;
        xt::view(u, xt::keep(n4), 0) = 2.0;

        REQUIRE(xt::allclose(sys.u(), u));
    }

    SECTION("Simple example")
    {
        GooseFEM::Mesh::Quad4::Regular layer_elas(20, 6);
        GooseFEM::Mesh::Quad4::Regular layer_plas(20, 1);

        auto x0 = layer_elas.coor();
        auto x1 = layer_plas.coor();
        auto x2 = layer_elas.coor();

        xt::view(x1, xt::all(), 1) += xt::amax(xt::view(x0, xt::all(), 1));
        xt::view(x2, xt::all(), 1) += xt::amax(xt::view(x1, xt::all(), 1));

        GooseFEM::Mesh::Stitch stitch;

        stitch.push_back(x0, layer_elas.conn());
        stitch.push_back(x1, layer_plas.conn());
        stitch.push_back(x2, layer_elas.conn());

        auto left = stitch.nodeset({
            layer_elas.nodesLeftOpenEdge(),
            layer_plas.nodesLeftEdge(),
            layer_elas.nodesLeftOpenEdge()
        });

        auto right = stitch.nodeset({
            layer_elas.nodesRightOpenEdge(),
            layer_plas.nodesRightEdge(),
            layer_elas.nodesRightOpenEdge()
        });

        auto bottom = stitch.nodeset(layer_elas.nodesBottomEdge(), 0);
        auto top = stitch.nodeset(layer_elas.nodesTopEdge(), 2);

        auto nelem = stitch.nelem();
        auto coor = stitch.coor();
        auto conn = stitch.conn();
        auto dofs = stitch.dofs();

        xt::view(dofs, xt::keep(right), xt::all()) = xt::view(dofs, xt::keep(left), xt::all());
        dofs(top(0), 0) =  dofs(top.periodic(-1), 0);
        dofs(bottom(0), 0) =  dofs(bottom.periodic(-1), 0);
        dofs = GooseFEM::Mesh::renumber(dofs);

        xt::xtensor<size_t, 1> iip = xt::empty<size_t>({bottom.size() * 2 + top.size()});

        xt::view(iip, xt::range(0, bottom.size())) =
            xt::view(dofs, xt::keep(bottom), 0);

        xt::view(iip, xt::range(bottom.size(), 2 * bottom.size())) =
            xt::view(dofs, xt::keep(bottom), 1);

        xt::view(iip, xt::range(2 * bottom.size(), 2 * bottom.size() + top.size())) =
            xt::view(dofs, xt::keep(top), 1);

        auto elas = xt::concatenate(xt::xtuple(stitch.elemmap(0), stitch.elemmap(2)));
        auto plas = stitch.elemmap(1);
        xt::xtensor<bool, 1> is_plastic = {false, true, false};

        FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System sys(
            coor, conn, dofs, iip, stitch.elemmap(), stitch.nodemap(), is_plastic);

        xt::xtensor<double, 2> epsy = 0.01 * xt::ones<double>({plas.size(), size_t(100)});
        epsy = xt::cumsum(epsy, 1);

        sys.setMassMatrix(1.0 * xt::ones<double>({nelem}));
        sys.setDampingMatrix(0.01 * xt::ones<double>({nelem}));
        sys.setElastic(10.0 * xt::ones<double>({elas.size()}), 1.0 * xt::ones<double>({elas.size()}));
        sys.setPlastic(10.0 * xt::ones<double>({plas.size()}), 1.0 * xt::ones<double>({plas.size()}), epsy);
        sys.setDt(0.1);
        sys.setDriveStiffness(1.0);

        xt::xtensor<bool, 2> drive = xt::zeros<bool>({3, 2});
        xt::xtensor<double, 2> u_target = xt::zeros<double>({3, 2});
        drive(0, 0) = true;
        drive(2, 0) = true;
        u_target(2, 0) = 0.1;
        sys.layerSetUbar(u_target, drive);

        sys.minimise();

        u_target(1, 0) = 0.05; // expected result

        REQUIRE(xt::allclose(u_target, sys.layerUbar(), 5e-2, 5e-3));
    }
}
