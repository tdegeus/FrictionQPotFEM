#include <FrictionQPotFEM/UniformMultiLayerIndividualDrive2d.h>
#include <GooseFEM/MeshQuad4.h>
#include <catch2/catch_all.hpp>
#include <xtensor/xrandom.hpp>

TEST_CASE(
    "FrictionQPotFEM::UniformMultiLayerIndividualDrive2d",
    "UniformMultiLayerIndividualDrive2d.h")
{
    SECTION("Basic")
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

        FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System sys(
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
            FrictionQPotFEM::epsy_initelastic_toquad(xt::ones<double>({nplas, size_t(1)})),
            1,
            1,
            1,
            0,
            xt::ones<bool>({layers.size(), size_t(2)}),
            1);

        REQUIRE(xt::all(xt::equal(is_plastic, sys.layerIsPlastic())));
    }

    SECTION("Force from drive")
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

        xt::xtensor<bool, 2> drive = xt::zeros<bool>({nlayer, size_t(2)});
        xt::xtensor<double, 2> u_target = xt::zeros<double>({nlayer, size_t(2)});
        drive(2, 0) = true;
        drive(4, 0) = true;
        u_target(2, 0) = 1.0;
        u_target(4, 0) = 2.0;

        FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System sys(
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
            FrictionQPotFEM::epsy_initelastic_toquad(xt::ones<double>({nplas, size_t(1)})),
            1,
            1,
            1,
            0,
            drive,
            1);

        sys.layerSetTargetUbar(u_target);

        xt::xtensor<double, 2> f = xt::zeros<double>({stitch.nnode(), stitch.ndim()});

        {
            xt::xtensor<size_t, 1> c0 = xt::xtensor<size_t, 1>{mesh.nodesBottomLeftCorner()};
            c0 = stitch.nodeset(c0, 2);

            xt::xtensor<size_t, 1> c1 = xt::xtensor<size_t, 1>{mesh.nodesBottomRightCorner()};
            c1 = stitch.nodeset(c1, 2);

            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesBottomOpenEdge(), 2);

            xt::view(f, xt::keep(c0), 0) -= 0.25;
            xt::view(f, xt::keep(c1), 0) -= 0.25;
            xt::view(f, xt::keep(e), 0) -= 0.5;
        }

        {
            xt::xtensor<size_t, 1> c0 = xt::xtensor<size_t, 1>{mesh.nodesTopLeftCorner()};
            c0 = stitch.nodeset(c0, 2);

            xt::xtensor<size_t, 1> c1 = xt::xtensor<size_t, 1>{mesh.nodesTopRightCorner()};
            c1 = stitch.nodeset(c1, 2);

            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesTopOpenEdge(), 2);

            xt::view(f, xt::keep(c0), 0) -= 0.25;
            xt::view(f, xt::keep(c1), 0) -= 0.25;
            xt::view(f, xt::keep(e), 0) -= 0.5;
        }

        {
            xt::xtensor<size_t, 1> c0 = xt::xtensor<size_t, 1>{mesh.nodesBottomLeftCorner()};
            c0 = stitch.nodeset(c0, 4);

            xt::xtensor<size_t, 1> c1 = xt::xtensor<size_t, 1>{mesh.nodesBottomRightCorner()};
            c1 = stitch.nodeset(c1, 4);

            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesBottomOpenEdge(), 4);

            xt::view(f, xt::keep(c0), 0) -= 0.5;
            xt::view(f, xt::keep(c1), 0) -= 0.5;
            xt::view(f, xt::keep(e), 0) -= 1.0;
        }

        {
            xt::xtensor<size_t, 1> c0 = xt::xtensor<size_t, 1>{mesh.nodesTopLeftCorner()};
            c0 = stitch.nodeset(c0, 4);

            xt::xtensor<size_t, 1> c1 = xt::xtensor<size_t, 1>{mesh.nodesTopRightCorner()};
            c1 = stitch.nodeset(c1, 4);

            xt::xtensor<size_t, 1> e = stitch.nodeset(mesh.nodesTopOpenEdge(), 4);

            xt::view(f, xt::keep(c0), 0) -= 0.5;
            xt::view(f, xt::keep(c1), 0) -= 0.5;
            xt::view(f, xt::keep(e), 0) -= 1.0;
        }

        REQUIRE(xt::allclose(f, sys.fdrive()));
    }

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
        drive(2, 0) = true;
        drive(4, 0) = true;
        u_target(2, 0) = 1.0;
        u_target(4, 0) = 2.0;

        FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System sys(
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
            FrictionQPotFEM::epsy_initelastic_toquad(100.0 * xt::ones<double>({nplas, size_t(1)})),
            1, // dt
            1,
            1,
            0,
            drive,
            1); // k_drive

        sys.layerSetTargetUbar(u_target);
        sys.layerSetUbar(u_target, drive);

        xt::xtensor<double, 2> u = xt::zeros<double>({stitch.nnode(), stitch.ndim()});
        auto n2 = stitch.nodemap(2);
        auto n4 = stitch.nodemap(4);
        xt::view(u, xt::keep(n2), 0) = 1.0;
        xt::view(u, xt::keep(n4), 0) = 2.0;

        REQUIRE(xt::allclose(sys.u(), u));
        REQUIRE(xt::allclose(sys.fdrive(), 0.0));
        REQUIRE(xt::allclose(sys.layerFdrive(), 0.0));
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

        auto left = stitch.nodeset(
            {layer_elas.nodesLeftOpenEdge(),
             layer_plas.nodesLeftEdge(),
             layer_elas.nodesLeftOpenEdge()});

        auto right = stitch.nodeset(
            {layer_elas.nodesRightOpenEdge(),
             layer_plas.nodesRightEdge(),
             layer_elas.nodesRightOpenEdge()});

        auto bottom = stitch.nodeset(layer_elas.nodesBottomEdge(), 0);
        auto top = stitch.nodeset(layer_elas.nodesTopEdge(), 2);

        auto coor = stitch.coor();
        auto conn = stitch.conn();
        auto dofs = stitch.dofs();

        // periodicity
        xt::view(dofs, xt::keep(right), xt::all()) = xt::view(dofs, xt::keep(left), xt::all());
        dofs(top(0), 0) = dofs(top.periodic(-1), 0);
        dofs(bottom(0), 0) = dofs(bottom.periodic(-1), 0);
        dofs = GooseFEM::Mesh::renumber(dofs);

        // fixed bottom (x + y) and top (x)
        xt::xtensor<size_t, 1> iip = xt::concatenate(xt::xtuple(
            xt::view(dofs, xt::keep(bottom), 0),
            xt::view(dofs, xt::keep(bottom), 1),
            xt::view(dofs, xt::keep(top), 1)));
        iip = xt::unique(iip);

        xt::xtensor<bool, 1> is_plastic = {false, true, false};
        auto elas = xt::concatenate(xt::xtuple(stitch.elemmap(0), stitch.elemmap(2)));
        auto plas = stitch.elemmap(1);
        size_t nelas = elas.size();
        size_t nplas = plas.size();
        xt::xtensor<double, 1> Ielas = xt::ones<double>({nelas});
        xt::xtensor<double, 1> Iplas = xt::ones<double>({nplas});
        xt::xtensor<double, 2> epsy = 0.01 * xt::ones<double>({plas.size(), size_t(100)});
        epsy = xt::cumsum(epsy, 1);

        xt::xtensor<bool, 2> drive = xt::zeros<bool>({3, 2});
        xt::xtensor<double, 2> u_target = xt::zeros<double>({3, 2});
        drive(0, 0) = true;
        drive(2, 0) = true;
        u_target(2, 0) = 0.1;

        FrictionQPotFEM::UniformMultiLayerIndividualDrive2d::System sys(
            coor,
            conn,
            dofs,
            iip,
            stitch.elemmap(),
            stitch.nodemap(),
            is_plastic,
            FrictionQPotFEM::moduli_toquad(xt::eval(10 * xt::ones<double>({nelas}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(1 * xt::ones<double>({nelas}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(10 * xt::ones<double>({nplas}))),
            FrictionQPotFEM::moduli_toquad(xt::eval(1 * xt::ones<double>({nplas}))),
            FrictionQPotFEM::epsy_initelastic_toquad(epsy),
            0.1, // dt
            1, // rho
            0.01, // alpha
            0,
            drive,
            1.0); // k_drive

        sys.layerSetTargetUbar(u_target);
        auto niter = sys.minimise();
        REQUIRE(niter >= 0);

        u_target(1, 0) = 0.05; // expected result (layer not prescribed)

        REQUIRE(xt::allclose(u_target, sys.layerUbar(), 5e-2, 5e-3));
        REQUIRE(xt::all(xt::equal(sys.plastic().i(), 5)));
    }
}
