
#include <catch2/catch.hpp>
#include <xtensor/xrandom.hpp>
#include <FrictionQPotFEM/UniformSingleLayer2d.h>

#define ISCLOSE(a, b) REQUIRE_THAT((a), Catch::WithinAbs((b), 1e-8));

TEST_CASE("FrictionQPotFEM::UniformSingleLayer2d", "UniformSingleLayer2d.h")
{

SECTION("System::addSimpleShearEventDriven")
{
    GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

    FrictionQPotFEM::UniformSingleLayer2d::System sys(
        mesh.coor(),
        mesh.conn(),
        mesh.dofs(),
        xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
        xt::xtensor<size_t, 1>{},
        xt::xtensor<size_t, 1>{0});

    sys.setMassMatrix(xt::ones<double>({1}));
    sys.setDampingMatrix(xt::ones<double>({1}));

    sys.setElastic(
        xt::xtensor<double, 1>{},
        xt::xtensor<double, 1>{});

    sys.setPlastic(
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

    sys.setDt(1.0);

    REQUIRE(sys.isHomogeneousElastic());

    double delta_eps = 1e-3;
    auto dV = sys.AsTensor<2>(sys.dV());
    xt::xtensor<double, 2> Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 0.0);

    sys.addSimpleShearEventDriven(delta_eps, false, +1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 1.0 - delta_eps / 2.0);

    sys.addSimpleShearEventDriven(delta_eps, true, +1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 1.0 + delta_eps / 2.0);

    sys.addSimpleShearEventDriven(delta_eps, false, +1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 2.0 - delta_eps / 2.0);

    sys.addSimpleShearEventDriven(delta_eps, true, +1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 2.0 + delta_eps / 2.0);

    sys.addSimpleShearEventDriven(delta_eps, false, -1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 2.0 + delta_eps / 2.0);

    sys.addSimpleShearEventDriven(delta_eps, true, -1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 2.0 - delta_eps / 2.0);

    sys.addSimpleShearEventDriven(delta_eps, false, -1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 1.0 + delta_eps / 2.0);

    sys.addSimpleShearEventDriven(delta_eps, true, -1.0);

    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 1.0 - delta_eps / 2.0);
}

SECTION("System::addSimpleShearToFixedStress")
{
    GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

    FrictionQPotFEM::UniformSingleLayer2d::System sys(
        mesh.coor(),
        mesh.conn(),
        mesh.dofs(),
        xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
        xt::xtensor<size_t, 1>{},
        xt::xtensor<size_t, 1>{0});

    sys.setMassMatrix(xt::ones<double>({1}));
    sys.setDampingMatrix(xt::ones<double>({1}));

    sys.setElastic(
        xt::xtensor<double, 1>{},
        xt::xtensor<double, 1>{});

    sys.setPlastic(
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

    sys.setDt(1.0);

    REQUIRE(sys.isHomogeneousElastic());

    auto dV = sys.AsTensor<2>(sys.dV());
    xt::xtensor<double, 2> Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    xt::xtensor<double, 2> Sigbar = xt::average(sys.Sig(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), 0.0);
    ISCLOSE(GM::Sigd(Sigbar)(), 0.0);

    double target_stress = 1.0;
    sys.addSimpleShearToFixedStress(target_stress);
    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    Sigbar = xt::average(sys.Sig(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), target_stress / 2.0);
    ISCLOSE(GM::Sigd(Sigbar)(), target_stress);

    target_stress = 1.5;
    sys.addSimpleShearToFixedStress(target_stress);
    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    Sigbar = xt::average(sys.Sig(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), target_stress / 2.0);
    ISCLOSE(GM::Sigd(Sigbar)(), target_stress);

    target_stress = 0.5;
    sys.addSimpleShearToFixedStress(target_stress);
    Epsbar = xt::average(sys.Eps(), dV, {0, 1});
    Sigbar = xt::average(sys.Sig(), dV, {0, 1});
    ISCLOSE(GM::Epsd(Epsbar)(), target_stress / 2.0);
    ISCLOSE(GM::Sigd(Sigbar)(), target_stress);
}

SECTION("System::triggerElementWithLocalSimpleShear")
{
    GooseFEM::Mesh::Quad4::Regular mesh(3, 3);

    auto dofs = mesh.dofs();
    auto top = mesh.nodesTopEdge();
    auto bottom = mesh.nodesBottomEdge();
    size_t nfix = top.size();
    xt::xtensor<size_t, 1> iip = xt::empty<size_t>({2 * mesh.ndim() * nfix});
    xt::view(iip, xt::range(0 * nfix, 1 * nfix)) = xt::view(dofs, xt::keep(bottom), 0);
    xt::view(iip, xt::range(1 * nfix, 2 * nfix)) = xt::view(dofs, xt::keep(bottom), 1);
    xt::view(iip, xt::range(2 * nfix, 3 * nfix)) = xt::view(dofs, xt::keep(top), 0);
    xt::view(iip, xt::range(3 * nfix, 4 * nfix)) = xt::view(dofs, xt::keep(top), 1);

    FrictionQPotFEM::UniformSingleLayer2d::System sys(
        mesh.coor(),
        mesh.conn(),
        dofs,
        iip,
        xt::xtensor<size_t, 1>{0, 1, 2, 3, 5, 6, 7, 8},
        xt::xtensor<size_t, 1>{4});

    sys.setMassMatrix(xt::ones<double>({mesh.nelem()}));
    sys.setDampingMatrix(xt::ones<double>({mesh.nelem()}));

    sys.setElastic(
        1.0 * xt::ones<double>({8}),
        1.0 * xt::ones<double>({8}));

    sys.setPlastic(
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

    sys.setDt(1.0);

    REQUIRE(sys.isHomogeneousElastic());

    double delta_eps = 1e-3;
    sys.triggerElementWithLocalSimpleShear(delta_eps, 0);

    auto eps = GM::Epsd(sys.Eps());
    auto plastic = sys.plastic();
    auto eps_p = xt::view(eps, xt::keep(plastic), xt::all());

    REQUIRE(xt::allclose(eps_p, 1.0 + delta_eps / 2.0));
}

SECTION("System::plastic_*")
{
    GooseFEM::Mesh::Quad4::Regular mesh(1, 1);

    FrictionQPotFEM::UniformSingleLayer2d::System full(
        mesh.coor(),
        mesh.conn(),
        mesh.dofs(),
        xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
        xt::xtensor<size_t, 1>{},
        xt::xtensor<size_t, 1>{0});

    full.setMassMatrix(xt::ones<double>({1}));
    full.setDampingMatrix(xt::ones<double>({1}));

    full.setElastic(
        xt::xtensor<double, 1>{},
        xt::xtensor<double, 1>{});

    full.setPlastic(
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

    full.setDt(1.0);

    FrictionQPotFEM::UniformSingleLayer2d::System reduced(
        mesh.coor(),
        mesh.conn(),
        mesh.dofs(),
        xt::arange<size_t>(mesh.nnode() * mesh.ndim()),
        xt::xtensor<size_t, 1>{},
        xt::xtensor<size_t, 1>{0});

    reduced.setMassMatrix(xt::ones<double>({1}));
    reduced.setDampingMatrix(xt::ones<double>({1}));

    reduced.setElastic(
        xt::xtensor<double, 1>{},
        xt::xtensor<double, 1>{});

    reduced.setPlastic(
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 1>{1.0},
        xt::xtensor<double, 2>{{1.0, 2.0, 3.0, 4.0}});

    reduced.setDt(1.0);

    double delta_eps = 1e-3;
    full.addSimpleShearEventDriven(delta_eps, false);
    reduced.addSimpleShearEventDriven(delta_eps, false);

    REQUIRE(xt::allclose(full.Eps(), reduced.Eps()));
    REQUIRE(xt::allclose(full.Eps(), full.plastic_Eps()));
    REQUIRE(xt::allclose(reduced.Eps(), reduced.plastic_Eps()));

    REQUIRE(xt::allclose(full.Sig(), reduced.Sig()));
    REQUIRE(xt::allclose(full.Sig(), full.plastic_Sig()));
    REQUIRE(xt::allclose(reduced.Sig(), reduced.plastic_Sig()));

    REQUIRE(xt::allclose(full.plastic_CurrentYieldLeft(), reduced.plastic_CurrentYieldLeft()));
    REQUIRE(xt::allclose(full.plastic_CurrentYieldRight(), reduced.plastic_CurrentYieldRight()));
    REQUIRE(xt::all(xt::equal(full.plastic_CurrentIndex(), reduced.plastic_CurrentIndex())));
}

SECTION("System vs. HybridSystem")
{
    // Stress/strain response computed using previous versions

    xt::xtensor<double, 1> check_Eps = {0, 2e-06, 4e-06, 6e-06, 8e-06, 1e-05, 1.2e-05, 1.4e-05,
        1.6e-05, 1.8e-05, 2e-05, 2.2e-05, 2.4e-05, 2.6e-05, 2.8e-05, 3e-05, 3.2e-05, 3.4e-05,
        3.6e-05, 3.8e-05, 4e-05, 4.2e-05, 4.4e-05, 4.6e-05, 4.8e-05, 5e-05, 5.2e-05, 5.4e-05,
        5.6e-05, 5.8e-05, 6e-05, 6.2e-05, 6.4e-05, 6.6e-05, 6.8e-05, 7e-05, 7.2e-05, 7.4e-05,
        7.6e-05, 7.8e-05, 8e-05, 8.2e-05, 8.4e-05, 8.6e-05, 8.8e-05, 9e-05, 9.2e-05, 9.4e-05,
        9.6e-05, 9.8e-05, 0.0001, 0.000102, 0.000104, 0.000106, 0.000108, 0.00011, 0.000112,
        0.000114, 0.000116, 0.000118, 0.00012, 0.000122, 0.000124, 0.000126, 0.000128, 0.00013,
        0.000132, 0.000134, 0.000136, 0.000138, 0.00014, 0.000142, 0.000144, 0.000146, 0.000148,
        0.00015, 0.000152, 0.000154, 0.000156, 0.000158, 0.00016, 0.000162, 0.000164, 0.000166,
        0.000168, 0.00017, 0.000172, 0.000174, 0.000176, 0.000178, 0.00018, 0.000182, 0.000184,
        0.000186, 0.000188, 0.00019, 0.000192, 0.000194, 0.000196, 0.000198, 0.0002, 0.000202,
        0.000204, 0.000206, 0.000208, 0.00021, 0.000212, 0.000214, 0.000216, 0.000218, 0.00022,
        0.000222, 0.000224, 0.000226, 0.000228, 0.00023, 0.000232, 0.000234, 0.000236, 0.000238,
        0.00024, 0.000242, 0.000244, 0.000246, 0.000248, 0.00025, 0.000252, 0.000254, 0.000256,
        0.000258, 0.00026, 0.000262, 0.000264, 0.000266, 0.000268, 0.00027, 0.000272, 0.000274,
        0.000276, 0.000278, 0.00028, 0.000282, 0.000284, 0.000286, 0.000288, 0.00029, 0.000292,
        0.000294, 0.000296, 0.000298, 0.0003, 0.000302, 0.000304, 0.000306, 0.000308, 0.00031,
        0.000312, 0.000314, 0.000316, 0.000318, 0.00032, 0.000322, 0.000324, 0.000326, 0.000328,
        0.00033, 0.000332, 0.000334, 0.000336, 0.000338, 0.00034, 0.000342, 0.000344, 0.000346,
        0.000348, 0.00035, 0.000352, 0.000354, 0.000356, 0.000358, 0.00036, 0.000362, 0.000364,
        0.000366, 0.000368, 0.00037, 0.000372, 0.000374, 0.000376, 0.000378, 0.00038, 0.000382,
        0.000384, 0.000386, 0.000388, 0.00039, 0.000392, 0.000394, 0.000396, 0.000398, 0.0004,
        0.000402, 0.000404, 0.000406, 0.000408, 0.00041, 0.000412, 0.000414, 0.000416, 0.000418,
        0.00042, 0.000422, 0.000424, 0.000426, 0.000428, 0.00043, 0.000432, 0.000434, 0.000436,
        0.000438, 0.00044, 0.000442, 0.000444, 0.000446, 0.000448, 0.00045, 0.000452, 0.000454,
        0.000456, 0.000458, 0.00046, 0.000462, 0.000464, 0.000466, 0.000468, 0.00047, 0.000472,
        0.000474, 0.000476, 0.000478, 0.00048, 0.000482, 0.000484, 0.000486, 0.000488, 0.00049,
        0.000492, 0.000494, 0.000496, 0.000498, 0.0005, 0.000502, 0.000504, 0.000506, 0.000508,
        0.00051, 0.000512, 0.000514, 0.000516, 0.000518, 0.00052, 0.000522, 0.000524, 0.000526,
        0.000528, 0.00053, 0.000532, 0.000534, 0.000536, 0.000538, 0.00054, 0.000542, 0.000544,
        0.000546, 0.000548, 0.00055, 0.000552, 0.000554, 0.000556, 0.000558, 0.00056, 0.000562,
        0.000564, 0.000566, 0.000568, 0.00057, 0.000572, 0.000574, 0.000576, 0.000578, 0.00058,
        0.000582, 0.000584, 0.000586, 0.000588, 0.00059, 0.000592, 0.000594, 0.000596, 0.000598,
        0.0006, 0.000602, 0.000604, 0.000606, 0.000608, 0.00061, 0.000612, 0.000614, 0.000616,
        0.000618, 0.00062, 0.000622, 0.000624, 0.000626, 0.000628, 0.00063, 0.000632, 0.000634,
        0.000636, 0.000638, 0.00064, 0.000642, 0.000644, 0.000646, 0.000648, 0.00065, 0.000652,
        0.000654, 0.000656, 0.000658, 0.00066, 0.000662, 0.000664, 0.000666, 0.000668, 0.00067,
        0.000672, 0.000674, 0.000676, 0.000678, 0.00068, 0.000682, 0.000684, 0.000686, 0.000688,
        0.00069, 0.000692, 0.000694, 0.000696, 0.000698, 0.0007, 0.000702, 0.000704, 0.000706,
        0.000708, 0.00071, 0.000712, 0.000714, 0.000716, 0.000718, 0.00072, 0.000722, 0.000724,
        0.000726, 0.000728, 0.00073, 0.000732, 0.000734, 0.000736, 0.000738, 0.00074, 0.000742,
        0.000744, 0.000746, 0.000748, 0.00075, 0.000752, 0.000754, 0.000756, 0.000758, 0.00076,
        0.000762, 0.000764, 0.000766, 0.000768, 0.00077, 0.000772, 0.000774, 0.000776, 0.000778,
        0.00078, 0.000782, 0.000784, 0.000786, 0.000788, 0.00079, 0.000792, 0.000794, 0.000796,
        0.000798, 0.0008, 0.000802, 0.000804, 0.000806, 0.000808, 0.00081, 0.000812, 0.000814,
        0.000816, 0.000818, 0.00082, 0.000822, 0.000824, 0.000826, 0.000828, 0.00083, 0.000832,
        0.000834, 0.000836, 0.000838, 0.00084, 0.000842, 0.000844, 0.000846, 0.000848, 0.00085,
        0.000852, 0.000854, 0.000856, 0.000858, 0.00086, 0.000862, 0.000864, 0.000866, 0.000868,
        0.00087, 0.000872, 0.000874, 0.000876, 0.000878, 0.00088, 0.000882, 0.000884, 0.000886,
        0.000888, 0.00089, 0.000892, 0.000894, 0.000896, 0.000898, 0.0009, 0.000902, 0.000904,
        0.000906, 0.000908, 0.00091, 0.000912, 0.000914, 0.000916, 0.000918, 0.00092, 0.000922,
        0.000924, 0.000926, 0.000928, 0.00093, 0.000932, 0.000934, 0.000936, 0.000938, 0.00094,
        0.000942, 0.000944, 0.000946, 0.000948, 0.00095, 0.000952, 0.000954, 0.000956, 0.000958,
        0.00096, 0.000962, 0.000964, 0.000966, 0.000968, 0.00097, 0.000972, 0.000974, 0.000976,
        0.000978, 0.00098, 0.000982, 0.000984, 0.000986, 0.000988, 0.00099, 0.000992, 0.000994,
        0.000996, 0.000998, 0.001, 0.001002, 0.001004, 0.001006, 0.001008, 0.00101, 0.001012,
        0.001014, 0.001016, 0.001018, 0.00102, 0.001022, 0.001024, 0.001026, 0.001028, 0.00103,
        0.001032, 0.001034, 0.001036, 0.001038, 0.00104, 0.001042, 0.001044, 0.001046, 0.001048,
        0.00105, 0.001052, 0.001054, 0.001056, 0.001058, 0.00106, 0.001062, 0.001064, 0.001066,
        0.001068, 0.00107, 0.001072, 0.001074, 0.001076, 0.001078, 0.00108, 0.001082, 0.001084,
        0.001086, 0.001088, 0.00109, 0.001092, 0.001094, 0.001096, 0.001098, 0.0011, 0.001102,
        0.001104, 0.001106, 0.001108, 0.00111, 0.001112, 0.001114, 0.001116, 0.001118, 0.00112,
        0.001122, 0.001124, 0.001126, 0.001128, 0.00113, 0.001132, 0.001134, 0.001136, 0.001138,
        0.00114, 0.001142, 0.001144, 0.001146, 0.001148, 0.00115, 0.001152, 0.001154, 0.001156,
        0.001158, 0.00116, 0.001162, 0.001164, 0.001166, 0.001168, 0.00117, 0.001172, 0.001174,
        0.001176, 0.001178, 0.00118, 0.001182, 0.001184, 0.001186, 0.001188, 0.00119, 0.001192,
        0.001194, 0.001196, 0.001198, 0.0012, 0.001202, 0.001204, 0.001206, 0.001208, 0.00121,
        0.001212, 0.001214, 0.001216, 0.001218, 0.00122, 0.001222, 0.001224, 0.001226, 0.001228,
        0.00123, 0.001232, 0.001234, 0.001236, 0.001238, 0.00124, 0.001242, 0.001244, 0.001246,
        0.001248, 0.00125, 0.001252, 0.001254, 0.001256, 0.001258, 0.00126, 0.001262, 0.001264,
        0.001266, 0.001268, 0.00127, 0.001272, 0.001274, 0.001276, 0.001278, 0.00128, 0.001282,
        0.001284, 0.001286, 0.001288, 0.00129, 0.001292, 0.001294, 0.001296, 0.001298, 0.0013,
        0.001302, 0.001304, 0.001306, 0.001308, 0.00131, 0.001312, 0.001314, 0.001316, 0.001318,
        0.00132, 0.001322, 0.001324, 0.001326, 0.001328, 0.00133, 0.001332, 0.001334, 0.001336,
        0.001338, 0.00134, 0.001342, 0.001344, 0.001346, 0.001348, 0.00135, 0.001352, 0.001354,
        0.001356, 0.001358, 0.00136, 0.001362, 0.001364, 0.001366, 0.001368, 0.00137, 0.001372,
        0.001374, 0.001376, 0.001378, 0.00138, 0.001382, 0.001384, 0.001386, 0.001388, 0.00139,
        0.001392, 0.001394, 0.001396, 0.001398, 0.0014, 0.001402, 0.001404, 0.001406, 0.001408,
        0.00141, 0.001412, 0.001414, 0.001416, 0.001418, 0.00142, 0.001422, 0.001424, 0.001426,
        0.001428, 0.00143, 0.001432, 0.001434, 0.001436, 0.001438, 0.00144, 0.001442, 0.001444,
        0.001446, 0.001448, 0.00145, 0.001452, 0.001454, 0.001456, 0.001458, 0.00146, 0.001462,
        0.001464, 0.001466, 0.001468, 0.00147, 0.001472, 0.001474, 0.001476, 0.001478, 0.00148,
        0.001482, 0.001484, 0.001486, 0.001488, 0.00149, 0.001492, 0.001494, 0.001496, 0.001498,
        0.0015, 0.001502, 0.001504, 0.001506, 0.001508, 0.00151, 0.001512, 0.001514, 0.001516,
        0.001518, 0.00152, 0.001522, 0.001524, 0.001526, 0.001528, 0.00153, 0.001532, 0.001534,
        0.001536, 0.001538, 0.00154, 0.001542, 0.001544, 0.001546, 0.001548, 0.00155, 0.001552,
        0.001554, 0.001556, 0.001558, 0.00156, 0.001562, 0.001564, 0.001566, 0.001568, 0.00157,
        0.001572, 0.001574, 0.001576, 0.001578, 0.00158, 0.001582, 0.001584, 0.001586, 0.001588,
        0.00159, 0.001592, 0.001594, 0.001596, 0.001598, 0.0016, 0.001602, 0.001604, 0.001606,
        0.001608, 0.00161, 0.001612, 0.001614, 0.001616, 0.001618, 0.00162, 0.001622, 0.001624,
        0.001626, 0.001628, 0.00163, 0.001632, 0.001634, 0.001636, 0.001638, 0.00164, 0.001642,
        0.001644, 0.001646, 0.001648, 0.00165, 0.001652, 0.001654, 0.001656, 0.001658, 0.00166,
        0.001662, 0.001664, 0.001666, 0.001668, 0.00167, 0.001672, 0.001674, 0.001676, 0.001678,
        0.00168, 0.001682, 0.001684, 0.001686, 0.001688, 0.00169, 0.001692, 0.001694, 0.001696,
        0.001698, 0.0017, 0.001702, 0.001704, 0.001706, 0.001708, 0.00171, 0.001712, 0.001714,
        0.001716, 0.001718, 0.00172, 0.001722, 0.001724, 0.001726, 0.001728, 0.00173, 0.001732,
        0.001734, 0.001736, 0.001738, 0.00174, 0.001742, 0.001744, 0.001746, 0.001748, 0.00175,
        0.001752, 0.001754, 0.001756, 0.001758, 0.00176, 0.001762, 0.001764, 0.001766, 0.001768,
        0.00177, 0.001772, 0.001774, 0.001776, 0.001778, 0.00178, 0.001782, 0.001784, 0.001786,
        0.001788, 0.00179, 0.001792, 0.001794, 0.001796, 0.001798, 0.0018, 0.001802, 0.001804,
        0.001806, 0.001808, 0.00181, 0.001812, 0.001814, 0.001816, 0.001818, 0.00182, 0.001822,
        0.001824, 0.001826, 0.001828, 0.00183, 0.001832, 0.001834, 0.001836, 0.001838, 0.00184,
        0.001842, 0.001844, 0.001846, 0.001848, 0.00185, 0.001852, 0.001854, 0.001856, 0.001858,
        0.00186, 0.001862, 0.001864, 0.001866, 0.001868, 0.00187, 0.001872, 0.001874, 0.001876,
        0.001878, 0.00188, 0.001882, 0.001884, 0.001886, 0.001888, 0.00189, 0.001892, 0.001894,
        0.001896, 0.001898, 0.0019, 0.001902, 0.001904, 0.001906, 0.001908, 0.00191, 0.001912,
        0.001914, 0.001916, 0.001918, 0.00192, 0.001922, 0.001924, 0.001926, 0.001928, 0.00193,
        0.001932, 0.001934, 0.001936, 0.001938, 0.00194, 0.001942, 0.001944, 0.001946, 0.001948,
        0.00195, 0.001952, 0.001954, 0.001956, 0.001958, 0.00196, 0.001962, 0.001964, 0.001966,
        0.001968, 0.00197, 0.001972, 0.001974, 0.001976, 0.001978, 0.00198, 0.001982, 0.001984,
        0.001986, 0.001988, 0.00199, 0.001992, 0.001994, 0.001996, 0.001998, 0.002};

    xt::xtensor<double, 1> check_Sig = {0, 2e-06, 4e-06, 6e-06, 8e-06, 1e-05, 1.2e-05, 1.4e-05,
        1.6e-05, 1.8e-05, 2e-05, 2.2e-05, 2.4e-05, 2.6e-05, 2.8e-05, 3e-05, 3.2e-05, 3.4e-05,
        3.6e-05, 3.8e-05, 4e-05, 4.2e-05, 4.4e-05, 4.6e-05, 4.8e-05, 5e-05, 5.2e-05, 5.4e-05,
        5.6e-05, 5.8e-05, 6e-05, 6.2e-05, 6.4e-05, 6.6e-05, 6.8e-05, 7e-05, 7.2e-05, 7.4e-05,
        7.6e-05, 7.8e-05, 8e-05, 8.2e-05, 8.4e-05, 8.6e-05, 8.8e-05, 9e-05, 9.2e-05, 9.4e-05,
        9.6e-05, 9.8e-05, 0.0001, 0.000102, 0.000104, 0.000106, 0.000108, 0.00011, 0.000112,
        0.000114, 0.000116, 0.000118, 0.00012, 0.000122, 0.000124, 0.000126, 0.000128, 0.00013,
        0.000132, 0.000134, 0.000136, 0.000138, 0.00014, 0.000142, 0.000144, 0.000146, 0.000148,
        0.00015, 0.000152, 0.000154, 0.000156, 0.000158, 0.00016, 0.000162, 0.000164, 0.000166,
        0.000168, 0.00017, 0.000172, 0.000174, 0.000176, 0.000178, 0.00018, 0.000182, 0.000184,
        0.000186, 0.000188, 0.00019, 0.000192, 0.000194, 0.000196, 0.000198, 0.0002, 0.000202,
        0.000204, 0.000206, 0.000208, 0.00021, 0.000212, 0.000214, 0.000216, 0.000218, 0.00022,
        0.000222, 0.000224, 0.000226, 0.000228, 0.00023, 0.000232, 0.000234, 0.000236, 0.000238,
        0.00024, 0.000242, 0.000244, 0.000246, 0.000248, 0.00025, 0.000252, 0.000254, 0.000256,
        0.000258, 0.00026, 0.000262, 0.000264, 0.000266, 0.000268, 0.00027, 0.000227556,
        0.000229556, 0.000231556, 0.000233556, 0.000235556, 0.000237556, 0.000239556, 0.000241556,
        0.000243555, 0.000245555, 0.000247555, 0.000249555, 0.000211511, 0.000213511, 0.000215511,
        0.000217511, 0.000219511, 0.000221511, 0.000223511, 0.000225511, 0.00022751, 0.00022951,
        0.00023151, 0.00023351, 0.00023551, 0.00023751, 0.00023951, 0.00024151, 0.00024351,
        0.00024551, 0.00024751, 0.00024951, 0.00025151, 0.00025351, 0.00025551, 0.000257509,
        0.000259509, 0.000261509, 0.000263509, 0.000265509, 0.000267509, 0.000269509, 0.000271509,
        0.000273509, 0.000275509, 0.000277509, 0.000279509, 0.000281509, 0.000283509, 0.000285509,
        0.000287508, 0.000289508, 0.000291508, 0.000293508, 0.000295508, 0.000297508, 0.000299508,
        0.000301508, 0.000303508, 0.000305508, 0.000307508, 0.000309508, 0.000311508, 0.000313508,
        0.000315508, 0.000317508, 0.000319508, 0.000321507, 0.000323507, 0.000325507, 0.000327507,
        0.000329507, 0.000331507, 0.000333507, 0.000335507, 0.000337507, 0.000339507, 0.000341507,
        0.000343507, 0.000345507, 0.000329346, 0.000331346, 0.000333346, 0.000335346, 0.000337346,
        0.000339346, 0.000248973, 0.000250973, 0.000252973, 0.000254973, 0.000256973, 0.000258973,
        0.000260973, 0.000262973, 0.000264973, 0.000266973, 0.000268973, 0.000270973, 0.000272973,
        0.000274973, 0.000276973, 0.0002165, 0.00019006, 0.00019206, 0.00019406, 0.00019606,
        0.00019806, 0.00020006, 0.00020206, 0.00020406, 0.00020606, 0.000208059, 0.000210059,
        0.000212059, 0.000214059, 0.000216059, 0.000218059, 0.000220059, 0.000222059, 0.000224059,
        0.000226059, 0.000228059, 0.000230059, 0.000232059, 0.000234059, 0.000236059, 0.000238059,
        0.000240059, 0.000242059, 0.000244059, 0.000246059, 0.000248059, 0.000250059, 0.000252059,
        0.000254059, 0.000256059, 0.000258059, 0.000260059, 0.000262059, 0.000264059, 0.000266059,
        0.000268059, 0.000270059, 0.000272059, 0.000274059, 0.000276059, 0.000278059, 0.000280059,
        0.000282059, 0.000284059, 0.000286059, 0.000239775, 0.000241775, 0.000243775, 0.000245775,
        0.000247775, 0.000249775, 0.000251775, 0.000253775, 0.000255775, 0.000257775, 0.000259775,
        0.000261775, 0.000263775, 0.000265775, 0.000267775, 0.000269775, 0.000271775, 0.000273775,
        0.000275775, 0.000277775, 0.000279775, 0.000281775, 0.000283775, 0.000285775, 0.000231147,
        0.000233147, 0.000235147, 0.000237147, 0.000239147, 0.000241147, 0.000243147, 0.000245147,
        0.000247147, 0.000249147, 0.000251147, 0.000253147, 0.000255147, 0.000257147, 0.000259147,
        0.000261147, 0.000263147, 0.000265147, 0.000267147, 0.000269147, 0.000271147, 0.000273147,
        0.000275147, 0.000277147, 0.000279147, 0.000281147, 0.000283147, 0.000285147, 0.000287147,
        0.000289147, 0.000291147, 0.000293147, 0.000295147, 0.000297147, 0.000299147, 0.000301147,
        0.000303147, 0.000305147, 0.000307147, 0.000309147, 0.000311147, 0.000313147, 0.000315147,
        0.000317147, 0.000319147, 0.000321147, 0.000323147, 0.000325147, 0.000327147, 0.000329147,
        0.000331147, 0.000333147, 0.000335147, 0.000337147, 0.000339147, 0.000341147, 0.000343147,
        0.000345147, 0.000347147, 0.000227944, 0.000229944, 0.000231944, 0.000233944, 0.000235944,
        0.000237944, 0.000149286, 0.000151286, 0.000153286, 0.000155286, 0.000157286, 0.000159286,
        0.000161286, 0.000163286, 0.000165286, 0.000167286, 0.000169286, 0.000171286, 0.000173286,
        0.000175286, 0.000177286, 0.000179286, 0.000181286, 0.000183286, 0.000185286, 0.000187286,
        0.000189286, 0.000191286, 0.000193286, 0.000195286, 0.000197286, 0.000199286, 0.000201286,
        0.000203286, 0.000205286, 0.000207286, 0.000209286, 0.000211286, 0.000213286, 0.000215286,
        0.000217286, 0.000148851, 0.000150851, 0.000152851, 0.000154851, 0.000156851, 0.000158851,
        0.000110681, 0.000112681, 0.000114681, 0.000116681, 0.000118681, 0.000120681, 0.000122681,
        0.000124681, 0.000126681, 0.000128681, 0.000130681, 0.000132681, 0.000134681, 0.000136681,
        9.8159e-05, 0.000100159, 0.000102159, 0.000104159, 0.000106159, 0.000108159, 0.000110159,
        0.000112159, 0.000114159, 0.000116159, 0.000118159, 0.000120159, 0.000122159, 0.000124159,
        0.000126159, 0.000128159, 0.000130159, 0.000132159, 0.000134159, 0.000136159, 0.000138159,
        0.000140159, 0.000142159, 0.000144159, 0.000146159, 0.000148159, 0.000150159, 0.000152159,
        0.000154159, 0.000156159, 0.000158159, 0.000160159, 0.000162159, 0.000164159, 0.000166159,
        0.000168159, 0.000170159, 0.000172159, 0.000174159, 0.000176159, 0.000178159, 0.000180159,
        0.000182159, 0.000184159, 0.000186159, 0.000188159, 0.000190159, 0.000192159, 0.000194159,
        0.000196159, 0.000198159, 0.000200159, 0.000202159, 0.000204159, 0.000206159, 0.000197065,
        0.000199065, 0.000201065, 0.000203065, 0.000205065, 0.000207065, 0.000209065, 0.000211065,
        0.000213065, 0.000215065, 0.000217065, 0.000219065, 0.000221065, 0.000223065, 0.000225065,
        0.000227065, 0.000229065, 0.000231065, 0.000233065, 0.000235065, 0.000237065, 0.000239065,
        0.000241065, 0.000243065, 0.000245065, 0.000247065, 0.000249065, 0.000251065, 0.000253065,
        0.000255065, 0.000257065, 0.000259065, 0.000261065, 0.000263065, 0.000265065, 0.000267065,
        0.000269065, 0.000271065, 0.000273065, 0.000275065, 0.000277065, 0.000279065, 0.000281065,
        0.000283065, 0.000285064, 0.000287064, 0.000289064, 0.000245267, 0.000247267, 0.000249267,
        0.000251267, 0.000253267, 0.000255267, 0.000257267, 0.000259267, 0.000261267, 0.000263267,
        0.000265267, 0.000267267, 0.000269267, 0.000271267, 0.000273267, 0.000275267, 0.000277267,
        0.000279267, 0.000281267, 0.000283267, 0.000285267, 0.000287267, 0.000289267, 0.000291267,
        0.000293267, 0.000295267, 0.000276164, 0.000278164, 0.000280164, 0.000282164, 0.000284164,
        0.000286164, 0.000288164, 0.000290164, 0.000292164, 0.000294164, 0.000296164, 0.000298164,
        0.000300164, 0.000302164, 0.000304164, 0.000306164, 0.000308164, 0.000310164, 0.000312164,
        0.000314164, 0.000316164, 0.000318164, 0.000320164, 0.000322164, 0.000313791, 0.000315791,
        0.000317791, 0.000319791, 0.000321791, 0.000323791, 0.000325791, 0.000327791, 0.000329791,
        0.000331791, 0.000333791, 0.000335791, 0.000337791, 0.000339791, 0.000341791, 0.000343791,
        0.000345791, 0.000347791, 0.000349791, 0.000351791, 0.000353791, 0.000355791, 0.000357791,
        0.000359791, 0.000361791, 0.000363791, 0.000365791, 0.000367791, 0.000369791, 0.000371791,
        0.000373791, 0.000375791, 0.000377791, 0.000379791, 0.000381791, 0.000383791, 0.000385791,
        0.000387791, 0.000389791, 0.000391791, 0.000393791, 0.000395791, 0.000397791, 0.000399791,
        0.000401791, 0.000403791, 0.000405791, 0.000305797, 0.000307797, 0.000309797, 0.000311797,
        0.000313797, 0.000315797, 0.000317797, 0.000242175, 0.000244175, 0.000246175, 0.000248175,
        0.000250175, 0.000252175, 0.000254175, 0.000256175, 0.000212097, 0.000214097, 0.000216097,
        0.000218097, 0.000220097, 0.000222097, 0.000224097, 0.000226097, 0.000228097, 0.000193632,
        0.000195632, 0.000197632, 0.000199632, 0.000182566, 0.000184566, 0.000186566, 0.000188566,
        0.000190566, 0.000192566, 0.000194566, 0.000196566, 0.000198566, 0.000169651, 0.000171651,
        0.000173651, 0.000175651, 0.000177651, 0.000179651, 0.000181651, 0.000183651, 0.000185651,
        0.000187651, 0.000189651, 0.000191651, 0.000193651, 0.000195651, 0.000197651, 0.000199651,
        0.000201651, 0.000203651, 0.000205651, 0.000207651, 0.000209651, 0.000211651, 0.000213651,
        0.000215651, 0.000217651, 0.000210487, 0.000212487, 0.000214487, 0.000216487, 0.000218487,
        0.000220487, 0.000222487, 0.000224487, 0.000226487, 0.000228487, 0.000230487, 0.000232487,
        0.000234487, 0.000236487, 0.000238487, 0.000240487, 0.000242487, 0.000244487, 0.000235452,
        0.000237452, 0.000239452, 0.000241452, 0.000243452, 0.000245452, 0.000247452, 0.000249452,
        0.000251451, 0.000253451, 0.000255451, 0.000257451, 0.000259451, 0.000252287, 0.000254287,
        0.000256287, 0.000258287, 0.000260287, 0.000262287, 0.000264287, 0.000266287, 0.000268287,
        0.000270287, 0.000272287, 0.000274287, 0.000276287, 0.000278287, 0.000280287, 0.000282287,
        0.000284287, 0.000286287, 0.000288287, 0.000290287, 0.000292287, 0.000294287, 0.000296287,
        0.000298287, 0.000300287, 0.000302287, 0.000304287, 0.000306287, 0.000308287, 0.000310287,
        0.000312287, 0.000314287, 0.000316287, 0.000318287, 0.000320287, 0.000322287, 0.000324287,
        0.000326287, 0.000328287, 0.000330287, 0.000332287, 0.000334287, 0.000336287, 0.000338287,
        0.000340287, 0.000342287, 0.000344287, 0.000346287, 0.000348287, 0.000350287, 0.000352287,
        0.000354287, 0.000356287, 0.000358287, 0.000360287, 0.000362287, 0.000364287, 0.000366287,
        0.000368287, 0.000370287, 0.000372287, 0.000374287, 0.000376287, 0.000378287, 0.000380287,
        0.000382287, 0.000384287, 0.000386287, 0.000388287, 0.000390287, 0.000392287, 0.000394287,
        0.000396287, 0.000398287, 0.000400287, 0.000402287, 0.000404287, 0.000406287, 0.000408287,
        0.000410287, 0.000412287, 0.000414287, 0.000416287, 0.000418287, 0.000420287, 0.000422287,
        0.000424287, 0.000426287, 0.000428287, 0.000430287, 0.000432287, 0.000305187, 0.000307187,
        0.000309187, 0.000311187, 0.000313187, 0.000315187, 0.000317187, 0.000319187, 0.000268407,
        0.000270407, 0.000272407, 0.000274407, 0.000276407, 0.000278407, 0.000280407, 0.000282407,
        0.000284407, 0.000286407, 0.000269972, 0.000271972, 0.000273972, 0.000266884, 0.000268884,
        0.000270884, 0.000272884, 0.000274884, 0.000276884, 0.000278884, 0.000280884, 0.000282884,
        0.000284884, 0.000286884, 0.000288884, 0.000290884, 0.000292884, 0.000294884, 0.000296884,
        0.000298884, 0.000300884, 0.000302884, 0.000304884, 0.000142451, 0.000144451, 0.000146451,
        0.000148451, 0.000150451, 0.000152451, 0.000154451, 0.000156451, 0.00015845, 0.00016045,
        0.00016245, 0.00016445, 0.00016645, 0.00016845, 0.00017045, 0.00017245, 0.00017445,
        0.000166786, 0.000168786, 0.000170786, 0.000172786, 0.000174786, 0.000176786, 0.000178786,
        0.000180786, 0.000182786, 0.000184786, 0.000186786, 0.000128838, 0.000130838, 0.000132838,
        0.000134838, 0.000136838, 0.000138838, 0.000140838, 0.000142838, 0.000144838, 0.000146838,
        0.000148838, 0.000150838, 0.000152838, 0.000154838, 0.000156838, 0.000158838, 0.000160838,
        0.000162838, 0.000164838, 0.000166838, 0.000168838, 0.000170838, 0.000166005, 0.000168005,
        0.000170005, 0.000172005, 0.000174005, 0.000176005, 0.000178005, 0.000180005, 0.000182005,
        0.000184005, 0.000186005, 0.000188005, 0.000190005, 0.000192005, 0.000194005, 0.000196005,
        0.000198005, 0.000200005, 0.000202005, 0.000204005, 0.000206005, 0.000208005, 0.000210005,
        0.000212005, 0.000214005, 0.000216005, 0.000218005, 0.000220005, 0.000222005, 0.000224005,
        0.000226005, 0.000228005, 0.000230005, 0.000232005, 0.000234005, 0.000236005, 0.000238005,
        0.000240005, 0.000242005, 0.000244005, 0.000235207, 0.000237207, 0.000239207, 0.000241207,
        0.000243207, 0.000245207, 0.000247207, 0.000249207, 0.000251207, 0.000253207, 0.000255207,
        0.000257207, 0.000259207, 0.000261207, 0.000263207, 0.000265207, 0.000267207, 0.000269207,
        0.00025979, 0.00026179, 0.00026379, 0.00026579, 0.00026779, 0.00026979, 0.00027179,
        0.00027379, 0.00027579, 0.00027779, 0.00027979, 0.00028179, 0.00028379, 0.00028579,
        0.00028779, 0.00028979, 0.00029179, 0.00029379, 0.00029579, 0.00029779, 0.00029979,
        0.000286491, 0.000288491, 0.000233402, 0.000235402, 0.000237402, 0.000239402, 0.000241402,
        0.000243402, 0.000245402, 0.000247402, 0.000249402, 0.000251402, 0.000253402, 0.000255402,
        0.000257402, 0.000259402, 0.000261402};

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

    xt::xtensor<double, 1> compute_Eps = xt::zeros<double>({dF.shape(0)});
    xt::xtensor<double, 1> compute_Sig = xt::zeros<double>({dF.shape(0)});
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
                break;
            }
        }

        full.quench();
        reduced.quench();

        stop.reset();

        REQUIRE(xt::allclose(full.Eps(), reduced.Eps()));
        REQUIRE(xt::allclose(full.Sig(), reduced.Sig()));

        xt::xtensor<double, 2> Epsbar = xt::average(full.Eps(), dV, {0, 1});
        xt::xtensor<double, 2> Sigbar = xt::average(full.Sig(), dV, {0, 1});

        compute_Eps(inc) = GM::Epsd(Epsbar)();
        compute_Sig(inc) = GM::Epsd(Sigbar)();
    }

    REQUIRE(xt::allclose(compute_Eps, check_Eps));
    REQUIRE(xt::allclose(compute_Sig, check_Sig));

}

}
