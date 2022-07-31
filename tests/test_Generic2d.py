import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import numpy as np


def initelastic_allquad(draw, nip=4):
    """
    'Broadcast' to initialise elastic, same for all integration points.
    """

    epsy = np.empty([draw.shape[0], nip, draw.shape[1] + 1])

    for q in range(nip):
        epsy[:, q, 0] = -draw[:, 0]
        epsy[:, q, 1:] = draw

    return epsy


class test_Generic2d(unittest.TestCase):
    """
    Tests
    """

    def test_version_dependencies(self):

        deps = FrictionQPotFEM.Generic2d.version_dependencies()
        deps = [i.split("=")[0] for i in deps]

        self.assertTrue("frictionqpotfem" in deps)
        self.assertTrue("gmatelastoplasticqpot" in deps)
        self.assertTrue("gmattensor" in deps)
        self.assertTrue("qpot" in deps)
        self.assertTrue("xtensor" in deps)
        self.assertTrue("xtensor-python" in deps)
        self.assertTrue("xtl" in deps)

    def test_eventDrivenSimpleShear(self):
        """
        Simple test of event driven simple shear in a homogeneous system:
        Load forward and backward (for the latter: test current implementation limitation).
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = np.cumsum(np.ones((plastic.size, 5)), axis=1)

        self.assertTrue(
            np.allclose(initelastic_allquad(epsy), FrictionQPotFEM.epsy_initelastic_toquad(epsy))
        )

        system = FrictionQPotFEM.Generic2d.System(
            coor=coor,
            conn=mesh.conn(),
            dofs=dofs,
            iip=dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
        )

        self.assertEqual(system.rho, 1)
        self.assertEqual(system.alpha, 1)
        self.assertEqual(system.dt, 1)

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        for loop in range(2):

            if loop == 0:
                system.eventDriven_setDeltaU(delta_u)
                delta_u = system.eventDriven_deltaU
            else:
                system.eventDriven_setDeltaU(delta_u)
                system.u = np.zeros_like(coor)

            settings = [
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 1, 0, +1, 0],  # :    |.   |    |    |
                [+1, 0, 1, -1, 0],  # :    |   .|    |    |
                [+1, 1, 1, +1, 0],  # :    |    |.   |    |
                [+1, 0, 2, -1, 0],  # :    |    |   .|    |
                [+1, 1, 2, +1, 0],  # :    |    |    |.   |
                [-1, 0, 2, +1, 0],  # :    |    |    |.   |
                [-1, 1, 2, -1, 0],  # :    |    |   .|    |
                [-1, 0, 1, +1, 0],  # :    |    |.   |    |
                [-1, 1, 1, -1, 0],  # :    |   .|    |    |
                [-1, 0, 0, +1, 0],  # :    |.   |    |    |
                [-1, 1, 0, -1, 0],  # :   .|    |    |    |
                [-1, 0, 0, -1, 1],  # :   .|    |    |    | (symmetry, throw)
                [-1, 1, 0, +1, 1],  # :    |.   |    |    | (symmetry, not tested)
                [-1, 0, 1, -1, 1],  # :    |   .|    |    | (symmetry, not tested)
            ]

            for direction, kick, index, f, throw in settings:

                eps_expect = epsy[0, index] + f * 0.5 * 0.1

                if throw:
                    with self.assertRaises(IndexError):
                        system.eventDrivenStep(0.1, kick, direction)
                    break

                system.eventDrivenStep(0.1, kick, direction)

                self.assertTrue(np.allclose(GMat.Epsd(system.plastic.Eps), eps_expect))
                self.assertTrue(system.residual() < 1e-5)

                self.assertTrue(system.boundcheck_right(0))

                if index == 3:
                    self.assertFalse(system.boundcheck_right(3))

    def test_eventDrivenSimpleShear_random(self):
        """
        Like :py:func:`test_eventDrivenSimpleShear` but with random yield strains.
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = 1e-2 * np.cumsum(np.random.random((plastic.size, 100)), axis=1)
        deps = 0.1 * np.min(np.diff(epsy, axis=1))

        system = FrictionQPotFEM.Generic2d.System(
            coor=coor,
            conn=mesh.conn(),
            dofs=dofs,
            iip=dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
        )

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        system.eventDriven_setDeltaU(delta_u)

        kicks = np.zeros(50, dtype=bool)
        kicks[1::2] = True

        for kick in kicks:
            idx_n = np.copy(system.plastic.i)
            u_n = np.copy(system.u)
            uptr = system.u

            system.eventDrivenStep(deps, kick, iterative=True)

            if not kick:
                self.assertTrue(not np.allclose(u_n, uptr))

            idx = system.plastic.i
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

            system.u = u_n
            system.eventDrivenStep(deps, kick)
            idx = system.plastic.i
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

        for kick in kicks:
            idx_n = np.copy(system.plastic.i)
            u_n = np.copy(system.u)

            system.u = u_n
            system.eventDrivenStep(deps, kick, -1, iterative=True)
            idx = system.plastic.i
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

            if np.any(idx_n == 0):
                with self.assertRaises(IndexError):
                    system.eventDrivenStep(deps, kick, -1)
                break

            system.u = u_n
            system.eventDrivenStep(deps, kick, -1)
            idx = system.plastic.i
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

    def test_eventDrivenSimpleShear_element(self):
        """
        Like :py:func:`test_eventDrivenSimpleShear` but with slightly different yield strains
        per element.
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = np.cumsum(np.ones((plastic.size, 5)), axis=1)

        system = FrictionQPotFEM.Generic2d.System(
            coor=coor,
            conn=mesh.conn(),
            dofs=dofs,
            iip=dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
        )

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        for loop in range(2):

            if loop == 0:
                system.eventDriven_setDeltaU(delta_u)
                delta_u = system.eventDriven_deltaU
            else:
                system.eventDriven_setDeltaU(delta_u)
                system.u = np.zeros_like(coor)

            settings = [
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 1, 0, +1, 0],  # :    |.   |    |    |
                [+1, 0, 1, -1, 0],  # :    |   .|    |    |
                [+1, 1, 1, +1, 0],  # :    |    |.   |    |
                [+1, 0, 2, -1, 0],  # :    |    |   .|    |
                [+1, 1, 2, +1, 0],  # :    |    |    |.   |
            ]

            for direction, kick, index, f, throw in settings:

                if not kick:
                    eps_expect = epsy[0, index] + f * 0.5 * 0.05
                else:
                    eps_expect = epsy[0, index] + f * 0.5 * 0.05

                if throw:
                    with self.assertRaises(IndexError):
                        system.eventDrivenStep(0.05, kick, direction, yield_element=True)
                    break

                system.eventDrivenStep(0.05, kick, direction, yield_element=True)

                self.assertTrue(np.allclose(GMat.Epsd(system.plastic.Eps), eps_expect))
                self.assertTrue(system.residual() < 1e-5)

    def test_flowSteps(self):
        """
        Basic test of:
        -   Generic2d.System.flowSteps
        -   Generic2d.System.timeSteps
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = 100 * np.ones((plastic.size, 1))

        system = FrictionQPotFEM.Generic2d.System(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofs(),
            iip=np.arange(mesh.nnode * mesh.ndim),
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
        )

        x = mesh.coor()
        v = np.zeros_like(x)

        for i in range(v.shape[0]):
            v[i, 0] = 0.1 * (x[i, 1] - x[0, 1])

        system.flowSteps(n=10, v=v, nmargin=0)

        # displacement is added affinely in an elastic system:
        # there is not residual force -> the system stays uniform
        self.assertTrue(np.allclose(system.u, 10 * v))
        self.assertTrue(np.allclose(system.t, 10))

        system.timeSteps(n=10, nmargin=0)

        self.assertTrue(np.allclose(system.u, 10 * v))
        self.assertTrue(np.allclose(system.t, 20))

    def test_damping_alpha_no_eta(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = np.ones((plastic.size, 1))
        alpha = 1.2

        system = FrictionQPotFEM.Generic2d.System(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofsPeriodic(),
            iip=[],
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=alpha,
            eta=0,
        )

        system.v = np.ones_like(mesh.coor())
        assert np.allclose(system.vector.AsDofs(system.fdamp), alpha)

    def test_damping_no_alpha_eta(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        conn = mesh.conn()

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = np.ones((plastic.size, 1))
        eta = 3.4

        system = FrictionQPotFEM.Generic2d.System(
            coor=coor,
            conn=conn,
            dofs=mesh.dofsPeriodic(),
            iip=[],
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=0,
            eta=eta,
        )

        f = np.zeros_like(coor)
        v = np.zeros_like(coor)

        v[conn[-3:, :], 0] = 2

        f[conn[:3, -2:], 0] = -eta
        f[conn[-3:, :2], 0] = eta

        system.v = v
        assert np.allclose(system.fdamp, f)

    def test_damping_alpha_eta(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        conn = mesh.conn()

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = np.ones((plastic.size, 1))
        alpha = 1.2
        eta = 3.4

        system = FrictionQPotFEM.Generic2d.System(
            coor=coor,
            conn=conn,
            dofs=mesh.dofsPeriodic(),
            iip=[],
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=alpha,
            eta=eta,
        )

        f = np.zeros_like(coor)
        v = np.zeros_like(coor)

        v[conn[-3:, :], 0] = 2

        f[conn[-3:, :], 0] = 2 * alpha
        f[conn[:3, -2:], 0] += -eta
        f[conn[-3:, :2], 0] += eta

        system.v = v
        assert np.allclose(system.fdamp, f)


if __name__ == "__main__":

    unittest.main()
