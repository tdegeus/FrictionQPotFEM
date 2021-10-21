import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import numpy as np


class test_Generic2d(unittest.TestCase):
    """
    Tests
    """

    def test_eventDrivenSimpleShear(self):
        """
        Simple test of event driven simple shear in a homogeneous system:
        Load forward and backward (for the latter: test current implementation limitation).
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor,
            mesh.conn(),
            dofs,
            dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        nelas = system.elastic().size
        nplas = system.plastic().size

        epsy = np.cumsum(np.ones((nplas, 5)), axis=1)

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        for loop in range(2):

            if loop == 0:
                system.eventDriven_setDeltaU(delta_u)
                delta_u = system.eventDriven_deltaU()
            else:
                system.eventDriven_setDeltaU(delta_u)
                system.setU(np.zeros_like(coor))

            #                      #      1    2    3    4  (position)
            settings = [           #      0    1    2    3  (index)
                [+1, 0, 0, -1, 0], # :   .|    |    |    |
                [+1, 0, 0, -1, 0], # :   .|    |    |    |
                [+1, 1, 0, +1, 0], # :    |.   |    |    |
                [+1, 0, 1, -1, 0], # :    |   .|    |    |
                [+1, 1, 1, +1, 0], # :    |    |.   |    |
                [+1, 0, 2, -1, 0], # :    |    |   .|    |
                [+1, 1, 2, +1, 0], # :    |    |    |.   |
                [-1, 0, 2, +1, 0], # :    |    |    |.   |
                [-1, 1, 2, -1, 0], # :    |    |   .|    |
                [-1, 0, 1, +1, 0], # :    |    |.   |    |
                [-1, 1, 1, -1, 0], # :    |   .|    |    |
                [-1, 0, 0, +1, 0], # :    |.   |    |    |
                [-1, 1, 0, -1, 0], # :   .|    |    |    |
                [-1, 0, 0, -1, 1], # :   .|    |    |    | (symmetry, throw)
                [-1, 1, 0, +1, 1], # :    |.   |    |    | (symmetry, not tested)
                [-1, 0, 1, -1, 1], # :    |   .|    |    | (symmetry, not tested)
            ]

            for direction, kick, index, f, throw in settings:

                eps_expect = epsy[0, index] + f * 0.5 * 0.1

                if throw:
                    with self.assertRaises(IndexError):
                        system.eventDrivenStep(0.1, kick, direction)
                    break

                system.eventDrivenStep(0.1, kick, direction)
                self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), eps_expect))
                self.assertTrue(system.residual() < 1e-5)

    def test_eventDrivenSimpleShear2(self):
        """
        Like :py:func:`test_eventDrivenSimpleShear` but with random yield strains.
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor,
            mesh.conn(),
            dofs,
            dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        nelas = system.elastic().size
        nplas = system.plastic().size

        epsy = 1e-3 * np.cumsum(np.random.random((nplas, 100)), axis=1)
        deps = 0.1 * np.min(np.diff(epsy, axis=1))

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        system.eventDriven_setDeltaU(delta_u)

        kicks = np.zeros(10, dtype=bool)
        kicks[1::2] = True

        for kick in kicks:
            idx_n = system.plastic_CurrentIndex().astype(int)
            system.eventDrivenStep(deps, kick)
            idx = system.plastic_CurrentIndex().astype(int)

            if kick:
                self.assertTrue(np.sum(idx - idx_n) == 4)
            else:
                self.assertTrue(np.all(idx == idx_n))

        for kick in kicks:
            idx_n = system.plastic_CurrentIndex().astype(int)

            if np.any(idx_n == 0):
                with self.assertRaises(IndexError):
                    system.eventDrivenStep(deps, kick, -1)
                break

            system.eventDrivenStep(deps, kick, -1)

            idx = system.plastic_CurrentIndex().astype(int)

            if kick:
                self.assertTrue(np.sum(idx - idx_n) == -4)
            else:
                self.assertTrue(np.all(idx == idx_n))

    def test_flowSteps(self):
        """
        Basic test of:
        -   Generic2d.System.flowSteps
        -   Generic2d.System.timeSteps
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        system = FrictionQPotFEM.Generic2d.HybridSystem(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            np.arange(mesh.nnode() * mesh.ndim()),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(6), np.ones(6))
        system.setPlastic(np.ones(3), np.ones(3), [[100.0], [100.0], [100.0]])
        system.setDt(1.0)

        x = mesh.coor()
        v = np.zeros_like(x)

        for i in range(v.shape[0]):
            v[i, 0] = 0.1 * (x[i, 1] - x[0, 1])

        system.flowSteps(10, v)

        # displacement is added affinely in an elastic system:
        # there is not residual force -> the system stays uniform
        self.assertTrue(np.allclose(system.u(), 10 * v))
        self.assertTrue(np.allclose(system.t(), 10))

        system.timeSteps(10)

        self.assertTrue(np.allclose(system.u(), 10 * v))
        self.assertTrue(np.allclose(system.t(), 20))


if __name__ == "__main__":

    unittest.main()
