import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import numpy as np


class test_Generic2d(unittest.TestCase):
    """
    Tests
    """

    def test_scalePerturbation(self):

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
        delta_u = np.zeros_like(x)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (x[i, 0] - x[0, 0])

        system.setU(delta_u)
        Epsd_delta = GMat.Deviatoric(system.plastic_Eps(0, 0))
        system.setU(np.zeros_like(x))

        for eps_target in [0.001, 0.1, 1.0]:
            Epsd_t = GMat.Deviatoric(system.plastic_Eps(0, 0))
            c = FrictionQPotFEM.Generic2d.scalePerturbation(Epsd_t, Epsd_delta, eps_target)
            system.setU(system.u() + c * delta_u)
            self.assertTrue(np.isclose(GMat.Epsd(system.plastic_Eps(0, 0)), eps_target))

    def test_eventDrivenSimpleShear(self):

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

        epsy = 1e-3 * np.cumsum(np.ones((nplas, 5)), axis=1)

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 0] - coor[0, 0])

        for loop in range(2):

            if loop == 0:
                system.eventDriven_setDeltaU(delta_u)
                delta_u = system.eventDriven_deltaU()
            else:
                system.eventDriven_setDeltaU(delta_u)
                system.setU(np.zeros_like(coor))

            directions = [1, 1, 1, 1, 1, 1, -1, -1, -1]
            kicks = [False, False, True, False, True, False, False, True, False]
            indices = [0, 0, 0, 1, 1, 2, 1, 1, 0]
            multi = [-1, -1, +1, -1, +1, -1, +1, -1, +1]

            for direction, kick, index, m in zip(directions, kicks, indices, multi):

                eps_expect = epsy[0, index] + m * 0.5 * 1e-4
                system.eventDrivenStep(1e-4, kick, direction)
                self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), eps_expect))
                self.assertTrue(system.residual() < 1e-5)

    def test_eventDrivenSimpleShear2(self):

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
            delta_u[i, 0] = 0.1 * (coor[i, 0] - coor[0, 0])

        system.eventDriven_setDeltaU(delta_u)

        kicks = np.zeros(50, dtype=bool)
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
            system.eventDrivenStep(deps, kick, -1)
            idx = system.plastic_CurrentIndex().astype(int)

            if kick:
                self.assertTrue(np.sum(idx - idx_n) == -4)
            else:
                self.assertTrue(np.all(idx == idx_n))


if __name__ == "__main__":

    unittest.main()
