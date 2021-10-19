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

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        system = FrictionQPotFEM.UniformSingleLayer2d.System(
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

        for loop in range(2):

            if loop == 0:
                system.initEventDrivenSimpleShear()
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

    def test_eventDrivenSimpleShear_historic(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        system = FrictionQPotFEM.UniformSingleLayer2d.System(
            coor,
            mesh.conn(),
            dofs,
            dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        nelas = system.elastic().size
        nplas = system.plastic().size

        epsy = 1e-1 * np.cumsum(np.random.random((nplas, 2000)), axis=1)

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)

        system.initEventDrivenSimpleShear()

        kicks = np.zeros(1800, dtype=bool)
        kicks[1::2] = True

        for kick in kicks:

            u = system.u()
            epsy = system.plastic_CurrentYieldRight()
            eps_n = GMat.Epsd(system.plastic_Eps())
            idx_n = system.plastic_CurrentIndex()

            system.eventDrivenStep(1e-4, kick)
            idx = system.plastic_CurrentIndex()
            eps = GMat.Epsd(system.plastic_Eps())
            delta_u = system.u() - u
            system.setU(u)

            # note that this implementation is flawed in negative direction
            system.addSimpleShearEventDriven(1e-4, kick)
            check_idx = system.plastic_CurrentIndex()
            check_eps = GMat.Epsd(system.plastic_Eps())
            check_u = system.u() - u

            if not np.allclose(delta_u, check_u):
                self.assertTrue(np.all(idx == check_idx))
                i = np.argmin(epsy - eps_n)
                if kick:
                    self.assertTrue(np.sum(idx.astype(int) - idx_n.astype(int)) <= 4)
                    self.assertTrue(np.sum(check_idx.astype(int) - idx_n.astype(int)) <= 4)
                    self.assertTrue(np.isclose(eps.ravel()[i], epsy.ravel()[i] + 0.5 * 1e-4))
                    # note: only failing here, by about 1e-5 (so small error)
                    self.assertTrue(np.isclose(check_eps.ravel()[i], epsy.ravel()[i] + 0.5 * 1e-4))
                else:
                    self.assertTrue(np.sum(idx.astype(int) - idx_n.astype(int)) == 0)
                    self.assertTrue(np.sum(check_idx.astype(int) - idx_n.astype(int)) == 0)
                    self.assertTrue(np.isclose(eps.ravel()[i], epsy.ravel()[i] - 0.5 * 1e-4))
                    self.assertTrue(np.isclose(check_eps.ravel()[i], epsy.ravel()[i] - 0.5 * 1e-4))

            else:
                self.assertTrue(np.allclose(delta_u, check_u))


if __name__ == "__main__":

    unittest.main()
