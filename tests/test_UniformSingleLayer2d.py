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


if __name__ == "__main__":

    unittest.main()
