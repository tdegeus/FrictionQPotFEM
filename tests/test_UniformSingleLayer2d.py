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
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        epsy = 1e-3 * np.cumsum(np.ones((plastic.size, 5)), axis=1)

        system = FrictionQPotFEM.UniformSingleLayer2d.System(
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

        for loop in range(2):

            if loop == 0:
                system.initEventDrivenSimpleShear()
                delta_u = system.eventDriven_deltaU
            else:
                system.eventDriven_setDeltaU(delta_u)
                system.u = np.zeros_like(coor)

            directions = [1, 1, 1, 1, 1, 1, -1, -1, -1]
            kicks = [False, False, True, False, True, False, False, True, False]
            indices = [0, 0, 0, 1, 1, 2, 1, 1, 0]
            multi = [-1, -1, +1, -1, +1, -1, +1, -1, +1]

            for direction, kick, index, m in zip(directions, kicks, indices, multi):

                eps_expect = epsy[0, index] + m * 0.5 * 1e-4
                system.eventDrivenStep(1e-4, kick, direction)
                self.assertTrue(np.allclose(GMat.Epsd(system.plastic.Eps), eps_expect))
                self.assertTrue(system.residual() < 1e-5)


if __name__ == "__main__":

    unittest.main()
