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

        mesh = GooseFEM.Mesh.Quad4.Regular(nx=3, ny=2 * 3 + 1)
        coor = mesh.coor()
        conn = mesh.conn()
        elem = mesh.elementgrid()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftEdge()[1:], ...] = dofs[mesh.nodesRightEdge()[1:], ...]
        layers = [elem[:3, :].ravel(), elem[3, :].ravel(), elem[4:, :].ravel()]
        layers = [np.sort(i) for i in layers]

        system = FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System(
            coor,
            conn,
            dofs,
            dofs[mesh.nodesBottomEdge(), :].ravel(),
            layers,
            [np.unique(conn[i, :]) for i in layers],
            [False, True, False],
        )

        nelas = system.elastic().size
        nplas = system.plastic().size

        epsy = 1e-3 * np.cumsum(np.ones((nplas, 5)), axis=1)

        system.rho = 1
        system.alpha = 1
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.dt = 1
        system.layerSetDriveStiffness(1e-3)

        drive_active = np.zeros((3, 2), dtype=bool)
        drive_u = np.zeros((3, 2), dtype=float)

        drive_active[-1, 0] = True
        drive_u[-1, 0] = 5.0

        for loop in range(2):

            if loop == 0:
                system.initEventDriven(drive_u, drive_active)
                delta_active = system.eventDriven_targetActive()
                delta_u = system.eventDriven_deltaU()
                delta_ubar = system.eventDriven_deltaUbar()
            else:
                system.initEventDriven(delta_ubar, delta_active, delta_u)
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
