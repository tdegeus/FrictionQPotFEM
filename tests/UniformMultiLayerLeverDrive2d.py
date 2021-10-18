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
        nelem = mesh.nelem()
        coor = mesh.coor()
        conn = mesh.conn()
        elem = mesh.elementgrid()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftEdge()[1:], ...] = dofs[mesh.nodesRightEdge()[1:], ...]
        layers = [elem[:3, :].ravel(), elem[3, :].ravel(), elem[4:, :].ravel()]
        layers = [np.sort(i) for i in layers]

        system = FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System(
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

        Hi = []
        for layer in layers:
            yl = coor[conn[layer[0], 0], 1]
            yu = coor[conn[layer[-1], 3], 1]
            Hi += [0.5 * (yu + yl)]
        H = np.max(coor[:, 1])

        epsy = 1e-3 * np.cumsum(np.ones((nplas, 5)), axis=1)

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)
        system.layerSetDriveStiffness(1e-3)
        system.setLeverProperties(H, Hi)

        drive_active = np.zeros((3, 2), dtype=bool)
        drive_active[-1, 0] = True

        for loop in range(2):

            if loop == 0:
                system.initEventDriven(0.1, drive_active)
                delta_xlever = system.eventDriven_deltaLeverPosition()
                delta_active = system.eventDriven_targetActive()
                delta_u = system.eventDriven_deltaU()
                delta_ubar = system.eventDriven_deltaUbar()
            else:
                system.initEventDriven(delta_xlever, drive_active, delta_u, delta_ubar)
                system.setU(np.zeros_like(coor))

            system.eventDrivenStep(1e-4, False)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 0] - 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, False)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 0] - 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, True)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 0] + 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, False)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 1] - 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, True)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 1] + 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, False)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 2] - 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, False, -1)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 1] + 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, True, -1)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 1] - 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)

            system.eventDrivenStep(1e-4, False, -1)
            self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), epsy[0, 0] + 0.5 * 1e-4))
            self.assertTrue(system.residual() < 1e-5)


if __name__ == "__main__":

    unittest.main()
