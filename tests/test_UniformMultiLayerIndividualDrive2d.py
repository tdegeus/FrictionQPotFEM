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
        is_plastic = [False, True, False]

        nelas = sum(i.size for i, p in zip(layers, is_plastic) if not p)
        nplas = sum(i.size for i, p in zip(layers, is_plastic) if p)
        epsy = 1e-3 * np.cumsum(np.ones((nplas, 5)), axis=1)

        system = FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System(
            coor=coor,
            conn=conn,
            dofs=dofs,
            iip=dofs[mesh.nodesBottomEdge(), :].ravel(),
            elem=layers,
            node=[np.unique(conn[i, :]) for i in layers],
            layer_is_plastic=is_plastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(nelas)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(nelas)),
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(nplas)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(nplas)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
            drive_is_active=np.ones((len(layers), 2)),
            k_drive=1e-3,
        )

        drive_active = np.zeros((3, 2), dtype=bool)
        drive_u = np.zeros((3, 2), dtype=float)

        drive_active[-1, 0] = True
        drive_u[-1, 0] = 5.0

        for loop in range(2):
            if loop == 0:
                system.initEventDriven(drive_u, drive_active)
                delta_active = system.eventDriven_targetActive
                delta_u = system.eventDriven_deltaU
                delta_ubar = system.eventDriven_deltaUbar
            else:
                system.initEventDriven(delta_ubar, delta_active, delta_u)
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
