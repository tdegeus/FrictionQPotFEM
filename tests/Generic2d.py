import unittest

import GMatElastoPlasticQPot.Cartesian2d as GMat
import FrictionQPotFEM
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


if __name__ == "__main__":

    unittest.main()
