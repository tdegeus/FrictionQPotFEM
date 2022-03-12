import unittest

import FrictionQPotFEM
import GooseFEM
import numpy as np


class Test(unittest.TestCase):
    """
    Tests.
    """

    def test_FrictionQPotFEM_Generic2d_System(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        system = FrictionQPotFEM.Generic2d.System(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            np.arange(mesh.nnode() * mesh.ndim()),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        self.assertEqual(system.N(), 3)
        self.assertEqual(system.type(), "FrictionQPotFEM.Generic2d.System")

    def test_FrictionQPotFEM_UniformSingleLayer2d_System(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        system = FrictionQPotFEM.UniformSingleLayer2d.System(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            np.arange(mesh.nnode() * mesh.ndim()),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        self.assertEqual(system.N(), 3)
        self.assertEqual(system.type(), "FrictionQPotFEM.UniformSingleLayer2d.System")

    def test_FrictionQPotFEM_UniformMultiLayerIndividualDrive2d_System(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        system = FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            np.arange(mesh.nnode() * mesh.ndim()),
            [i * 3 + np.arange(3) for i in range(3)],
            [i * 4 + np.arange(8) for i in range(3)],
            [False, True, False],
        )

        self.assertEqual(system.N(), 3)
        self.assertEqual(system.type(), "FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System")

    def test_FrictionQPotFEM_UniformMultiLayerLeverDrive2d_System(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        system = FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System(
            mesh.coor(),
            mesh.conn(),
            mesh.dofs(),
            np.arange(mesh.nnode() * mesh.ndim()),
            [i * 3 + np.arange(3) for i in range(3)],
            [i * 4 + np.arange(8) for i in range(3)],
            [False, True, False],
        )

        self.assertEqual(system.N(), 3)
        self.assertEqual(system.type(), "FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System")


if __name__ == "__main__":

    unittest.main()
