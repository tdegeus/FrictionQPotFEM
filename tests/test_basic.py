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

        elastic = np.array([0, 1, 2, 6, 7, 8])
        plastic = np.array([3, 4, 5])

        system = FrictionQPotFEM.Generic2d.System(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofs(),
            iip=np.arange(mesh.nnode * mesh.ndim),
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(np.ones([plastic.size, 1])),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
        )

        self.assertEqual(system.N, 3)
        self.assertEqual(system.type, "FrictionQPotFEM.Generic2d.System")

    def test_FrictionQPotFEM_UniformSingleLayer2d_System(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        elastic = np.array([0, 1, 2, 6, 7, 8])
        plastic = np.array([3, 4, 5])

        system = FrictionQPotFEM.UniformSingleLayer2d.System(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofs(),
            iip=np.arange(mesh.nnode * mesh.ndim),
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(np.ones([plastic.size, 1])),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
        )

        self.assertEqual(system.N, 3)
        self.assertEqual(system.type, "FrictionQPotFEM.UniformSingleLayer2d.System")

    def test_FrictionQPotFEM_UniformMultiLayerIndividualDrive2d_System(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        layers = [i * 3 + np.arange(3) for i in range(3)]
        is_plastic = [False, True, False]
        nelas = sum(i.size for i, p in zip(layers, is_plastic) if not p)
        nplas = sum(i.size for i, p in zip(layers, is_plastic) if p)

        system = FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofs(),
            iip=np.arange(mesh.nnode * mesh.ndim),
            elem=layers,
            node=[i * 4 + np.arange(8) for i in range(3)],
            layer_is_plastic=is_plastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(nelas)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(nelas)),
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(nplas)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(nplas)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(np.ones([nplas, 1])),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
            drive_is_active=np.ones((len(layers), 2)),
            k_drive=1,
        )

        self.assertEqual(system.N, 3)
        self.assertEqual(system.type, "FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System")

    def test_FrictionQPotFEM_UniformMultiLayerLeverDrive2d_System(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)

        layers = [i * 3 + np.arange(3) for i in range(3)]
        is_plastic = [False, True, False]
        nelas = sum(i.size for i, p in zip(layers, is_plastic) if not p)
        nplas = sum(i.size for i, p in zip(layers, is_plastic) if p)

        system = FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofs(),
            iip=np.arange(mesh.nnode * mesh.ndim),
            elem=layers,
            node=[i * 4 + np.arange(8) for i in range(3)],
            layer_is_plastic=is_plastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(np.ones(nelas)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(nelas)),
            plastic_K=FrictionQPotFEM.moduli_toquad(np.ones(nplas)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(nplas)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(np.ones([nplas, 1])),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
            drive_is_active=np.ones((len(layers), 2)),
            k_drive=1,
            H=3,
            hi=np.arange(len(layers)),
        )

        self.assertEqual(system.N, 3)
        self.assertEqual(system.type, "FrictionQPotFEM.UniformMultiLayerLeverDrive2d.System")


if __name__ == "__main__":

    unittest.main()
