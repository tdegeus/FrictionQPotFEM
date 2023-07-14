import time
import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import numpy as np


class test_Generic2d(unittest.TestCase):
    """
    Tests
    """

    def test_dimension(self):
        for ih, h in enumerate([1, 1.5, 2, np.pi, 4]):
            mesh = GooseFEM.Mesh.Quad4.Regular(3, 3, h=h)
            coor = mesh.coor()
            dofs = mesh.dofs()
            dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]
            iip = dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel()

            elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
            plastic = np.array([3, 4, 5], dtype=np.uint64)
            eps0 = 5e-4
            epsy = 2 * eps0 * np.cumsum(np.ones((plastic.size, 5)), axis=1)
            sig0 = 2 * 1 * eps0

            system = FrictionQPotFEM.UniformSingleLayerThermal2d.System(
                coor=coor,
                conn=mesh.conn(),
                dofs=dofs,
                iip=iip,
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
                temperature_dinc=2,
                temperature_seed=0,
                temperature=3.4 * sig0,
            )

            if ih == 0:
                f0 = np.copy(system.fthermal)
            else:
                self.assertTrue(np.allclose(f0, system.fthermal / h))

            vector = GooseFEM.VectorPartitioned(mesh.conn(), dofs, iip)
            matrix = GooseFEM.MatrixPartitioned(mesh.conn(), dofs, iip)
            solver = GooseFEM.MatrixPartitionedSolver()
            elem = GooseFEM.Element.Quad4.Quadrature(vector.AsElement(coor))
            mat = GMat.Elastic2d(
                np.ones([vector.nelem, elem.nip]), np.ones([vector.nelem, elem.nip])
            )

            Ke = elem.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
            matrix.assemble(Ke)

            ninc = 500
            sig = np.zeros_like(mat.Sig[plastic, ...])
            abssig = np.zeros_like(mat.Sig[plastic, ...])

            for inc in range(ninc):
                disp = np.zeros_like(coor)
                system.u = disp
                solver.solve(matrix, system.fthermal, disp)

                ue = vector.AsElement(disp)
                elem.symGradN_vector(ue, mat.Eps)
                mat.refresh()

                sig += mat.Sig[plastic, ...]
                abssig += np.abs(mat.Sig[plastic, ...])
                system.timeStep()

                self.assertTrue(
                    np.allclose(
                        disp[mesh.nodesLeftOpenEdge(), ...], disp[mesh.nodesRightOpenEdge(), ...]
                    )
                )

            self.assertLess(np.abs(np.mean(GMat.Sigd(abssig / ninc)) / system.temperature - 1), 0.2)
            self.assertLess(np.mean(GMat.Sigd(sig / ninc)) / system.temperature, 0.1)

    def test_restore(self):
        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]
        iip = dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel()

        elastic = np.array([0, 1, 2, 6, 7, 8], dtype=np.uint64)
        plastic = np.array([3, 4, 5], dtype=np.uint64)
        eps0 = 5e-4
        epsy = 2 * eps0 * np.cumsum(np.ones((plastic.size, 5)), axis=1)
        sig0 = 2 * 1 * eps0

        system = FrictionQPotFEM.UniformSingleLayerThermal2d.System(
            coor=coor,
            conn=mesh.conn(),
            dofs=dofs,
            iip=iip,
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
            temperature_dinc=2,
            temperature_seed=int(time.time()),
            temperature=sig0,
        )

        ninc = 500
        incs = list(range(ninc))
        f = np.zeros([ninc] + list(system.fthermal.shape))

        for inc in incs:
            f[inc, ...] = system.fthermal
            system.timeStep()

        np.random.shuffle(incs)

        for inc in incs:
            system.inc = inc
            self.assertTrue(np.allclose(f[inc, ...], system.fthermal))


if __name__ == "__main__":
    unittest.main()
