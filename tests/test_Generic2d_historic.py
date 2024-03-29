import os
import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import prrng


class test_Generic2d(unittest.TestCase):
    """
    Tests
    """

    def test_historic(self):
        """
        A simple historic run.
        Thanks to prrng this test can be run on any platform, but also from any API (Python or C++).
        """

        # Define a geometry

        N = 3**2
        h = np.pi
        L = h * float(N)

        mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N, h)

        coor = mesh.coor()
        conn = mesh.conn()
        dofs = mesh.dofs()

        plastic = mesh.elementsMiddleLayer()
        elastic = np.setdiff1d(np.arange(mesh.nelem), plastic)

        left = mesh.nodesLeftOpenEdge()
        right = mesh.nodesRightOpenEdge()
        dofs[left] = dofs[right]

        top = mesh.nodesTopEdge()
        bottom = mesh.nodesBottomEdge()
        iip = np.concatenate((dofs[bottom].ravel(), dofs[top].ravel()))

        c = 1.0
        G = 1.0
        K = 10.0 * G
        rho = G / c**2.0
        qL = 2.0 * np.pi / L
        qh = 2.0 * np.pi / h
        alpha = np.sqrt(2.0) * qL * c * rho
        dt = 1.0 / (c * qh) / 10.0

        generators = prrng.pcg32_array(np.arange(N), np.zeros(N))
        epsy = np.hstack((generators.random([1]), generators.weibull([1000], k=2.0)))
        epsy *= 1.0e-3
        epsy += 1.0e-5
        epsy = np.cumsum(epsy, 1)

        # Initialise system

        system = FrictionQPotFEM.Generic2d.System(
            coor=coor,
            conn=conn,
            dofs=dofs,
            iip=iip,
            elastic_elem=elastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(K * np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(G * np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=FrictionQPotFEM.moduli_toquad(K * np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(G * np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=dt,
            rho=rho,
            alpha=alpha,
            eta=0,
        )

        # Run

        dF = np.zeros((1001, 2, 2))
        dF[1:, 0, 1] = 0.004 / 1000.0

        collect_Eps = np.zeros(dF.shape[0])
        collect_Sig = np.zeros(dF.shape[0])
        collect_Sig_plastic = np.zeros(dF.shape[0])
        dV = system.quad.AsTensor(2, system.dV)

        for step in range(dF.shape[0]):

            u = system.u

            for i in range(mesh.nnode):
                for j in range(mesh.ndim):
                    for k in range(mesh.ndim):
                        u[i, j] += dF[step, j, k] * (coor[i, k] - coor[0, k])

            system.u = u
            ret = system.minimise(nmargin=5)
            self.assertTrue(ret == 0)

            Epsbar = np.average(system.Eps(), weights=dV, axis=(0, 1))
            Sigbar = np.average(system.Sig(), weights=dV, axis=(0, 1))
            collect_Eps[step] = GMat.Epsd(Epsbar)
            collect_Sig[step] = GMat.Sigd(Sigbar)
            collect_Sig_plastic[step] = GMat.Sigd(np.mean(system.plastic.Sig, axis=(0, 1)))

        with h5py.File(os.path.splitext(__file__)[0] + ".h5") as file:
            self.assertTrue(np.allclose(collect_Eps, file["Eps"][...]))
            self.assertTrue(np.allclose(collect_Sig, file["Sig"][...]))
            self.assertTrue(np.allclose(collect_Sig_plastic, file["Sig_plastic"][...]))
            self.assertTrue(np.allclose(system.u, file["u_last"][...]))


if __name__ == "__main__":

    unittest.main()
