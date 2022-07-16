import os
import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import prrng


class test_UniformSingleLayer2d(unittest.TestCase):
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
        elastic = np.setdiff1d(np.arange(mesh.nelem()), plastic)

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

        system = FrictionQPotFEM.UniformSingleLayer2d.System(
            coor, conn, dofs, iip, elastic, plastic
        )
        system.setMassMatrix(rho * np.ones(mesh.nelem()))
        system.setDampingMatrix(alpha * np.ones(mesh.nelem()))
        system.setElastic(K * np.ones(elastic.size), G * np.ones(elastic.size))
        system.setPlastic(K * np.ones(plastic.size), G * np.ones(plastic.size), epsy)
        system.dt = dt

        # Run

        system.initEventDrivenSimpleShear()

        ninc = 500
        collect_Eps = np.zeros(ninc)
        collect_Sig = np.zeros(ninc)
        collect_Sig_plastic = np.zeros(ninc)
        quad = system.quad()
        dV = quad.AsTensor(2, quad.dV())

        for inc in range(ninc):

            if inc % 2 == 0:
                system.eventDrivenStep(deps=1e-5, kick=True)
                system.minimise()
            else:
                system.eventDrivenStep(deps=1e-5, kick=False)

            Epsbar = np.average(system.Eps(), weights=dV, axis=(0, 1))
            Sigbar = np.average(system.Sig(), weights=dV, axis=(0, 1))
            collect_Eps[inc] = GMat.Epsd(Epsbar)
            collect_Sig[inc] = GMat.Sigd(Sigbar)
            collect_Sig_plastic[inc] = GMat.Sigd(np.mean(system.plastic_Sig(), axis=(0, 1)))

        with h5py.File(os.path.splitext(__file__)[0] + ".h5") as file:
            self.assertTrue(np.allclose(collect_Eps, file["Eps"][...]))
            self.assertTrue(np.allclose(collect_Sig, file["Sig"][...]))
            self.assertTrue(np.allclose(collect_Sig_plastic, file["Sig_plastic"][...]))
            self.assertTrue(np.allclose(system.u(), file["u_last"][...]))


if __name__ == "__main__":

    unittest.main()
