import os
import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np
import prrng


class test_UniformMultiLayerIndividualDrive2d(unittest.TestCase):
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

        mesh = GooseFEM.Mesh.Quad4.Regular(N, 11, h)

        coor = mesh.coor()
        conn = mesh.conn()
        dofs = mesh.dofs()
        elem = mesh.elementgrid()

        height = [1.5 * h, 3.5 * h, 5.5 * h, 7.5 * h, 9.5 * h]
        active = [[False, False], [False, False], [True, False], [False, False], [True, False]]
        layers = [
            elem[:3, :].ravel(),
            elem[3, :].ravel(),
            elem[4:7, :].ravel(),
            elem[7, :].ravel(),
            elem[8:, :].ravel(),
        ]
        layers = [np.sort(i) for i in layers]
        is_plastic = [False, True, False, True, False]

        left = mesh.nodesLeftOpenEdge()
        right = mesh.nodesRightOpenEdge()
        dofs[left] = dofs[right]

        mesh.nodesTopEdge()
        mesh.nodesBottomEdge()

        nelas = sum(i.size for i, p in zip(layers, is_plastic) if not p)
        nplas = sum(i.size for i, p in zip(layers, is_plastic) if p)

        generators = prrng.pcg32_array(np.arange(nplas), np.zeros(nplas))
        epsy = np.hstack((generators.random([1]), generators.weibull([1000], k=2.0)))
        epsy *= 1.0e-3
        epsy += 1.0e-5
        epsy = np.cumsum(epsy, 1)

        c = 1.0
        G = 1.0
        K = 10.0 * G
        rho = G / c**2.0
        qL = 2.0 * np.pi / L
        qh = 2.0 * np.pi / h
        alpha = np.sqrt(2.0) * qL * c * rho
        dt = 1.0 / (c * qh) / 10.0

        system = FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System(
            coor=coor,
            conn=conn,
            dofs=dofs,
            iip=dofs[mesh.nodesBottomEdge(), :].ravel(),
            elem=layers,
            node=[np.unique(conn[i, :]) for i in layers],
            layer_is_plastic=is_plastic,
            elastic_K=FrictionQPotFEM.moduli_toquad(K * np.ones(nelas)),
            elastic_G=FrictionQPotFEM.moduli_toquad(G * np.ones(nelas)),
            plastic_K=FrictionQPotFEM.moduli_toquad(K * np.ones(nplas)),
            plastic_G=FrictionQPotFEM.moduli_toquad(G * np.ones(nplas)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=dt,
            rho=rho,
            alpha=alpha,
            eta=0,
            drive_is_active=active,
            k_drive=1e-3,
        )

        # Drive

        delta_u = system.layerTargetUbar_affineSimpleShear(0.1, height)
        system.initEventDriven(delta_u, active)

        nstep = 20
        collect_Eps = np.zeros(nstep)
        collect_Sig = np.zeros(nstep)
        collect_Sig_plastic = np.zeros(nstep)
        dV = system.quad.AsTensor(2, system.dV)

        for step in range(nstep):

            if step % 2 == 0:
                system.eventDrivenStep(deps=1e-5, kick=True, iterative=True, yield_element=False)
                niter = system.minimise()
                self.assertTrue(niter >= 0)
            else:
                system.eventDrivenStep(deps=1e-5, kick=False, iterative=True, yield_element=False)

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
