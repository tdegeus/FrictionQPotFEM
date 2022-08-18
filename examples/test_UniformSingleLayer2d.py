import os
import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot  # noqa: F401
import GooseFEM
import h5py
import numpy as np
import prrng
import tqdm

basedir = os.path.dirname(os.path.abspath(__file__))
basename = os.path.splitext(os.path.basename(__file__))[0]
filepath = os.path.join(basedir, f"{basename}.h5")


class test_UniformSingleLayer2d(unittest.TestCase):
    """
    Example for UniformSingleLayer2d.
    This also assures that the code does not deviate.

    Checked backward compatibility:
    - Event-driven >= 0.18.0.
    - Applied fixed displacement >= 0.10.0.
    """

    def test_main(self):

        N = 3**2
        h = np.pi
        L = h * N

        mesh = GooseFEM.Mesh.Quad4.FineLayer(N, N, h)

        coor = mesh.coor()
        conn = mesh.conn()
        dofs = mesh.dofs()

        plastic = mesh.elementsMiddleLayer()
        elastic = np.setdiff1d(np.arange(mesh.nelem), plastic)

        dofs[mesh.nodesRightOpenEdge(), :] = dofs[mesh.nodesLeftOpenEdge(), :]
        dofs = GooseFEM.Mesh.renumber(dofs)

        iip = np.concatenate(
            (
                dofs[mesh.nodesBottomEdge(), :].ravel(),
                dofs[mesh.nodesTopEdge(), :].ravel(),
            )
        )

        c = 1
        G = 1
        K = 10 * G
        rho = G / c**2
        qL = 2 * np.pi / L
        qh = 2 * np.pi / h
        alpha = np.sqrt(2) * qL * c * rho
        dt = 1 / (c * qh) / 10

        initstate = np.arange(N)
        initseq = np.zeros(N)
        generators = prrng.pcg32_array(initstate, initseq)

        epsy = 1e-5 + 1e-3 * generators.weibull([1000], k=2.0, l=1.0)
        epsy[:, 0] = 1e-5 + 1e-3 * generators.random([1]).ravel()
        epsy = np.cumsum(epsy, 1)

        deps = 1e-5

        system = FrictionQPotFEM.UniformSingleLayer2d.System(
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

        system.initEventDrivenSimpleShear()

        check = {}
        pbar = tqdm.tqdm(range(1000))

        for step in pbar:

            inc_n = system.inc
            i_n = np.copy(system.plastic.i)

            if step % 2 == 0:

                system.eventDrivenStep(deps, kick=True)
                self.assertTrue(np.max(system.plastic.i - i_n) == 1)

                ret = system.minimise(nmargin=5)
                self.assertTrue(ret == 0)
                pbar.set_description(f"step: {step:4d}, iiter = {system.inc - inc_n:8d}")

            else:

                system.eventDrivenStep(deps, kick=False)
                self.assertTrue(np.all(system.plastic.i == i_n))
                self.assertTrue(system.residual() < 1e-2)

            check[step] = np.copy(system.u)

        with h5py.File(filepath) as file:
            for step in check:
                self.assertTrue(np.allclose(file[f"{step:d}"][...], check[step]))


if __name__ == "__main__":

    unittest.main()
