import os
import unittest

import FrictionQPotFEM
import GooseFEM
import h5py
import numpy as np

basedir = os.path.dirname(os.path.abspath(__file__))
basename = os.path.splitext(os.path.basename(__file__))[0]
filepath = os.path.join(basedir, f"{basename}.h5")


class test_stich_mesh(unittest.TestCase):
    """
    Simple example to stitch finite element mesh.
    """

    def test_main(self):

        layer_elas = GooseFEM.Mesh.Quad4.Regular(20, 6)
        layer_plas = GooseFEM.Mesh.Quad4.Regular(20, 1)

        stitch = GooseFEM.Mesh.Vstack()

        stitch.push_back(
            layer_elas.coor(),
            layer_elas.conn(),
            layer_elas.nodesBottomEdge(),
            layer_elas.nodesTopEdge(),
        )

        stitch.push_back(
            layer_plas.coor(),
            layer_plas.conn(),
            layer_plas.nodesBottomEdge(),
            layer_plas.nodesTopEdge(),
        )

        stitch.push_back(
            layer_elas.coor(),
            layer_elas.conn(),
            layer_elas.nodesBottomEdge(),
            layer_elas.nodesTopEdge(),
        )

        left = stitch.nodeset(
            [
                layer_elas.nodesLeftOpenEdge(),
                layer_plas.nodesLeftEdge(),
                layer_elas.nodesLeftOpenEdge(),
            ]
        )

        right = stitch.nodeset(
            [
                layer_elas.nodesRightOpenEdge(),
                layer_plas.nodesRightEdge(),
                layer_elas.nodesRightOpenEdge(),
            ]
        )

        bottom = stitch.nodeset(layer_elas.nodesBottomEdge(), 0)
        top = stitch.nodeset(layer_elas.nodesTopEdge(), 2)

        stitch.nelem
        coor = stitch.coor
        conn = stitch.conn
        ndim = stitch.ndim
        nlayer = stitch.nmesh

        dofs = stitch.dofs()
        dofs[right, :] = dofs[left, :]
        dofs[top[0], 0] = dofs[top[-1], 0]
        dofs[bottom[0], 0] = dofs[bottom[-1], 0]
        dofs = GooseFEM.Mesh.renumber(dofs)

        iip = np.concatenate((dofs[bottom, :].ravel(), dofs[top, 1].ravel()))

        elas = []
        plas = []

        for i in range(nlayer):
            if i % 2 == 0:
                elas += list(stitch.elemmap(i))
            else:
                plas += list(stitch.elemmap(i))

        epsy = np.cumsum(0.01 * np.ones((len(plas), 100)), axis=1)

        ubar = np.zeros((nlayer, ndim))
        drive = np.zeros((nlayer, ndim), dtype=bool)
        ubar[-1, 0] = 0.5
        drive[-1, 0] = True

        system = FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System(
            coor=coor,
            conn=conn,
            dofs=dofs,
            iip=iip,
            elem=stitch.elemmap(),
            node=stitch.nodemap(),
            layer_is_plastic=[False, True, False],
            elastic_K=FrictionQPotFEM.moduli_toquad(10 * np.ones(len(elas))),
            elastic_G=FrictionQPotFEM.moduli_toquad(1 * np.ones(len(elas))),
            plastic_K=FrictionQPotFEM.moduli_toquad(10 * np.ones(len(plas))),
            plastic_G=FrictionQPotFEM.moduli_toquad(1 * np.ones(len(plas))),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=0.1,
            rho=1,
            alpha=0.01,
            eta=0,
            drive_is_active=drive,
            k_drive=10,
        )

        system.layerSetTargetUbar(ubar)
        ret = system.minimise(nmargin=5)
        self.assertGreaterEqual(ret, 0)

        ubar[1, 0] = 0.25
        self.assertTrue(np.allclose(ubar, system.layerUbar(), rtol=1e-4, atol=5e-2))

        with h5py.File(filepath) as file:
            self.assertTrue(np.allclose(file["u"][...], system.u))

        # color_layer = np.zeros(stitch.nelem)
        # for i in range(3):
        #     color_layer[stitch.elemmap(i)] = i

        # import matplotlib.pyplot as plt
        # import GooseMPL as gplt

        # plt.style.use(["goose", "goose-latex"])

        # fig, ax = plt.subplots()

        # gplt.patch(coor=coor + system.u, conn=conn, cindex=color_layer, axis=ax)
        # plt.show()
        # plt.close(fig)


if __name__ == "__main__":

    unittest.main()
