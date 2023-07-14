import pathlib
import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
    import GooseMPL as gplt

    plt.style.use(["goose", "goose-latex", "goose-autolayout"])
    plot = True
except ImportError:
    plot = False

root = pathlib.Path(__file__).parent
basename = pathlib.Path(__file__).stem


class System(FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System):
    def __init__(
        self, nlayers: int, kdrive: float, nrow: int = 4, ncol: int = 10, eps0: float = 1000
    ):
        """
        A multilayer system with ``nlayers`` weak layers.

        :param nlayers: number of weak layers
        :param kdrive: driving stiffness
        :param nrow: number of elements rows in the elastic layers
        :param ncol: number of elements columns in any layer
        :param eps0: scale of yield strain
        """
        layer_elas = GooseFEM.Mesh.Quad4.Regular(ncol, nrow)
        layer_plas = GooseFEM.Mesh.Quad4.Regular(ncol, 1)
        stitch = GooseFEM.Mesh.Vstack()

        stitch.push_back(
            layer_elas.coor(),
            layer_elas.conn(),
            layer_elas.nodesBottomEdge(),
            layer_elas.nodesTopEdge(),
        )
        left = [layer_elas.nodesLeftOpenEdge()]
        right = [layer_elas.nodesRightOpenEdge()]

        for i in range(nlayers):
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

            left += [
                layer_plas.nodesLeftEdge(),
                layer_elas.nodesLeftOpenEdge(),
            ]

            right += [
                layer_plas.nodesRightEdge(),
                layer_elas.nodesRightOpenEdge(),
            ]

        left = stitch.nodeset(left)
        right = stitch.nodeset(right)
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

        iip = np.unique(np.concatenate((dofs[bottom, :].ravel(), dofs[top, 1].ravel())))

        elas = []
        plas = []

        for i in range(nlayer):
            if i % 2 == 0:
                elas += list(stitch.elemmap(i))
            else:
                plas += list(stitch.elemmap(i))

        epsy = np.cumsum(eps0 * np.ones((len(plas), 100)), axis=1)

        self.height = []
        for i in range(nlayer):
            yl = coor[conn[stitch.elemmap(i)[0], 0], 1]
            yu = coor[conn[stitch.elemmap(i)[-1], 3], 1]
            self.height += [0.5 * (yu + yl)]

        drive = np.zeros((nlayer, ndim), dtype=bool)
        layer_is_plastic = np.zeros(nlayer, dtype=bool)
        for i in range(nlayers):
            drive[(i + 1) * 2, 0] = True
            layer_is_plastic[i * 2 + 1] = True

        FrictionQPotFEM.UniformMultiLayerIndividualDrive2d.System.__init__(
            self,
            coor=coor,
            conn=conn,
            dofs=dofs,
            iip=iip,
            elem=stitch.elemmap(),
            node=stitch.nodemap(),
            layer_is_plastic=layer_is_plastic,
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
            k_drive=kdrive,
        )

        self.elem_color = np.zeros(stitch.nelem)
        for i in range(nlayer):
            self.elem_color[stitch.elemmap(i)] = i


class test_historic(unittest.TestCase):
    def test_main(self):
        """
        Check against historical results.
        """
        system = System(nlayers=1, kdrive=10, nrow=6, ncol=20, eps0=0.01)
        system.layerTargetUbar[-1, 0] = 0.5
        system.refresh()

        ret = system.minimise(nmargin=5)
        self.assertEqual(ret, 0)
        self.assertLess(system.residual, 1e-5)

        with h5py.File(root / f"{basename}.h5") as file:
            self.assertTrue(np.allclose(file["u"][...], system.u))


class test_perturbation(unittest.TestCase):
    def test_main(self):
        """
        Prescribe simple shear and check how well it is reproduced.
        """

        def run(nlayers, ax_target, ax_stress):
            ret_kdrive = [1e-10, 1e-6, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
            ret_u = []
            ret_ubar = []
            ret_sigma = []
            gamma0 = 0.15

            for kdrive in ret_kdrive:
                system = System(nlayers, kdrive)
                for i in range(system.layerTargetUbar.shape[0]):
                    if system.layerTargetActive[i, 0]:
                        system.layerTargetUbar[i, 0] = system.height[i] * gamma0
                ubar = system.layerTargetUbar.copy()

                system.refresh()
                ret = system.minimise(nmargin=5)
                self.assertEqual(ret, 0)
                self.assertLess(system.residual, 2e-5)

                ret_u.append(system.u.copy())
                ret_ubar.append(system.layerUbar.copy())

                dV = system.quad.AsTensor(2, system.quad.dV)
                ret_sigma.append(GMat.Sigd(np.average(system.Sig, weights=dV, axis=(0, 1))))

            if not plot:
                return

            ax_target.plot(
                ret_kdrive,
                [(ubar[-1, 0] - u[-1, 0]) / system.height[-1] / gamma0 for u in ret_ubar],
                marker="o",
            )
            ax_stress.plot(ret_kdrive, ret_sigma, marker="o")

            fig, axes = plt.subplots(nrows=len(ret_kdrive), figsize=(8, 10))
            for i in range(len(ret_kdrive)):
                ax = axes[i]
                gplt.patch(
                    coor=system.coor + ret_u[i],
                    conn=system.conn,
                    cindex=system.elem_color,
                    axis=ax,
                    cmap="jet",
                    clim=[0, 6],
                )
                ax.set_xlim([0, 12])
                ax.set_ylim([0, system.coor[-1, -1]])
                ax.set_title(rf"$K = {ret_kdrive[i]:.0e}$")
            fig.savefig(root / f"{basename}_perturbation_n={nlayers}_mesh.pdf")
            plt.close(fig)

        # --

        if plot:
            fig, axes = gplt.subplots(ncols=2)
        else:
            ax = [None, None]

        run(1, *axes)
        run(2, *axes)
        run(3, *axes)

        if not plot:
            return

        ax = axes[0]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$(\bar{u}_0 - \bar{u}) / (h \gamma_0)$")
        ax.set_ylim([1e-5, 2])

        ax = axes[1]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$\Sigma$")

        fig.savefig(root / f"{basename}_perturbation.pdf")
        plt.close(fig)


class test_load(unittest.TestCase):
    def test_main(self):
        """
        Load to a fixed shear.
        """

        def run(nlayers, ax_shear, ax_stress):
            ret_kdrive = [1e-10, 1e-6, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
            ret_ubar = []
            ret_sigma = []
            gamma0 = 0.15

            for kdrive in ret_kdrive:
                system = System(nlayers, kdrive)
                for i in range(system.layerTargetUbar.shape[0]):
                    if system.layerTargetActive[i, 0]:
                        system.layerTargetUbar[i, 0] = system.height[i] * gamma0

                system.refresh()
                ret = system.minimise(nmargin=5)
                self.assertEqual(ret, 0)
                self.assertLess(system.residual, 2e-5)

                gamma = GMat.Epsd(system.Eps[0, 0, ...])
                c = gamma / gamma0
                system.u = system.u / c
                system.layerTargetUbar = system.layerTargetUbar / c
                system.refresh()
                self.assertLess(system.residual, 2e-5)
                self.assertAlmostEqual(GMat.Epsd(system.Eps[0, 0, ...]), gamma0)

                dV = system.quad.AsTensor(2, system.quad.dV)
                ret_sigma.append(GMat.Sigd(np.average(system.Sig, weights=dV, axis=(0, 1))))
                ret_ubar.append(system.layerTargetUbar[-1, 0] / system.height[-1] / gamma0)

            if not plot:
                return

            ax_shear.plot(ret_kdrive, ret_ubar, marker="o")
            ax_stress.plot(ret_kdrive, ret_sigma, marker="o")

        # --

        if plot:
            fig, axes = gplt.subplots(ncols=2)
        else:
            ax = [None, None]

        run(1, *axes)
        run(2, *axes)
        run(3, *axes)

        if not plot:
            return

        ax = axes[0]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$\gamma$")

        ax = axes[1]
        ax.set_xscale("log")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$\Sigma$")
        ax.set_ylim([0, 0.3])

        fig.savefig(root / f"{basename}_stress.pdf")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
