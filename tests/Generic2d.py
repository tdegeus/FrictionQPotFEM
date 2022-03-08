import unittest

import FrictionQPotFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GooseFEM
import numpy as np


class test_Generic2d(unittest.TestCase):
    """
    Tests
    """

    def test_eventDrivenSimpleShear(self):
        """
        Simple test of event driven simple shear in a homogeneous system:
        Load forward and backward (for the latter: test current implementation limitation).
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor,
            mesh.conn(),
            dofs,
            dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        nelas = system.elastic().size
        nplas = system.plastic().size

        epsy = np.cumsum(np.ones((nplas, 5)), axis=1)

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        for loop in range(2):

            if loop == 0:
                system.eventDriven_setDeltaU(delta_u)
                delta_u = system.eventDriven_deltaU()
            else:
                system.eventDriven_setDeltaU(delta_u)
                system.setU(np.zeros_like(coor))

            settings = [
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 1, 0, +1, 0],  # :    |.   |    |    |
                [+1, 0, 1, -1, 0],  # :    |   .|    |    |
                [+1, 1, 1, +1, 0],  # :    |    |.   |    |
                [+1, 0, 2, -1, 0],  # :    |    |   .|    |
                [+1, 1, 2, +1, 0],  # :    |    |    |.   |
                [-1, 0, 2, +1, 0],  # :    |    |    |.   |
                [-1, 1, 2, -1, 0],  # :    |    |   .|    |
                [-1, 0, 1, +1, 0],  # :    |    |.   |    |
                [-1, 1, 1, -1, 0],  # :    |   .|    |    |
                [-1, 0, 0, +1, 0],  # :    |.   |    |    |
                [-1, 1, 0, -1, 0],  # :   .|    |    |    |
                [-1, 0, 0, -1, 1],  # :   .|    |    |    | (symmetry, throw)
                [-1, 1, 0, +1, 1],  # :    |.   |    |    | (symmetry, not tested)
                [-1, 0, 1, -1, 1],  # :    |   .|    |    | (symmetry, not tested)
            ]

            for direction, kick, index, f, throw in settings:

                eps_expect = epsy[0, index] + f * 0.5 * 0.1

                if throw:
                    with self.assertRaises(IndexError):
                        system.eventDrivenStep(0.1, kick, direction)
                    break

                system.eventDrivenStep(0.1, kick, direction)
                self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), eps_expect))
                self.assertTrue(system.residual() < 1e-5)

    def test_eventDrivenSimpleShear_random(self):
        """
        Like :py:func:`test_eventDrivenSimpleShear` but with random yield strains.
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor,
            mesh.conn(),
            dofs,
            dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        nelas = system.elastic().size
        nplas = system.plastic().size

        epsy = 1e-3 * np.cumsum(np.random.random((nplas, 100)), axis=1)
        deps = 0.1 * np.min(np.diff(epsy, axis=1))

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        system.eventDriven_setDeltaU(delta_u)

        kicks = np.zeros(50, dtype=bool)
        kicks[1::2] = True

        for inc, kick in enumerate(kicks):
            idx_n = system.plastic_CurrentIndex()
            u_n = system.u()

            system.eventDrivenStep(deps, kick, iterative=True)
            idx = system.plastic_CurrentIndex()
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

            system.setU(u_n)
            system.eventDrivenStep(deps, kick)
            idx = system.plastic_CurrentIndex()
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

        for kick in kicks:
            idx_n = system.plastic_CurrentIndex()
            u_n = system.u()

            system.setU(u_n)
            system.eventDrivenStep(deps, kick, -1, iterative=True)
            idx = system.plastic_CurrentIndex()
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

            if np.any(idx_n == 0):
                with self.assertRaises(IndexError):
                    system.eventDrivenStep(deps, kick, -1)
                break

            system.setU(u_n)
            system.eventDrivenStep(deps, kick, -1)
            idx = system.plastic_CurrentIndex()
            if kick:
                self.assertTrue(not np.all(idx == idx_n))
            else:
                self.assertTrue(np.all(idx == idx_n))

    def test_eventDrivenSimpleShear_element(self):
        """
        Like :py:func:`test_eventDrivenSimpleShear` but with slightly different yield strains
        per element.
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        coor = mesh.coor()
        dofs = mesh.dofs()
        dofs[mesh.nodesLeftOpenEdge(), ...] = dofs[mesh.nodesRightOpenEdge(), ...]

        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor,
            mesh.conn(),
            dofs,
            dofs[np.concatenate((mesh.nodesBottomEdge(), mesh.nodesTopEdge())), :].ravel(),
            [0, 1, 2, 6, 7, 8],
            [3, 4, 5],
        )

        nelas = system.elastic().size
        nplas = system.plastic().size

        epsy = np.cumsum(np.ones((nplas, 5)), axis=1)

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(nelas), np.ones(nelas))
        system.setPlastic(np.ones(nplas), np.ones(nplas), epsy)
        system.setDt(1.0)

        epsy_element = np.zeros(epsy.shape)
        mat = system.material_plastic()
        for e in range(mat.shape()[0]):
            c = mat.refCusp([e, 1])
            y = c.epsy() + 0.1
            epsy_element[e, :] = y[1:]
            c.reset_epsy(y, init_elastic=False)

        delta_u = np.zeros_like(coor)

        for i in range(delta_u.shape[0]):
            delta_u[i, 0] = 0.1 * (coor[i, 1] - coor[0, 1])

        for loop in range(2):

            if loop == 0:
                system.eventDriven_setDeltaU(delta_u)
                delta_u = system.eventDriven_deltaU()
            else:
                system.eventDriven_setDeltaU(delta_u)
                system.setU(np.zeros_like(coor))

            settings = [
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 0, 0, -1, 0],  # :   .|    |    |    |
                [+1, 1, 0, +1, 0],  # :    |.   |    |    |
                [+1, 0, 1, -1, 0],  # :    |   .|    |    |
                [+1, 1, 1, +1, 0],  # :    |    |.   |    |
                [+1, 0, 2, -1, 0],  # :    |    |   .|    |
                [+1, 1, 2, +1, 0],  # :    |    |    |.   |
            ]

            for direction, kick, index, f, throw in settings:

                if not kick:
                    eps_expect = epsy[0, index] + f * 0.5 * 0.05
                else:
                    eps_expect = epsy_element[0, index] + f * 0.5 * 0.05

                if throw:
                    with self.assertRaises(IndexError):
                        system.eventDrivenStep(0.05, kick, direction, yield_element=True)
                    break

                system.eventDrivenStep(0.05, kick, direction, yield_element=True)

                self.assertTrue(np.allclose(GMat.Epsd(system.plastic_Eps()), eps_expect))
                self.assertTrue(system.residual() < 1e-5)

    def test_flowSteps(self):
        """
        Basic test of:
        -   Generic2d.System.flowSteps
        -   Generic2d.System.timeSteps
        """

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofs(),
            iip=np.arange(mesh.nnode() * mesh.ndim()),
            elem_elastic=[0, 1, 2, 6, 7, 8],
            elem_plastic=[3, 4, 5],
        )

        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(np.ones(nelem))
        system.setElastic(np.ones(6), np.ones(6))
        system.setPlastic(np.ones(3), np.ones(3), [[100.0], [100.0], [100.0]])
        system.setDt(1.0)

        x = mesh.coor()
        v = np.zeros_like(x)

        for i in range(v.shape[0]):
            v[i, 0] = 0.1 * (x[i, 1] - x[0, 1])

        system.flowSteps(10, v)

        # displacement is added affinely in an elastic system:
        # there is not residual force -> the system stays uniform
        self.assertTrue(np.allclose(system.u(), 10 * v))
        self.assertTrue(np.allclose(system.t(), 10))

        system.timeSteps(10)

        self.assertTrue(np.allclose(system.u(), 10 * v))
        self.assertTrue(np.allclose(system.t(), 20))

    def test_damping_alpha_no_eta(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        nelem = mesh.nelem()
        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor=mesh.coor(),
            conn=mesh.conn(),
            dofs=mesh.dofsPeriodic(),
            iip=[],
            elem_elastic=[0, 1, 2, 6, 7, 8],
            elem_plastic=[3, 4, 5],
        )

        alpha = 1.2
        system.setMassMatrix(np.ones(nelem))
        system.setDampingMatrix(alpha * np.ones(nelem))
        system.setElastic(np.ones(6), np.ones(6))
        system.setPlastic(np.ones(3), np.ones(3), [[100.0], [100.0], [100.0]])
        system.setDt(1.0)

        system.setV(np.ones_like(mesh.coor()))
        assert np.allclose(system.vector().AsDofs(system.fdamp()), alpha)

    def test_damping_no_alpha_eta(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        conn = mesh.conn()
        nelem = mesh.nelem()
        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor=coor,
            conn=conn,
            dofs=mesh.dofsPeriodic(),
            iip=[],
            elem_elastic=[0, 1, 2, 6, 7, 8],
            elem_plastic=[3, 4, 5],
        )

        eta = 3.4
        system.setMassMatrix(np.ones(nelem))
        system.setEta(eta)
        system.setElastic(np.ones(6), np.ones(6))
        system.setPlastic(np.ones(3), np.ones(3), [[100.0], [100.0], [100.0]])
        system.setDt(1.0)

        f = np.zeros_like(coor)
        v = np.zeros_like(coor)

        v[conn[-3:, :], 0] = 2

        f[conn[:3, -2:], 0] = -eta
        f[conn[-3:, :2], 0] = eta

        system.setV(v)
        assert np.allclose(system.fdamp(), f)


    def test_damping_alpha_eta(self):

        mesh = GooseFEM.Mesh.Quad4.Regular(3, 3)
        coor = mesh.coor()
        conn = mesh.conn()
        nelem = mesh.nelem()
        system = FrictionQPotFEM.Generic2d.HybridSystem(
            coor=coor,
            conn=conn,
            dofs=mesh.dofsPeriodic(),
            iip=[],
            elem_elastic=[0, 1, 2, 6, 7, 8],
            elem_plastic=[3, 4, 5],
        )

        alpha = 1.2
        eta = 3.4
        system.setMassMatrix(np.ones(nelem))
        system.setEta(eta)
        system.setDampingMatrix(alpha * np.ones(nelem))
        system.setElastic(np.ones(6), np.ones(6))
        system.setPlastic(np.ones(3), np.ones(3), [[100.0], [100.0], [100.0]])
        system.setDt(1.0)

        f = np.zeros_like(coor)
        v = np.zeros_like(coor)

        v[conn[-3:, :], 0] = 2

        f[conn[-3:, :], 0] = 2 * alpha
        f[conn[:3, -2:], 0] += -eta
        f[conn[-3:, :2], 0] += eta

        system.setV(v)
        assert np.allclose(system.fdamp(), f)


if __name__ == "__main__":

    unittest.main()
