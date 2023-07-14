import unittest

import FrictionQPotFEM
import FrictionQPotFEM.UniformSingleLayer2d as model
import GMatElastoPlasticQPot.Cartesian2d as GMat
import GMatTensor.Cartesian2d as gtens
import GooseFEM
import numpy as np


def ComputePerturbation(sigma_star_test, trigger, mat, quad, vector, K, Solver):
    fext = np.zeros(vector.shape_nodevec())
    disp = np.zeros(vector.shape_nodevec())

    # pre-stress
    Sigstar = np.zeros(quad.shape_qtensor(2))
    for q in range(quad.nip):
        Sigstar[trigger, q, :, :] = sigma_star_test
        Sigstar[trigger, q, :, :] = sigma_star_test

    # strain, stress, tangent
    Eps = np.zeros(quad.shape_qtensor(2))
    Sig = -Sigstar

    # residual force
    fe = quad.Int_gradN_dot_tensor2_dV(Sig)
    fint = vector.AssembleNode(fe)
    fext = vector.Copy_p(fint, fext)
    fres = fext - fint

    # solve
    disp = Solver.Solve(K, fres, disp)

    # post-process
    # ------------

    ue = vector.AsElement(disp)
    Eps = quad.SymGradN_vector(ue)
    mat.Eps = Eps

    return gtens.Deviatoric(Eps[trigger, 0]), disp, np.copy(mat.Eps), np.copy(mat.Sig)


def GetYieldSurface(E, gamma, dE, dgamma, epsy=0.5, N=100):
    # solve for "p = 0"
    a = dgamma**2
    b = 2 * gamma * dgamma
    c = gamma**2 + E**2 - epsy**2
    D = b**2 - 4 * a * c
    smax = (-b + np.sqrt(D)) / (2.0 * a)
    smin = (-b - np.sqrt(D)) / (2.0 * a)

    # solve for "s = 0"
    a = dE**2
    b = 2 * E * dE
    c = E**2 + gamma**2 - epsy**2
    D = b**2 - 4 * a * c
    pmax = (-b + np.sqrt(D)) / (2.0 * a)
    pmin = (-b - np.sqrt(D)) / (2.0 * a)

    # add to list
    S = np.empty((2, N))
    P = np.empty((2, N))
    n = int(-smin / (smax - smin) * N)
    S[:, :n] = np.linspace(smin, 0, n).reshape(1, -1)
    S[:, n:] = np.linspace(0, smax, (N - n) + 1)[1:].reshape(1, -1)
    P[0, n - 1] = pmax
    P[1, n - 1] = pmin
    P[:, 0] = 0
    P[:, -1] = 0

    # solve for fixed "s"
    for i, s in enumerate(S[0, :-1]):
        if i in [0, n - 1]:
            continue
        a = dE**2
        b = 2 * E * dE
        c = E**2 + (gamma + s * dgamma) ** 2 - epsy**2
        D = b**2 - 4 * a * c
        P[0, i] = (-b + np.sqrt(D)) / (2.0 * a)
        P[1, i] = (-b - np.sqrt(D)) / (2.0 * a)

    return P, S


def ComputeEnergy(P, S, Eps, Sig, dV, Eps_s, Sig_s, Eps_p, Sig_p, e):
    dE = np.empty(P.size)

    for i, (p, s) in enumerate(zip(P.ravel(), S.ravel())):
        dE[i] = np.sum(gtens.A2_ddot_B2(Sig + p * Sig_p + s * Sig_s, p * Eps_p + s * Eps_s) * dV)

    return (dE / np.sum(dV[e, :])).reshape(P.shape)


class test_Generic2d(unittest.TestCase):
    """
    Tests
    """

    def test_basic(self):
        # ------------------------
        # initialise configuration
        # ------------------------

        # mesh
        # ----

        # define mesh
        mesh = GooseFEM.Mesh.Quad4.FineLayer(21, 21)

        # mesh dimensions
        nelem = mesh.nelem
        mesh.nne
        mesh.ndim

        # mesh definitions
        coor = mesh.coor()
        conn = mesh.conn()
        dofs = mesh.dofs()

        # node sets
        nodesLft = mesh.nodesLeftOpenEdge()
        nodesRgt = mesh.nodesRightOpenEdge()
        nodesTop = mesh.nodesTopEdge()
        nodesBot = mesh.nodesBottomEdge()

        # element sets
        plastic = mesh.elementsMiddleLayer()
        elastic = np.setdiff1d(np.arange(nelem), plastic)

        # periodicity and fixed displacements DOFs
        # ----------------------------------------

        dofs[nodesRgt, :] = dofs[nodesLft, :]

        dofs = GooseFEM.Mesh.renumber(dofs)

        iip = np.concatenate(
            (dofs[nodesBot, 0], dofs[nodesBot, 1], dofs[nodesTop, 0], dofs[nodesTop, 1])
        )

        # simulation variables
        # --------------------

        # vector definition
        vector = GooseFEM.VectorPartitioned(conn, dofs, iip)

        # allocate system matrix
        K = GooseFEM.MatrixPartitioned(conn, dofs, iip)
        Solver = GooseFEM.MatrixPartitionedSolver()

        # nodal quantities
        disp = np.zeros(coor.shape)
        fint = np.zeros(coor.shape)
        fext = np.zeros(coor.shape)
        fres = np.zeros(coor.shape)

        # element/material definition
        # ---------------------------

        # element definition
        quad = GooseFEM.Element.Quad4.Quadrature(vector.AsElement(coor))

        # material definition
        mat = GMat.Elastic2d(10 * np.ones([nelem, quad.nip]), np.ones([nelem, quad.nip]))

        # solve
        # -----

        # strain, stress, tangent
        ue = vector.AsElement(disp)
        mat.Eps = quad.SymGradN_vector(ue)

        # stiffness matrix
        Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(mat.C)
        K.assemble(Ke)

        # residual force
        fe = quad.Int_gradN_dot_tensor2_dV(mat.Sig)
        fint = vector.AssembleNode(fe)
        fext = vector.Copy_p(fint, fext)
        fres = fext - fint

        # set fixed displacements
        disp[nodesTop, 0] = +5.0
        disp[nodesTop, 1] = np.sin(np.linspace(0, 2.0 * np.pi, len(nodesTop)) - np.pi)
        disp[nodesBot, 1] = np.cos(np.linspace(0, 2.0 * np.pi, len(nodesBot)) - np.pi)

        # solve
        disp = Solver.Solve(K, fres, disp)

        # compute new state
        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        mat.Eps = Eps
        Sig = np.copy(mat.Sig)

        # ----------------------
        # Effect of perturbation
        # ----------------------

        epsy = np.cumsum(np.ones(50 * plastic.size).reshape(plastic.size, -1), axis=0)

        sys = model.System(
            coor=coor,
            conn=conn,
            dofs=dofs,
            iip=iip,
            elastic_elem=elastic,
            elastic_K=10 * FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            elastic_G=FrictionQPotFEM.moduli_toquad(np.ones(elastic.size)),
            plastic_elem=plastic,
            plastic_K=10 * FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_G=FrictionQPotFEM.moduli_toquad(np.ones(plastic.size)),
            plastic_epsy=FrictionQPotFEM.epsy_initelastic_toquad(epsy),
            dt=1,
            rho=1,
            alpha=1,
            eta=0,
        )

        modelTrigger = model.LocalTriggerFineLayerFull(sys)
        modelTrigger.setState(Eps, Sig, 0.5 * np.ones((len(plastic), 4)), 100)

        Barrier = []

        for itrigger, trigger in enumerate(plastic):
            Sigstar_s = np.array([[0.0, +1.0], [+1.0, 0.0]])

            Sigstar_p = np.array([[+1.0, 0.0], [0.0, -1.0]])

            Epsstar_s, u_s, Eps_s, Sig_s = ComputePerturbation(
                Sigstar_s, trigger, mat, quad, vector, K, Solver
            )

            Epsstar_p, u_p, Eps_p, Sig_p = ComputePerturbation(
                Sigstar_p, trigger, mat, quad, vector, K, Solver
            )

            self.assertTrue(np.allclose(u_s, modelTrigger.u_s(itrigger)))
            self.assertTrue(np.allclose(u_p, modelTrigger.u_p(itrigger)))
            self.assertTrue(np.allclose(Eps_s, modelTrigger.Eps_s(itrigger)))
            self.assertTrue(np.allclose(Eps_p, modelTrigger.Eps_p(itrigger)))
            self.assertTrue(np.allclose(Sig_s, modelTrigger.Sig_s(itrigger)))
            self.assertTrue(np.allclose(Sig_p, modelTrigger.Sig_p(itrigger)))

            # Current state for triggered element
            Epsd = gtens.Deviatoric(Eps[trigger, 0])
            gamma = Epsd[0, 1]
            E = Epsd[0, 0]

            # Find which (s, p) lie on the yield surface,
            # and to which energy change those perturbations lead
            Py, Sy = GetYieldSurface(E, gamma, Epsstar_p[0, 0], Epsstar_s[0, 1])
            Ey = ComputeEnergy(
                Py, Sy, Eps, Sig, quad.dV, Eps_s, Sig_s, Eps_p, Sig_p, plastic[itrigger]
            )

            # Plot perturbed configuration

            smin = Sy.ravel()[np.argmin(Ey)]
            pmin = Py.ravel()[np.argmin(Ey)]
            Barrier += [np.min(Ey)]

            delta_u = pmin * u_p + smin * u_s
            self.assertTrue(np.allclose(delta_u, modelTrigger.delta_u(itrigger, 0)))

        Barrier = np.array(Barrier)
        self.assertTrue(np.allclose(Barrier, modelTrigger.barriers[:, 0]))


if __name__ == "__main__":
    unittest.main()
