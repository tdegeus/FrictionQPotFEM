import GMatElastoPlasticQPot.Cartesian2d as GMat
import GMatTensor.Cartesian2d as gtens
import GooseFEM
import GooseMPL as gplt
import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["goose", "goose-latex"])


def ComputePerturbation(sigma_star_test, trigger, mat, quad, vector, K, Solver):

    fext = vector.AllocateNodevec(0.0)
    disp = vector.AllocateNodevec(0.0)

    # pre-stress
    Sigstar = quad.AllocateQtensor(2, 0.0)
    for q in range(nip):
        Sigstar[trigger, q, :, :] = sigma_star_test
        Sigstar[trigger, q, :, :] = sigma_star_test

    # strain, stress, tangent
    Eps = quad.AllocateQtensor(2, 0.0)
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
    mat.setStrain(Eps)
    Sig = mat.Stress()

    return GMat.Deviatoric(Eps[trigger, 0]), disp, Eps, Sig


def GetYieldSurface(E, gamma, dE, dgamma, epsy=0.5, N=100):

    # solve for "p = 0"
    a = dgamma ** 2
    b = 2 * gamma * dgamma
    c = gamma ** 2 + E ** 2 - epsy ** 2
    D = b ** 2 - 4 * a * c
    smax = (-b + np.sqrt(D)) / (2.0 * a)
    smin = (-b - np.sqrt(D)) / (2.0 * a)

    # solve for "s = 0"
    a = dE ** 2
    b = 2 * E * dE
    c = E ** 2 + gamma ** 2 - epsy ** 2
    D = b ** 2 - 4 * a * c
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
        a = dE ** 2
        b = 2 * E * dE
        c = E ** 2 + (gamma + s * dgamma) ** 2 - epsy ** 2
        D = b ** 2 - 4 * a * c
        P[0, i] = (-b + np.sqrt(D)) / (2.0 * a)
        P[1, i] = (-b - np.sqrt(D)) / (2.0 * a)

    return P, S


def ComputeEnergy(P, S, Eps, Sig, dV, Eps_s, Sig_s, Eps_p, Sig_p):

    dE = np.empty(P.size)

    for i, (p, s) in enumerate(zip(P.ravel(), S.ravel())):
        dE[i] = np.sum(gtens.A2_ddot_B2(Sig + p * Sig_p + s * Sig_s, p * Eps_p + s * Eps_s) * dV)

    return dE.reshape(P.shape)


# ------------------------
# initialise configuration
# ------------------------

# mesh
# ----

# define mesh
mesh = GooseFEM.Mesh.Quad4.FineLayer(21, 21)

# mesh dimensions
nelem = mesh.nelem()
nne = mesh.nne()
ndim = mesh.ndim()

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

iip = np.concatenate((dofs[nodesBot, 0], dofs[nodesBot, 1], dofs[nodesTop, 0], dofs[nodesTop, 1]))

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
nip = quad.nip()

# material definition
mat = GMat.Array2d([nelem, nip])

iden = np.zeros(mat.shape(), dtype="int")
iden[elastic, :] = 1
mat.setElastic(iden, 10.0, 1.0)

iden = np.zeros(mat.shape(), dtype="int")
iden[plastic, :] = 1
mat.setElastic(iden, 10.0, 1.0)

# solve
# -----

# strain, stress, tangent
ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
Sig = mat.Stress()
C = mat.Tangent()

# stiffness matrix
Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C)
K.assemble(Ke)

# residual force
fe = quad.Int_gradN_dot_tensor2_dV(Sig)
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
mat.setStrain(Eps)
Sig = mat.Stress()

# ----------------------
# Effect of perturbation
# ----------------------

elmap = mesh.roll(1)
nconfig = plastic[-1] - elmap[plastic[-1]]
N = len(plastic)
config = []

for i in range(nconfig):

    Sigstar_s = np.array([[0.0, +1.0], [+1.0, 0.0]])

    Sigstar_p = np.array([[+1.0, 0.0], [0.0, -1.0]])

    Epsstar_s, u_s, Eps_s, Sig_s = ComputePerturbation(
        Sigstar_s, plastic[i], mat, quad, vector, K, Solver
    )
    Epsstar_p, u_p, Eps_p, Sig_p = ComputePerturbation(
        Sigstar_p, plastic[i], mat, quad, vector, K, Solver
    )

    ret = {
        "u_p": u_p,
        "u_s": u_s,
        "Eps_p": Eps_p,
        "Eps_s": Eps_s,
        "Sig_p": Sig_p,
        "Sig_s": Sig_s,
        "elemmap": [],
        "nodemap": [],
    }

    for roll in range(int(N / nconfig)):
        elemmap = mesh.roll(roll)
        ret["elemmap"] += [elemmap]
        ret["nodemap"] += [GooseFEM.Mesh.elemmap2nodemap(elemmap, coor, conn)]

    config += [ret]

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

    # Check with perturbation that is rolled periodically
    iconfig = int(itrigger % nconfig)
    roll = int((itrigger - itrigger % nconfig) / nconfig)
    elemmap = config[iconfig]["elemmap"][roll]
    nodemap = config[iconfig]["nodemap"][roll]
    assert np.allclose(config[iconfig]["Eps_s"][elemmap, :], Eps_s)
    assert np.allclose(config[iconfig]["Eps_p"][elemmap, :], Eps_p)
    assert np.allclose(config[iconfig]["Sig_s"][elemmap, :], Sig_s)
    assert np.allclose(config[iconfig]["Sig_p"][elemmap, :], Sig_p)
    assert np.allclose(config[iconfig]["u_s"][nodemap, :], u_s)
    assert np.allclose(config[iconfig]["u_p"][nodemap, :], u_p)

    # Current state for triggered element
    Epsd = GMat.Deviatoric(Eps[trigger, 0])
    gamma = Epsd[0, 1]
    E = Epsd[0, 0]

    # Find which (s, p) lie on the yield surface,
    # and to which energy change those perturbations lead
    Py, Sy = GetYieldSurface(E, gamma, Epsstar_p[0, 0], Epsstar_s[0, 1])
    Ey = ComputeEnergy(Py, Sy, Eps, Sig, quad.dV(), Eps_s, Sig_s, Eps_p, Sig_p)

    # Plot perturbed configuration

    smin = Sy.ravel()[np.argmin(Ey)]
    pmin = Py.ravel()[np.argmin(Ey)]
    Barrier += [np.min(Ey)]

    u = disp + pmin * u_p + smin * u_s
    ue = vector.AsElement(u)
    mat.setStrain(quad.SymGradN_vector(ue))
    sigeq = GMat.Sigd(np.average(mat.Stress(), weights=quad.AsTensor(2, quad.dV()), axis=1))

    fig, ax = plt.subplots()

    gplt.patch(coor=coor + u, conn=conn, cindex=sigeq, cmap="Reds", axis=ax, clim=(0, 1.0))

    ax.axis("equal")
    plt.axis("off")

    fig.savefig(f"example_layer_config-perturbed_{itrigger:d}.pdf")
    plt.close(fig)


fig, ax = plt.subplots()
ax.plot(np.arange(len(Barrier)), Barrier)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\Delta E$")
fig.savefig("example_layer_barriers.pdf")
plt.close(fig)
