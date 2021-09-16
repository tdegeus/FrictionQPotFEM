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
        dE[i] = np.sum(
            gtens.A2_ddot_B2(Sig + p * Sig_p + s * Sig_s, p * Eps_p + s * Eps_s) * dV
        )

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

trigger = plastic[int(len(plastic) / 2.0)]

Sigstar_s = np.array([[0.0, +1.0], [+1.0, 0.0]])

Sigstar_p = np.array([[+1.0, 0.0], [0.0, -1.0]])

Epsstar_s, u_s, Eps_s, Sig_s = ComputePerturbation(
    Sigstar_s, trigger, mat, quad, vector, K, Solver
)
Epsstar_p, u_p, Eps_p, Sig_p = ComputePerturbation(
    Sigstar_p, trigger, mat, quad, vector, K, Solver
)

# Current state for triggered element
Epsd = GMat.Deviatoric(Eps[trigger, 0])
gamma = Epsd[0, 1]
E = Epsd[0, 0]

# Find which (s, p) lie on the yield surface, and to which energy change those perturbations lead
Py, Sy = GetYieldSurface(E, gamma, Epsstar_p[0, 0], Epsstar_s[0, 1])
Ey = ComputeEnergy(Py, Sy, Eps, Sig, quad.dV(), Eps_s, Sig_s, Eps_p, Sig_p)

# --------------------------------
# Explore different configurations
# --------------------------------

dV = quad.dV()
sig = np.zeros((201, 201))
eps = np.zeros(sig.shape)
energy = np.zeros(sig.shape)
P = np.linspace(-3, 3, sig.shape[0])
S = np.linspace(-3, 3, sig.shape[1])
E0 = np.average(0.5 * gtens.A2_ddot_B2(Sig, Eps), weights=dV) * np.sum(dV)

for i, p in enumerate(P):
    for j, s in enumerate(S):

        Eps_n = Eps + s * Eps_s + p * Eps_p
        Sig_n = Sig + s * Sig_s + p * Sig_p

        sig[i, j] = GMat.Sigd(Sig_n[trigger, 0])
        eps[i, j] = GMat.Epsd(Eps_n[trigger, 0])
        energy[i, j] = (
            np.average(0.5 * gtens.A2_ddot_B2(Sig_n, Eps_n), weights=dV) * np.sum(dV)
            - E0
        )

# Plot phase diagram - stress

fig, ax = plt.subplots()

h = ax.imshow(sig, cmap="jet", extent=[np.min(P), np.max(P), np.min(S), np.max(S)])

ax.plot([0, 0], [P[0], P[-1]], c="w", lw=1)
ax.plot([S[0], S[-1]], [0, 0], c="w", lw=1)

ax.plot(Sy[0, :], Py[0, :], c="w")
ax.plot(Sy[1, :], Py[1, :], c="w")

ax.plot(Sy.ravel()[np.argmin(Ey)], Py.ravel()[np.argmin(Ey)], c="r", marker="o")

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$p$")

fig.savefig("example_shear_phase-diagram_sig.pdf")
plt.close(fig)

# Plot phase diagram - strain

fig, ax = plt.subplots()

h = ax.imshow(eps, cmap="jet", extent=[np.min(P), np.max(P), np.min(S), np.max(S)])

ax.plot([0, 0], [P[0], P[-1]], c="w", lw=1)
ax.plot([S[0], S[-1]], [0, 0], c="w", lw=1)

ax.plot(Sy[0, :], Py[0, :], c="w")
ax.plot(Sy[1, :], Py[1, :], c="w")

ax.plot(Sy.ravel()[np.argmin(Ey)], Py.ravel()[np.argmin(Ey)], c="r", marker="o")

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$p$")

fig.savefig("example_shear_phase-diagram_eps.pdf")
plt.close(fig)

# Plot phase diagram - energy

fig, ax = plt.subplots()

h = ax.imshow(energy, cmap="jet", extent=[np.min(P), np.max(P), np.min(S), np.max(S)])

ax.plot([0, 0], [P[0], P[-1]], c="w", lw=1)
ax.plot([S[0], S[-1]], [0, 0], c="w", lw=1)

ax.plot(Sy[0, :], Py[0, :], c="w")
ax.plot(Sy[1, :], Py[1, :], c="w")

ax.plot(Sy.ravel()[np.argmin(Ey)], Py.ravel()[np.argmin(Ey)], c="r", marker="o")

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$p$")

fig.savefig("example_shear_phase-diagram_energy.pdf")
plt.close(fig)

# Plot phase diagram - energy contours

fig, ax = plt.subplots()

h = ax.contourf(S, P, energy)

cbar = fig.colorbar(h, aspect=10)

ax.plot([0, 0], [P[0], P[-1]], c="w", lw=1)
ax.plot([S[0], S[-1]], [0, 0], c="w", lw=1)

ax.plot(Sy[0, :], Py[0, :], c="w")
ax.plot(Sy[1, :], Py[1, :], c="w")

ax.plot(Sy.ravel()[np.argmin(Ey)], Py.ravel()[np.argmin(Ey)], c="r", marker="o")

ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$p$")

fig.savefig("example_shear_phase-diagram_energy-contour.pdf")
plt.close(fig)

# Plot initial configuration

sigeq = GMat.Sigd(np.average(Sig, weights=quad.AsTensor(2, quad.dV()), axis=1))

fig, ax = plt.subplots()

gplt.patch(
    coor=coor + disp, conn=conn, cindex=sigeq, cmap="Reds", axis=ax, clim=(0, 1.0)
)

ax.axis("equal")
plt.axis("off")

fig.savefig("example_shear_config.pdf")
plt.close(fig)

# Plot perturbed configuration

smin = Sy.ravel()[np.argmin(Ey)]
pmin = Py.ravel()[np.argmin(Ey)]

disp += pmin * u_p + smin * u_s
ue = vector.AsElement(disp)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
sigeq = GMat.Sigd(np.average(mat.Stress(), weights=quad.AsTensor(2, quad.dV()), axis=1))

fig, ax = plt.subplots()

gplt.patch(
    coor=coor + disp, conn=conn, cindex=sigeq, cmap="Reds", axis=ax, clim=(0, 1.0)
)

ax.axis("equal")
plt.axis("off")

fig.savefig("example_shear_config-perturbed.pdf")
plt.close(fig)
