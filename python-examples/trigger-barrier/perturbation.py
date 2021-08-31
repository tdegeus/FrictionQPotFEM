import GooseFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import GooseMPL as gplt
import GMatTensor.Cartesian2d as gtens

plt.style.use(["goose", "goose-latex"])


def ComputePerturbation(sigma_star_test):

    # mesh
    # ----

    # define mesh
    mesh = GooseFEM.Mesh.Quad4.FineLayer(13, 13)

    # mesh dimensions
    nelem = mesh.nelem()

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

    # strain
    ue = vector.AsElement(disp)
    Eps = quad.SymGradN_vector(ue)

    # pre-stress
    trigger = plastic[int(len(plastic) / 2.0)]
    Sigstar = quad.AllocateQtensor(2, 0.0)
    for q in range(nip):
        Sigstar[trigger, q, :, :] = sigma_star_test
        Sigstar[trigger, q, :, :] = sigma_star_test

    # stress & tangent
    mat.setStrain(Eps)
    Sig = mat.Stress() - Sigstar
    C = mat.Tangent()

    # stiffness matrix
    Ke = quad.Int_gradN_dot_tensor4_dot_gradNT_dV(C)
    K.assemble(Ke)

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

    return (
        GMat.Deviatoric(Eps[trigger, 0]),
        trigger,
        disp,
        coor,
        conn,
        vector,
        quad,
        mat,
    )


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


# Effect of perturbation
# ----------------------

Sigstar_s = np.array([[0.0, +1.0], [+1.0, 0.0]])

Sigstar_p = np.array([[+1.0, 0.0], [0.0, -1.0]])

Epsstar_s, trigger, u_s, coor, conn, vector, quad, mat = ComputePerturbation(Sigstar_s)
Epsstar_p, trigger, u_p, coor, conn, vector, quad, mat = ComputePerturbation(Sigstar_p)

# Plot - simple shear
# -------------------

ue = vector.AsElement(u_s)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
energy = np.average(mat.Energy(), weights=quad.dV(), axis=1)

fig, ax = plt.subplots()

gplt.patch(
    coor=coor + u_s, conn=conn, cindex=energy, cmap="RdGy_r", axis=ax, clim=(-0.1, 0.1)
)
gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)

ax.axis("equal")
plt.axis("off")

sm = plt.cm.ScalarMappable(
    cmap="RdGy_r", norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1)
)
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels([f"{i:.1f}" for i in labels])

fig.savefig("perturbation_simple-shear_pos.pdf")
plt.close(fig)

# Plot - pure shear
# -----------------

ue = vector.AsElement(u_p)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
energy = np.average(mat.Energy(), weights=quad.dV(), axis=1)

fig, ax = plt.subplots()

gplt.patch(
    coor=coor + u_p, conn=conn, cindex=energy, cmap="RdGy_r", axis=ax, clim=(-0.1, 0.1)
)
gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)

ax.axis("equal")
plt.axis("off")

sm = plt.cm.ScalarMappable(
    cmap="RdGy_r", norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1)
)
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels([f"{i:.1f}" for i in labels])

fig.savefig("perturbation_pure-shear_pos.pdf")
plt.close(fig)

# Plot - simple shear
# -------------------

ue = vector.AsElement(-u_s)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
energy = np.average(mat.Energy(), weights=quad.dV(), axis=1)

fig, ax = plt.subplots()

gplt.patch(
    coor=coor - u_s, conn=conn, cindex=energy, cmap="RdGy_r", axis=ax, clim=(-0.1, 0.1)
)
gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)

ax.axis("equal")
plt.axis("off")

sm = plt.cm.ScalarMappable(
    cmap="RdGy_r", norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1)
)
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels([f"{i:.1f}" for i in labels])

fig.savefig("perturbation_simple-shear_neg.pdf")
plt.close(fig)

# Plot - pure shear
# -----------------

ue = vector.AsElement(-u_p)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
energy = np.average(mat.Energy(), weights=quad.dV(), axis=1)

fig, ax = plt.subplots()

gplt.patch(
    coor=coor - u_p, conn=conn, cindex=energy, cmap="RdGy_r", axis=ax, clim=(-0.1, 0.1)
)
gplt.patch(coor=coor, conn=conn, linestyle="--", axis=ax)

ax.axis("equal")
plt.axis("off")

sm = plt.cm.ScalarMappable(
    cmap="RdGy_r", norm=mpl.colors.Normalize(vmin=-0.1, vmax=+0.1)
)
sm.set_array([])

ticks = [-0.1, 0.0, +0.1]
labels = [-0.1, 0.0, +0.1]
cbar = plt.colorbar(sm, ticks=ticks)
cbar.ax.set_yticklabels([f"{i:.1f}" for i in labels])

fig.savefig("perturbation_pure-shear_neg.pdf")
plt.close(fig)

# Perturbations to the yield surface
# ----------------------------------

# Effect of perturbations
ue = vector.AsElement(u_s)
Eps_s = quad.SymGradN_vector(ue)
mat.setStrain(Eps_s)
Sig_s = mat.Stress()

ue = vector.AsElement(u_p)
Eps_p = quad.SymGradN_vector(ue)
mat.setStrain(Eps_p)
Sig_p = mat.Stress()

# Current state
ue = vector.AsElement(0 * u_p)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
Sig = mat.Stress()

Epsd = GMat.Deviatoric(Eps[trigger, 0])
gamma = Epsd[0, 1]
E = Epsd[0, 0]

# Find which (s, p) lie on the yield surface, and to which energy change those perturbations lead
Py, Sy = GetYieldSurface(E, gamma, Epsstar_p[0, 0], Epsstar_s[0, 1])
Ey = ComputeEnergy(Py, Sy, Eps, Sig, quad.dV(), Eps_s, Sig_s, Eps_p, Sig_p)

# Phase diagram for strain, stress, and energy
# --------------------------------------------

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

        dEps = s * Eps_s + p * Eps_p
        dSig = s * Sig_s + p * Sig_p

        disp = s * u_s + p * u_p
        ue = vector.AsElement(disp)
        mat.setStrain(Eps_n)

        assert np.allclose(Eps_n, quad.SymGradN_vector(ue))
        assert np.allclose(Sig_n, mat.Stress())
        assert np.isclose(
            energy[i, j] + E0, np.average(mat.Energy(), weights=dV) * np.sum(dV)
        )
        assert np.isclose(
            energy[i, j],
            np.average(gtens.A2_ddot_B2(Sig + 0.5 * dSig, dEps), weights=dV)
            * np.sum(dV),
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

fig.savefig("perturbation_phase-diagram_sig.pdf")
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

fig.savefig("perturbation_phase-diagram_eps.pdf")
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

fig.savefig("perturbation_phase-diagram_energy.pdf")
plt.close(fig)

# Plot phase diagram - energy

fig, ax = plt.subplots()

h = ax.contourf(S, P, energy)

ax.plot([0, 0], [P[0], P[-1]], c="w", lw=1)
ax.plot([S[0], S[-1]], [0, 0], c="w", lw=1)

ax.plot(Sy[0, :], Py[0, :], c="w")
ax.plot(Sy[1, :], Py[1, :], c="w")

ax.plot(Sy.ravel()[np.argmin(Ey)], Py.ravel()[np.argmin(Ey)], c="r", marker="o")

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$p$")

fig.savefig("perturbation_phase-diagram_energy-contour.pdf")
plt.close(fig)

# Plot energy change for different perturbations

fig, ax = plt.subplots()

ax.plot(Py[0, :], Ey[0, :])
ax.plot(Py[1, :], Ey[1, :])

ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$\Delta E$")

fig.savefig("perturbation_yield-surface_delta-E.pdf")
plt.close(fig)

plt.show()
