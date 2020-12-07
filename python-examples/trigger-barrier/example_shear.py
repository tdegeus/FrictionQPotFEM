import GooseFEM
import GMatElastoPlasticQPot.Cartesian2d as GMat
import numpy as np
import matplotlib.pyplot as plt
import GooseMPL as gplt

plt.style.use(['goose', 'goose-latex'])


def InitConfig():

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

    iip = np.concatenate((
        dofs[nodesBot, 0],
        dofs[nodesBot, 1],
        dofs[nodesTop, 0],
        dofs[nodesTop, 1]
    ))

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

    I = np.zeros(mat.shape(), dtype='int')
    I[elastic, :] = 1
    mat.setElastic(I, 10.0, 1.0)

    I = np.zeros(mat.shape(), dtype='int')
    I[plastic, :] = 1
    mat.setElastic(I, 10.0, 1.0)

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
    dV = quad.dV()
    E = np.average(mat.Energy(), weights=dV) * np.sum(dV)

    return Eps, disp, E


def CheckEquilibrium(disp):

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

    iip = np.concatenate((
        dofs[nodesBot, 0],
        dofs[nodesBot, 1],
        dofs[nodesTop, 0],
        dofs[nodesTop, 1]
    ))

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

    I = np.zeros(mat.shape(), dtype='int')
    I[elastic, :] = 1
    mat.setElastic(I, 10.0, 1.0)

    I = np.zeros(mat.shape(), dtype='int')
    I[plastic, :] = 1
    mat.setElastic(I, 10.0, 1.0)

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

    # solve
    u = Solver.Solve(K, fres, disp)

    assert np.isclose(np.linalg.norm(fres), 0)
    assert np.isclose(np.linalg.norm(disp - u), 0)


def ComputePerturbation(sigma_star_test):

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

    iip = np.concatenate((
        dofs[nodesBot, 0],
        dofs[nodesBot, 1],
        dofs[nodesTop, 0],
        dofs[nodesTop, 1]
    ))

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

    I = np.zeros(mat.shape(), dtype='int')
    I[elastic, :] = 1
    mat.setElastic(I, 10.0, 1.0)

    I = np.zeros(mat.shape(), dtype='int')
    I[plastic, :] = 1
    mat.setElastic(I, 10.0, 1.0)

    # solve
    # -----

    # pre-stress
    trigger = plastic[int(len(plastic) / 2.0)]
    Sigstar = quad.AllocateQtensor(2, 0.0)
    for q in range(nip):
        Sigstar[trigger, q, :, :] = sigma_star_test
        Sigstar[trigger, q, :, :] = sigma_star_test

    # strain, stress, tangent
    ue = vector.AsElement(disp)
    Eps = quad.SymGradN_vector(ue)
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

    return GMat.Deviatoric(Eps[trigger, 0]), disp, trigger, vector, quad, mat, coor, conn

# ----------------------
# Effect of perturbation
# ----------------------

Sigstar_s = np.array([
    [ 0.0, +1.0],
    [+1.0,  0.0]])

Sigstar_p = np.array([
    [+1.0,  0.0],
    [ 0.0, -1.0]])

Epsdstar_s, u_s, trigger, vector, quad, mat, coor, conn = ComputePerturbation(Sigstar_s)
Epsdstar_p, u_p, trigger, vector, quad, mat, coor, conn = ComputePerturbation(Sigstar_p)

# --------------------------------
# Explore different configurations
# --------------------------------

Eps0, u0, Energy0 = InitConfig()

sig = np.zeros((201, 201))
eps = np.zeros(sig.shape)
energy = np.zeros(sig.shape)
dV = quad.dV()
P = np.linspace(-2, 2, sig.shape[0])
S = np.linspace(-2, 2, sig.shape[1])
S1 = np.NaN * np.ones((sig.shape[0]))
S2 = np.NaN * np.ones((sig.shape[0]))

dgamma = Epsdstar_s[0, 1]
dE = Epsdstar_p[0, 0]

Epsd = GMat.Deviatoric(Eps0[trigger, 0])
gamma = Epsd[0, 1]
E = Epsd[0, 0]

epsy = 0.4

for i, p in enumerate(P):

    a = dgamma ** 2
    b = 2 * gamma * dgamma
    c = gamma ** 2 + (E + p * dE) ** 2 - epsy ** 2
    D = b ** 2 - 4 * a * c
    if D >= 0:
        S1[i] = (-b + np.sqrt(D)) / (2 * a)
        S2[i] = (-b - np.sqrt(D)) / (2 * a)

        disp = u0 + S1[i] * u_s + p * u_p
        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        assert np.isclose(float(GMat.Epsd(Eps[trigger, 0, :, :])), epsy)

        disp = u0 + S2[i] * u_s + p * u_p
        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        assert np.isclose(float(GMat.Epsd(Eps[trigger, 0, :, :])), epsy)

    for j, s in enumerate(S):

        disp = u0 + s * u_s + p * u_p
        ue = vector.AsElement(disp)
        Eps = quad.SymGradN_vector(ue)
        mat.setStrain(Eps)
        Sig = mat.Stress()
        sig[i, j] = GMat.Sigd(Sig[trigger, 0])
        eps[i, j] = GMat.Epsd(Eps[trigger, 0])
        energy[i, j] = np.average(mat.Energy(), weights=dV) * np.sum(dV) - Energy0

e = np.zeros((2, len(S1)))

for i, (p, s) in enumerate(zip(P, S1)):
    disp = u0 + s * u_s + p * u_p
    ue = vector.AsElement(disp)
    Eps = quad.SymGradN_vector(ue)
    mat.setStrain(Eps)
    e[0, i] = np.average(mat.Energy(), weights=dV) * np.sum(dV) - Energy0

for i, (p, s) in enumerate(zip(P, S2)):
    disp = u0 + s * u_s + p * u_p
    ue = vector.AsElement(disp)
    Eps = quad.SymGradN_vector(ue)
    mat.setStrain(Eps)
    e[1, i] = np.average(mat.Energy(), weights=dV) * np.sum(dV) - Energy0

imin, jmin = np.unravel_index(np.argmin(e), e.shape)
pmin = P[jmin]
smin = S1[jmin] if imin == 0 else S2[jmin]

# Plot phase diagram - stress

fig, ax = plt.subplots()

h = ax.imshow(sig, cmap='jet', extent=[np.min(P), np.max(P), np.min(S), np.max(S)])

ax.plot(S1, P, c='w')
ax.plot(S2, P, c='w')

ax.plot([0, 0], [P[0], P[-1]], c='w', lw=1)
ax.plot([S[0], S[-1]], [0, 0], c='w', lw=1)

ax.plot(smin, pmin, c='r', marker='o')

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('example_shear_phase-diagram_sig.pdf')
plt.close(fig)

# Plot phase diagram - strain

fig, ax = plt.subplots()

h = ax.imshow(eps, cmap='jet', extent=[np.min(P), np.max(P), np.min(S), np.max(S)])

ax.plot(S1, P, c='w')
ax.plot(S2, P, c='w')

ax.plot([0, 0], [P[0], P[-1]], c='w', lw=1)
ax.plot([S[0], S[-1]], [0, 0], c='w', lw=1)

ax.plot(smin, pmin, c='r', marker='o')

cbar = fig.colorbar(h, aspect=10)

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('example_shear_phase-diagram_eps.pdf')
plt.close(fig)

# Plot phase diagram - energy

fig, ax = plt.subplots()

h = ax.imshow(energy, cmap='jet', extent=[np.min(P), np.max(P), np.min(S), np.max(S)])

cbar = fig.colorbar(h, aspect=10)

ax.plot(S1, P, c='w')
ax.plot(S2, P, c='w')

ax.plot([0, 0], [P[0], P[-1]], c='w', lw=1)
ax.plot([S[0], S[-1]], [0, 0], c='w', lw=1)

ax.plot(smin, pmin, c='r', marker='o')

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('example_shear_phase-diagram_energy.pdf')
plt.close(fig)

# Plot phase diagram - energy contours

fig, ax = plt.subplots()

h = ax.contourf(S, P, energy)

cbar = fig.colorbar(h, aspect=10)

ax.plot(S1, P, c='w')
ax.plot(S2, P, c='w')

ax.plot([0, 0], [P[0], P[-1]], c='w', lw=1)
ax.plot([S[0], S[-1]], [0, 0], c='w', lw=1)

ax.plot(smin, pmin, c='r', marker='o')

ax.set_xlabel(r'$s$')
ax.set_ylabel(r'$p$')

fig.savefig('example_shear_phase-diagram_energy-contour.pdf')
plt.close(fig)

# Plot initial configuration

ue = vector.AsElement(u0)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
sigeq = GMat.Sigd(np.average(mat.Stress(), weights=quad.AsTensor(2, quad.dV()), axis=1))

fig, ax = plt.subplots()

gplt.patch(coor=coor + u0, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 1.0))

ax.axis('equal')
plt.axis('off')

fig.savefig('example_shear_config.pdf')
plt.close(fig)

# Plot perturbed configuration

ue = vector.AsElement(u0 + pmin * u_p + smin * u_s)
Eps = quad.SymGradN_vector(ue)
mat.setStrain(Eps)
sigeq = GMat.Sigd(np.average(mat.Stress(), weights=quad.AsTensor(2, quad.dV()), axis=1))

fig, ax = plt.subplots()

gplt.patch(coor=coor + u0, conn=conn, cindex=sigeq, cmap='Reds', axis=ax, clim=(0, 1.0))

ax.axis('equal')
plt.axis('off')

fig.savefig('example_shear_config-perturbed.pdf')
plt.close(fig)
